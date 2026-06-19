from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import prepare_audio_for_ml
from .diarization import SpeakerTurn
from .torch_compat import ensure_torch_amp_compatibility

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterEmbedding:
    embedding_model: str
    embedding: list[float]
    sample_count: int
    total_duration: float


@dataclass(frozen=True)
class ProfileEmbedding:
    profile_id: str
    display_name: str
    embedding_model: str
    embedding: list[float]


@dataclass(frozen=True)
class ClusterMatch:
    cluster_label: str
    profile_id: str | None
    display_name: str | None
    confidence: float | None
    ambiguous: bool
    sample_count: int
    total_duration: float


class InsufficientSpeakerAudioError(ValueError):
    pass


class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        ensure_torch_amp_compatibility()
        from speechbrain.inference.speaker import SpeakerRecognition

        logger.info("Loading speaker embedding model: %s", self.model_name)
        self._model = SpeakerRecognition.from_hparams(
            source=self.model_name,
            run_opts={"device": self.device},
        )
        return self._model

    def embed_cluster(
        self,
        audio_path: str,
        turns: list[SpeakerTurn],
        max_turns: int = 8,
        min_turn_duration: float = 1.5,
    ) -> ClusterEmbedding:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        candidates = sorted(
            [turn for turn in turns if turn.end - turn.start >= min_turn_duration],
            key=lambda turn: turn.end - turn.start,
            reverse=True,
        )[:max_turns]

        if not candidates:
            raise InsufficientSpeakerAudioError("No speaker turns long enough to embed")

        prepared_path = prepare_audio_for_ml(audio_path)
        model = self._load_model()
        # Decode the prepared (mono/16k) file ONCE, then slice each turn in
        # memory. Previously every turn re-loaded the whole file from disk
        # (up to max_turns times per cluster), which dominated latency on long
        # recordings and re-decoded hundreds of MB repeatedly.
        full_waveform, sample_rate = _load_full_mono_16k(str(prepared_path))
        embeddings = []
        total_duration = 0.0

        for turn in candidates:
            waveform = _slice_segment(full_waveform, sample_rate, turn.start, turn.end)
            if waveform is None:
                continue
            embedding = model.encode_batch(waveform, normalize=True)
            embeddings.append(_to_numpy_vector(embedding))
            total_duration += turn.end - turn.start

        if not embeddings:
            raise ValueError("Could not extract embeddings for any speaker turn")

        centroid = np.mean(np.stack(embeddings), axis=0)
        centroid = _normalize(centroid)

        return ClusterEmbedding(
            embedding_model=self.model_name,
            embedding=centroid.astype(np.float32).tolist(),
            sample_count=len(embeddings),
            total_duration=total_duration,
        )

    def identify_cluster(
        self,
        audio_path: str,
        cluster_label: str,
        turns: list[SpeakerTurn],
        profiles: list[ProfileEmbedding],
        threshold: float,
        ambiguity_margin: float,
    ) -> ClusterMatch:
        try:
            cluster_embedding = self.embed_cluster(audio_path, turns)
        except InsufficientSpeakerAudioError:
            return ClusterMatch(
                cluster_label=cluster_label,
                profile_id=None,
                display_name=None,
                confidence=None,
                ambiguous=False,
                sample_count=0,
                total_duration=sum(max(0.0, turn.end - turn.start) for turn in turns),
            )

        scores: list[tuple[ProfileEmbedding, float]] = []

        query = np.asarray(cluster_embedding.embedding, dtype=np.float32)
        for profile in profiles:
            if profile.embedding_model != cluster_embedding.embedding_model:
                continue
            candidate = np.asarray(profile.embedding, dtype=np.float32)
            scores.append((profile, cosine_similarity(query, candidate)))

        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores:
            return ClusterMatch(
                cluster_label=cluster_label,
                profile_id=None,
                display_name=None,
                confidence=None,
                ambiguous=False,
                sample_count=cluster_embedding.sample_count,
                total_duration=cluster_embedding.total_duration,
            )

        best_profile, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else None
        ambiguous = second_score is not None and (best_score - second_score) < ambiguity_margin

        if best_score < threshold or ambiguous:
            return ClusterMatch(
                cluster_label=cluster_label,
                profile_id=None,
                display_name=None,
                confidence=best_score,
                ambiguous=ambiguous,
                sample_count=cluster_embedding.sample_count,
                total_duration=cluster_embedding.total_duration,
            )

        return ClusterMatch(
            cluster_label=cluster_label,
            profile_id=best_profile.profile_id,
            display_name=best_profile.display_name,
            confidence=best_score,
            ambiguous=False,
            sample_count=cluster_embedding.sample_count,
            total_duration=cluster_embedding.total_duration,
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def group_turns_by_cluster(turns: list[SpeakerTurn]) -> dict[str, list[SpeakerTurn]]:
    grouped: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.cluster_label, []).append(turn)
    return grouped


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm


def _to_numpy_vector(embedding: Any) -> np.ndarray:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu()
    if hasattr(embedding, "numpy"):
        embedding = embedding.numpy()
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _load_full_mono_16k(audio_path: str) -> tuple[Any, int]:
    """Decode an audio file once as mono 16k float32. Returns (waveform, sample_rate)."""
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        sample_rate = 16000
    return waveform.to(dtype=torch.float32), sample_rate


def _slice_segment(waveform: Any, sample_rate: int, start: float, end: float) -> Any | None:
    """Slice [start, end] (seconds) out of an already-decoded mono waveform."""
    if end <= start:
        return None
    start_frame = max(0, int(start * sample_rate))
    end_frame = min(waveform.shape[-1], int(end * sample_rate))
    if end_frame <= start_frame:
        return None
    return waveform[:, start_frame:end_frame]
