from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import prepare_audio_for_ml

logger = logging.getLogger(__name__)

# Long-audio handling. pyannote's clustering builds an O(windows^2) affinity
# matrix, so a multi-hour recording becomes pathologically slow (a 7h file can
# run for hours). Above the threshold we diarize in bounded windows; each
# window's clustering stays small. Per-window clusters are labelled uniquely
# ("SPEAKER_00@c3") and reconciled to enrolled profiles downstream by the
# embedding/identify step, so enrolled speakers still get consistent names.
# Tunable via env; set CHUNK_SECONDS<=0 to force single-pass (legacy) behaviour.
_CHUNK_SECONDS = float(os.getenv("MEETILY_DIARIZATION_CHUNK_SECONDS", "1500"))  # 25 min
_CHUNK_THRESHOLD_SECONDS = float(
    os.getenv("MEETILY_DIARIZATION_CHUNK_THRESHOLD_SECONDS", "1800")  # only chunk if >30 min
)


@dataclass(frozen=True)
class SpeakerTurn:
    cluster_label: str
    start: float
    end: float


class DiarizationService:
    def __init__(self, pipeline_name: str, hf_token: str | None, device: str = "cpu") -> None:
        self.pipeline_name = pipeline_name
        self.hf_token = hf_token
        self.device = device
        self._pipeline: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        from pyannote.audio import Pipeline

        logger.info("Loading diarization pipeline: %s", self.pipeline_name)
        pipeline = Pipeline.from_pretrained(
            self.pipeline_name,
            **_auth_kwargs(Pipeline.from_pretrained, self.hf_token),
        )
        if pipeline is None:
            raise RuntimeError(
                "Could not load pyannote pipeline. Confirm the Hugging Face token has access "
                f"to {self.pipeline_name} and that the model terms have been accepted."
            )

        if self.device != "cpu":
            import torch

            pipeline.to(torch.device(self.device))

        self._pipeline = pipeline
        return pipeline

    def diarize(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        prepared_path = prepare_audio_for_ml(audio_path)

        pipeline = self._load_pipeline()
        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(prepared_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        total_seconds = waveform.shape[-1] / float(sample_rate)

        if _CHUNK_SECONDS <= 0 or total_seconds <= _CHUNK_THRESHOLD_SECONDS:
            # Short recording: single pass over the whole file (best quality).
            turns = self._diarize_window(
                pipeline, {"waveform": waveform, "sample_rate": sample_rate}, kwargs, 0.0, ""
            )
        else:
            logger.warning(
                "Diarizing long audio (%.1f min) in %.0f-min chunks to bound clustering cost; "
                "unenrolled speakers may be labelled per chunk.",
                total_seconds / 60.0,
                _CHUNK_SECONDS / 60.0,
            )
            turns = []
            chunk_index = 0
            start_seconds = 0.0
            while start_seconds < total_seconds:
                end_seconds = min(total_seconds, start_seconds + _CHUNK_SECONDS)
                s_frame = int(start_seconds * sample_rate)
                e_frame = int(end_seconds * sample_rate)
                chunk_waveform = waveform[:, s_frame:e_frame]
                turns.extend(
                    self._diarize_window(
                        pipeline,
                        {"waveform": chunk_waveform, "sample_rate": sample_rate},
                        kwargs,
                        start_seconds,
                        f"@c{chunk_index}",
                    )
                )
                chunk_index += 1
                if end_seconds >= total_seconds:
                    break
                start_seconds = end_seconds

        turns.sort(key=lambda turn: (turn.start, turn.end, turn.cluster_label))
        return turns

    def _diarize_window(
        self,
        pipeline: Any,
        audio_input: Any,
        kwargs: dict[str, int],
        offset_seconds: float,
        label_suffix: str,
    ) -> list[SpeakerTurn]:
        output = pipeline(audio_input, **kwargs)
        annotation = (
            getattr(output, "exclusive_speaker_diarization", None)
            or getattr(output, "speaker_diarization", None)
            or output
        )
        window_turns: list[SpeakerTurn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            window_turns.append(
                SpeakerTurn(
                    cluster_label=f"{label}{label_suffix}",
                    start=float(segment.start) + offset_seconds,
                    end=float(segment.end) + offset_seconds,
                )
            )
        return window_turns


def _auth_kwargs(from_pretrained: Any, hf_token: str | None) -> dict[str, str]:
    if not hf_token:
        return {}

    parameters = inspect.signature(from_pretrained).parameters
    if "token" in parameters:
        return {"token": hf_token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": hf_token}

    return {}
