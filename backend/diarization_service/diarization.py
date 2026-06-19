from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import prepare_audio_for_ml

logger = logging.getLogger(__name__)


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

        output = pipeline(str(prepared_path), **kwargs)
        annotation = (
            getattr(output, "exclusive_speaker_diarization", None)
            or getattr(output, "speaker_diarization", None)
            or output
        )

        turns: list[SpeakerTurn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    cluster_label=str(label),
                    start=float(segment.start),
                    end=float(segment.end),
                )
            )

        turns.sort(key=lambda turn: (turn.start, turn.end, turn.cluster_label))
        return turns


def _auth_kwargs(from_pretrained: Any, hf_token: str | None) -> dict[str, str]:
    if not hf_token:
        return {}

    parameters = inspect.signature(from_pretrained).parameters
    if "token" in parameters:
        return {"token": hf_token}
    if "use_auth_token" in parameters:
        return {"use_auth_token": hf_token}

    return {}
