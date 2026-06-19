import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    host: str
    port: int
    hf_token: str | None
    diarization_pipeline: str
    embedding_model: str
    device: str
    recognition_threshold: float
    ambiguity_margin: float


def load_config() -> ServiceConfig:
    _load_local_env()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    return ServiceConfig(
        host=os.getenv("MEETILY_DIARIZATION_HOST", "127.0.0.1"),
        port=int(os.getenv("MEETILY_DIARIZATION_PORT", "8179")),
        hf_token=token,
        diarization_pipeline=os.getenv(
            "DIARIZATION_PIPELINE",
            "pyannote/speaker-diarization-3.1",
        ),
        embedding_model=os.getenv(
            "SPEAKER_EMBEDDING_MODEL",
            "speechbrain/spkrec-ecapa-voxceleb",
        ),
        device=_speaker_device(),
        # Cosine threshold for accepting a speaker match. Same-speaker pairs
        # across recordings score ~0.73, different speakers <=~0.21, so 0.72
        # rejected legitimate matches; 0.5 sits in the safe gap. Override via env.
        recognition_threshold=float(os.getenv("SPEAKER_RECOGNITION_THRESHOLD", "0.5")),
        ambiguity_margin=float(os.getenv("SPEAKER_AMBIGUITY_MARGIN", "0.05")),
    )


def _load_local_env() -> None:
    for path in _env_file_candidates():
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            key, value = _parse_env_line(line)
            if key:
                os.environ.setdefault(key, value)


def _env_file_candidates() -> list[Path]:
    return [
        Path(__file__).with_name(".env"),
        Path.home() / ".config" / "meetily" / "speaker.env",
    ]


def _parse_env_line(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None, ""

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip("\"'")
    if not key:
        return None, ""
    return key, value


def _speaker_device() -> str:
    configured = os.getenv("SPEAKER_DEVICE")
    if configured:
        return configured

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"
