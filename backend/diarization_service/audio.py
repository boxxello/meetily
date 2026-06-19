from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def prepare_audio_for_ml(audio_path: str) -> Path:
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    target = _cache_path(source)
    if target.exists() and target.stat().st_size > 44:
        return target

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare audio for speaker recognition")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f"{target.stem}.{os.getpid()}.tmp.wav")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(temp_target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temp_target.unlink(missing_ok=True)
        detail = result.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"Failed to prepare audio for speaker recognition: {detail}")

    temp_target.replace(target)
    return target


def _cache_path(source: Path) -> Path:
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    ).hexdigest()
    cache_root = Path(tempfile.gettempdir()) / "meetily-speaker-audio"
    return cache_root / f"{cache_key}.wav"
