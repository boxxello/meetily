from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from backend.diarization_service.config import load_config
from backend.diarization_service.diarization import DiarizationService, SpeakerTurn
from backend.diarization_service.embeddings import (
    EmbeddingService,
    ProfileEmbedding,
    group_turns_by_cluster,
)


DEFAULT_FIXTURE_URL = "https://www.youtube.com/watch?v=TLkA0RELQ1g"


@unittest.skipUnless(
    os.getenv("MEETILY_RUN_SPEAKER_INTEGRATION") == "1",
    "Set MEETILY_RUN_SPEAKER_INTEGRATION=1 to run the speaker recognition integration test.",
)
class SpeakerRecognitionIntegrationTest(unittest.TestCase):
    def test_youtube_fixture_reconciles_known_speaker_voiceprint(self) -> None:
        _log_phase("resolving audio fixture")
        audio_path = _speaker_audio_fixture()
        config = load_config()

        _log_phase(f"diarizing fixture audio: {audio_path}")
        diarizer = DiarizationService(
            pipeline_name=config.diarization_pipeline,
            hf_token=config.hf_token,
            device=config.device,
        )
        turns = diarizer.diarize(
            str(audio_path),
            min_speakers=2,
            max_speakers=2,
        )
        clusters = group_turns_by_cluster(turns)

        self.assertGreaterEqual(len(clusters), 2)

        _log_phase(f"embedding and identifying speaker across {len(clusters)} clusters")
        cluster_label, enrollment_turns, probe_turns = _speaker_cluster_split(clusters)
        embedder = EmbeddingService(
            model_name=config.embedding_model,
            device=config.device,
        )
        enrolled = embedder.embed_cluster(str(audio_path), enrollment_turns)

        profile = ProfileEmbedding(
            profile_id="fixture-speaker-profile",
            display_name="Fixture Speaker",
            embedding_model=enrolled.embedding_model,
            embedding=enrolled.embedding,
        )
        match = embedder.identify_cluster(
            audio_path=str(audio_path),
            cluster_label=cluster_label,
            turns=probe_turns,
            profiles=[profile],
            threshold=0.72,
            ambiguity_margin=0.05,
        )

        self.assertEqual(match.profile_id, profile.profile_id)
        self.assertEqual(match.display_name, profile.display_name)
        self.assertFalse(match.ambiguous)
        self.assertIsNotNone(match.confidence)
        self.assertGreaterEqual(match.confidence or 0.0, 0.72)
        self.assertGreater(match.sample_count, 0)
        self.assertGreater(match.total_duration, 0.0)


def _speaker_audio_fixture() -> Path:
    configured_path = os.getenv("MEETILY_INTEGRATION_AUDIO_PATH")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MEETILY_INTEGRATION_AUDIO_PATH does not exist: {path}")
        return path

    cache_dir = Path(
        os.getenv(
            "MEETILY_INTEGRATION_CACHE",
            ".integration-artifacts/speaker-recognition",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "elephants-dream.mp3"
    if target.is_file() and target.stat().st_size > 1024:
        return target.resolve()

    url = os.getenv("MEETILY_INTEGRATION_VIDEO_URL", DEFAULT_FIXTURE_URL)
    if _download_with_metube(url, target):
        return target.resolve()

    _download_with_ytdlp(url, target)
    return target.resolve()


def _download_with_metube(url: str, target: Path) -> bool:
    base_url = os.getenv("MEETILY_METUBE_URL", "http://metube.lan").rstrip("/")
    if not base_url:
        return False

    _log_phase(f"trying MeTube fixture download via {base_url}")
    payload = {
        "url": url,
        "quality": "best",
        "download_type": "audio",
        "codec": "auto",
        "format": "mp3",
        "auto_start": True,
        "subtitle_language": "en",
        "subtitle_mode": "prefer_manual",
    }
    try:
        _json_request(f"{base_url}/add", payload=payload, timeout=10)
    except (TimeoutError, HTTPError, URLError, OSError):
        pass

    deadline = time.monotonic() + int(os.getenv("MEETILY_METUBE_TIMEOUT_SECS", "20"))
    while time.monotonic() < deadline:
        try:
            history = _json_request(f"{base_url}/history", timeout=10)
        except (TimeoutError, HTTPError, URLError, OSError, json.JSONDecodeError):
            return False

        candidates = [
            item
            for item in history.get("done", [])
            if item.get("url") == url
            and item.get("status") == "finished"
            and item.get("filename")
        ]
        if candidates:
            filename = candidates[0]["filename"]
            if _download_metube_file(base_url, filename, target):
                return True
        time.sleep(5)

    return False


def _download_metube_file(base_url: str, filename: str, target: Path) -> bool:
    download_url = f"{base_url}/download/{quote(filename, safe='')}"
    temp_target = target.with_suffix(f".{os.getpid()}.tmp")
    try:
        with urlopen(download_url, timeout=30) as response:
            if response.status != 200:
                return False
            with temp_target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (TimeoutError, HTTPError, URLError, OSError):
        temp_target.unlink(missing_ok=True)
        return False

    if temp_target.stat().st_size <= 1024:
        temp_target.unlink(missing_ok=True)
        return False

    temp_target.replace(target)
    return True


def _download_with_ytdlp(url: str, target: Path) -> None:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError("Neither MeTube nor local yt-dlp could provide the integration fixture")

    _log_phase("falling back to local yt-dlp fixture download")
    temp_dir = Path(tempfile.mkdtemp(prefix="meetily-speaker-fixture-"))
    output_template = temp_dir / "fixture.%(ext)s"
    try:
        subprocess.run(
            [
                ytdlp,
                "--socket-timeout",
                "30",
                "--sleep-requests",
                "2",
                "--sleep-interval",
                "2",
                "--max-sleep-interval",
                "8",
                "--no-playlist",
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "--audio-format",
                "mp3",
                "-o",
                str(output_template),
                *_extra_ytdlp_args(),
                url,
            ],
            check=True,
        )
        candidates = sorted(temp_dir.glob("fixture.*"))
        if not candidates:
            raise RuntimeError("yt-dlp completed without producing an audio file")
        candidates[0].replace(target)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _json_request(url: str, payload: dict[str, object] | None = None, timeout: int = 10) -> dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise HTTPError(url, response.status, response.reason, response.headers, None)
        content = response.read()
    if not content:
        return {}
    return json.loads(content.decode("utf-8"))


def _speaker_cluster_split(
    clusters: dict[str, list[SpeakerTurn]],
) -> tuple[str, list[SpeakerTurn], list[SpeakerTurn]]:
    eligible_clusters = []
    for label, turns in clusters.items():
        usable_turns = [turn for turn in turns if turn.end - turn.start >= 1.5]
        total_duration = sum(turn.end - turn.start for turn in usable_turns)
        if total_duration >= 3.0:
            eligible_clusters.append((label, usable_turns, total_duration))

    if not eligible_clusters:
        raise AssertionError("No diarized speaker cluster has enough speech for voiceprint testing")

    label, turns, _ = max(eligible_clusters, key=lambda item: item[2])
    turns = sorted(turns, key=lambda turn: turn.end - turn.start, reverse=True)

    longest_turn = turns[0]
    if longest_turn.end - longest_turn.start >= 3.0:
        first, second = _split_turn_in_halves(longest_turn)
        return label, [first], [second]

    return label, [turns[0]], turns[1:]


def _split_turn_in_halves(turn: SpeakerTurn) -> tuple[SpeakerTurn, SpeakerTurn]:
    midpoint = turn.start + ((turn.end - turn.start) / 2.0)
    return (
        SpeakerTurn(cluster_label=turn.cluster_label, start=turn.start, end=midpoint),
        SpeakerTurn(cluster_label=turn.cluster_label, start=midpoint, end=turn.end),
    )


def _extra_ytdlp_args() -> list[str]:
    configured = os.getenv("MEETILY_YTDLP_ARGS", "").strip()
    if not configured:
        return []
    return shlex.split(configured)


def _log_phase(message: str) -> None:
    print(f"[speaker-integration] {message}", file=sys.stderr, flush=True)
