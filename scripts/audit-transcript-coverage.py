#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved transcript timing coverage against audio speech regions."
    )
    parser.add_argument("meeting_dir", nargs="?", help="Path to a Meetily recording folder")
    parser.add_argument(
        "--latest-completed",
        action="store_true",
        help="Audit the newest completed recording under --recordings-dir",
    )
    parser.add_argument(
        "--recordings-dir",
        default="~/Documents/meetily-recordings",
        help="Recordings directory used with --latest-completed",
    )
    parser.add_argument("--silence-db", default="-38dB", help="ffmpeg silencedetect noise threshold")
    parser.add_argument("--min-silence-sec", type=float, default=0.7)
    args = parser.parse_args()

    if args.latest_completed:
        folder = latest_completed_recording(Path(args.recordings_dir).expanduser())
        if folder is None:
            print(f"error: no completed recording found under {args.recordings_dir}", file=sys.stderr)
            return 2
    elif args.meeting_dir:
        folder = Path(args.meeting_dir).expanduser().resolve()
    else:
        parser.error("provide meeting_dir or use --latest-completed")

    if not folder.is_dir():
        print(f"error: meeting folder not found: {folder}", file=sys.stderr)
        return 2

    audio_path = resolve_audio_path(folder)
    transcript_path = folder / "transcripts.json"
    metadata_path = folder / "metadata.json"
    diagnostics_path = folder / "transcription_diagnostics.json"

    if not audio_path or not audio_path.exists():
        print(f"error: audio file not found in {folder}", file=sys.stderr)
        return 2
    if not transcript_path.exists():
        print(f"error: transcripts.json not found in {folder}", file=sys.stderr)
        return 2

    audio_duration = ffprobe_duration(audio_path)
    silence_ranges = detect_silence(audio_path, args.silence_db, args.min_silence_sec)
    speech_ranges = invert_ranges(silence_ranges, audio_duration)
    transcript_raw_ranges = load_transcript_ranges(transcript_path)
    transcript_ranges = merge_ranges(transcript_raw_ranges)
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    diagnostics = load_diagnostics(diagnostics_path) if diagnostics_path.exists() else []

    speech_covered = covered_duration(speech_ranges, transcript_ranges)
    speech_total = sum(end - start for start, end in speech_ranges)
    transcript_total = sum(end - start for start, end in transcript_ranges)
    large_uncovered = uncovered_ranges(speech_ranges, transcript_ranges, min_duration=0.75)
    transcript_gaps = adjacent_gaps(transcript_ranges, min_duration=2.0)

    print(f"folder: {folder}")
    print(f"status: {metadata.get('status', '<unknown>')}")
    print(f"audio_duration_sec: {audio_duration:.2f}")
    print(f"speech_regions: {len(speech_ranges)}")
    print(f"speech_duration_sec: {speech_total:.2f}")
    print(f"transcript_segments: {len(transcript_raw_ranges)}")
    print(f"transcript_coverage_ranges: {len(transcript_ranges)}")
    print(f"transcript_duration_sec: {transcript_total:.2f}")
    coverage = (speech_covered / speech_total * 100.0) if speech_total > 0 else 100.0
    print(f"speech_covered_by_transcripts_sec: {speech_covered:.2f} ({coverage:.1f}%)")
    print(f"uncovered_speech_regions_ge_0_75s: {len(large_uncovered)}")
    for start, end in large_uncovered[:20]:
        print(f"  uncovered: {start:.2f}-{end:.2f}s ({end - start:.2f}s)")
    if len(large_uncovered) > 20:
        print(f"  ... {len(large_uncovered) - 20} more")
    print(f"transcript_gaps_ge_2s: {len(transcript_gaps)}")
    for start, end in transcript_gaps[:20]:
        print(f"  transcript_gap: {start:.2f}-{end:.2f}s ({end - start:.2f}s)")
    if len(transcript_gaps) > 20:
        print(f"  ... {len(transcript_gaps) - 20} more")
    print_diagnostics_summary(diagnostics)

    return 0


def latest_completed_recording(recordings_dir: Path) -> Path | None:
    if not recordings_dir.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for folder in recordings_dir.iterdir():
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        if not (folder / "transcripts.json").exists():
            continue
        if resolve_audio_path(folder) is None:
            continue
        candidates.append((folder.stat().st_mtime, folder))

    if not candidates:
        return None
    return max(candidates)[1].resolve()


def resolve_audio_path(folder: Path) -> Path | None:
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        audio_file = metadata.get("audio_file")
        if isinstance(audio_file, str) and audio_file.strip():
            path = Path(audio_file)
            candidate = path if path.is_absolute() else folder / path
            if candidate.exists():
                return candidate

    for name in ("audio.mp4", "audio.m4a", "audio.wav", "audio.mp3"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def ffprobe_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def detect_silence(audio_path: Path, silence_db: str, min_silence_sec: float) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=n={silence_db}:d={min_silence_sec}",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    starts: list[float] = []
    ranges: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start: ([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue

        end_match = re.search(r"silence_end: ([0-9.]+)", line)
        if end_match and starts:
            ranges.append((starts.pop(0), float(end_match.group(1))))

    return merge_ranges(ranges)


def load_transcript_ranges(transcript_path: Path) -> list[tuple[float, float]]:
    data = load_json(transcript_path)
    ranges = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        start = segment.get("audio_start_time")
        end = segment.get("audio_end_time")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            ranges.append((float(start), float(end)))
    return ranges


def load_diagnostics(diagnostics_path: Path) -> list[dict]:
    data = load_json(diagnostics_path)
    events = data.get("events", [])
    return [event for event in events if isinstance(event, dict)]


def print_diagnostics_summary(events: list[dict]) -> None:
    if not events:
        print("transcription_diagnostics: missing")
        return

    by_event = Counter(str(event.get("event", "<unknown>")) for event in events)
    received_events = {"audio_chunk_received", "vad_chunk_received"}
    terminal_events = {
        "asr_emitted",
        "asr_retry_emitted",
        "asr_empty_retry_dropped",
        "asr_empty_no_retry",
        "model_not_ready_dropped",
        "audio_too_short_dropped",
        "model_unloaded_during_transcription",
        "transcription_error",
        "asr_retry_error",
    }
    received = sum(by_event.get(event, 0) for event in received_events)
    emitted = by_event.get("asr_emitted", 0) + by_event.get("asr_retry_emitted", 0)
    empty_retry = by_event.get("asr_empty_retry_pending", 0)
    empty_dropped = by_event.get("asr_empty_retry_dropped", 0) + by_event.get("asr_empty_no_retry", 0)
    hard_dropped = (
        by_event.get("model_not_ready_dropped", 0)
        + by_event.get("audio_too_short_dropped", 0)
        + by_event.get("model_unloaded_during_transcription", 0)
        + by_event.get("transcription_error", 0)
        + by_event.get("asr_retry_error", 0)
    )

    print(f"transcription_diagnostics_events: {len(events)}")
    print(f"diagnostic_audio_chunks_received: {received}")
    print(f"diagnostic_asr_emitted: {emitted}")
    print(f"diagnostic_empty_retry_pending: {empty_retry}")
    print(f"diagnostic_empty_dropped: {empty_dropped}")
    print(f"diagnostic_hard_dropped_or_error: {hard_dropped}")
    print("diagnostic_events_by_type:")
    for event, count in sorted(by_event.items()):
        print(f"  {event}: {count}")

    missing_emit_chunks = sorted(
        {
            event.get("chunk_id")
            for event in events
            if event.get("event") in received_events
            and event.get("chunk_id") is not None
        }
        - {
            event.get("chunk_id")
            for event in events
            if event.get("event") in terminal_events
            and event.get("chunk_id") is not None
        }
    )
    print(f"diagnostic_chunks_without_terminal_event: {len(missing_emit_chunks)}")
    if missing_emit_chunks:
        preview = ", ".join(str(chunk_id) for chunk_id in missing_emit_chunks[:20])
        suffix = "" if len(missing_emit_chunks) <= 20 else f", ... {len(missing_emit_chunks) - 20} more"
        print(f"  chunks: {preview}{suffix}")


def invert_ranges(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    speech = []
    cursor = 0.0
    for start, end in ranges:
        if start > cursor:
            speech.append((cursor, min(start, duration)))
        cursor = max(cursor, end)
    if cursor < duration:
        speech.append((cursor, duration))
    return [(start, end) for start, end in speech if end - start > 0.05]


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def covered_duration(
    speech_ranges: list[tuple[float, float]],
    transcript_ranges: list[tuple[float, float]],
) -> float:
    covered = 0.0
    for speech_start, speech_end in speech_ranges:
        for trans_start, trans_end in transcript_ranges:
            overlap = min(speech_end, trans_end) - max(speech_start, trans_start)
            if overlap > 0:
                covered += overlap
    return covered


def uncovered_ranges(
    speech_ranges: list[tuple[float, float]],
    transcript_ranges: list[tuple[float, float]],
    min_duration: float,
) -> list[tuple[float, float]]:
    uncovered = []
    for speech_start, speech_end in speech_ranges:
        cursor = speech_start
        for trans_start, trans_end in transcript_ranges:
            if trans_end <= cursor:
                continue
            if trans_start >= speech_end:
                break
            if trans_start > cursor:
                uncovered.append((cursor, min(trans_start, speech_end)))
            cursor = max(cursor, trans_end)
        if cursor < speech_end:
            uncovered.append((cursor, speech_end))
    return [(start, end) for start, end in uncovered if end - start >= min_duration]


def adjacent_gaps(ranges: list[tuple[float, float]], min_duration: float) -> list[tuple[float, float]]:
    gaps = []
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        if next_start - prev_end >= min_duration:
            gaps.append((prev_end, next_start))
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
