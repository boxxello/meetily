from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.diarization_service.diarization import SpeakerTurn
from backend.diarization_service.embeddings import EmbeddingService, ProfileEmbedding


class EmbeddingServiceTest(unittest.TestCase):
    def test_identify_cluster_returns_no_match_when_turns_are_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.touch()

            service = EmbeddingService("speechbrain/spkrec-ecapa-voxceleb")
            turns = [
                SpeakerTurn(cluster_label="SPEAKER_00", start=0.0, end=0.4),
                SpeakerTurn(cluster_label="SPEAKER_00", start=1.0, end=1.6),
            ]
            profiles = [
                ProfileEmbedding(
                    profile_id="known-speaker",
                    display_name="Known Speaker",
                    embedding_model="speechbrain/spkrec-ecapa-voxceleb",
                    embedding=[1.0, 0.0],
                )
            ]

            with (
                patch("backend.diarization_service.embeddings.prepare_audio_for_ml") as prepare,
                patch.object(service, "_load_model", side_effect=AssertionError("model loaded")),
            ):
                prepare.return_value = audio_path
                match = service.identify_cluster(
                    audio_path=str(audio_path),
                    cluster_label="SPEAKER_00",
                    turns=turns,
                    profiles=profiles,
                    threshold=0.72,
                    ambiguity_margin=0.05,
                )

            self.assertEqual(match.cluster_label, "SPEAKER_00")
            self.assertIsNone(match.profile_id)
            self.assertIsNone(match.display_name)
            self.assertIsNone(match.confidence)
            self.assertFalse(match.ambiguous)
            self.assertEqual(match.sample_count, 0)
            self.assertAlmostEqual(match.total_duration, 1.0)


if __name__ == "__main__":
    unittest.main()
