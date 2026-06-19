-- Speaker recognition profiles and transcript speaker assignments.
-- This is person-level speaker recognition, not mic/system/stereo source labeling.

CREATE TABLE IF NOT EXISTS speaker_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    color TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS speaker_voiceprints (
    id TEXT PRIMARY KEY,
    speaker_profile_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 1,
    total_duration REAL NOT NULL DEFAULT 0,
    source_meeting_id TEXT,
    source_cluster_label TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (speaker_profile_id) REFERENCES speaker_profiles(id) ON DELETE CASCADE,
    FOREIGN KEY (source_meeting_id) REFERENCES meetings(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS speaker_turns (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    cluster_label TEXT NOT NULL,
    speaker_profile_id TEXT,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    confidence REAL,
    assignment_source TEXT NOT NULL DEFAULT 'diarization',
    confirmed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
    FOREIGN KEY (speaker_profile_id) REFERENCES speaker_profiles(id) ON DELETE SET NULL
);

ALTER TABLE transcripts ADD COLUMN speaker_profile_id TEXT;
ALTER TABLE transcripts ADD COLUMN speaker_label TEXT;
ALTER TABLE transcripts ADD COLUMN speaker_confidence REAL;
ALTER TABLE transcripts ADD COLUMN speaker_confirmed INTEGER NOT NULL DEFAULT 0;
