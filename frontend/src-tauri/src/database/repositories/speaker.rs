use crate::database::models::{SpeakerProfile, SpeakerVoiceprint};
use chrono::Utc;
use sqlx::{Error as SqlxError, SqlitePool};
use uuid::Uuid;

pub struct SpeakerRepository;

impl SpeakerRepository {
    pub async fn list_profiles(pool: &SqlitePool) -> Result<Vec<SpeakerProfile>, SqlxError> {
        sqlx::query_as::<_, SpeakerProfile>(
            "SELECT * FROM speaker_profiles WHERE archived_at IS NULL ORDER BY display_name ASC",
        )
        .fetch_all(pool)
        .await
    }

    pub async fn create_profile(
        pool: &SqlitePool,
        display_name: &str,
        color: Option<&str>,
    ) -> Result<SpeakerProfile, SqlxError> {
        let profile = SpeakerProfile {
            id: format!("speaker-profile-{}", Uuid::new_v4()),
            display_name: display_name.trim().to_string(),
            color: color.map(str::to_string),
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            archived_at: None,
        };

        sqlx::query(
            "INSERT INTO speaker_profiles (id, display_name, color, created_at, updated_at, archived_at)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&profile.id)
        .bind(&profile.display_name)
        .bind(&profile.color)
        .bind(&profile.created_at)
        .bind(&profile.updated_at)
        .bind(&profile.archived_at)
        .execute(pool)
        .await?;

        Ok(profile)
    }

    pub async fn rename_profile(
        pool: &SqlitePool,
        profile_id: &str,
        display_name: &str,
    ) -> Result<Option<SpeakerProfile>, SqlxError> {
        let updated_at = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE speaker_profiles SET display_name = ?, updated_at = ? WHERE id = ? AND archived_at IS NULL",
        )
        .bind(display_name.trim())
        .bind(&updated_at)
        .bind(profile_id)
        .execute(pool)
        .await?;

        if result.rows_affected() == 0 {
            return Ok(None);
        }

        sqlx::query_as::<_, SpeakerProfile>(
            "SELECT * FROM speaker_profiles WHERE id = ? AND archived_at IS NULL",
        )
        .bind(profile_id)
        .fetch_optional(pool)
        .await
    }

    pub async fn list_voiceprints(pool: &SqlitePool) -> Result<Vec<SpeakerVoiceprint>, SqlxError> {
        sqlx::query_as::<_, SpeakerVoiceprint>("SELECT * FROM speaker_voiceprints")
            .fetch_all(pool)
            .await
    }

    pub async fn upsert_voiceprint(
        pool: &SqlitePool,
        profile_id: &str,
        embedding_model: &str,
        embedding: Vec<u8>,
        sample_count: i64,
        total_duration: f64,
        source_meeting_id: Option<&str>,
        source_cluster_label: Option<&str>,
    ) -> Result<SpeakerVoiceprint, SqlxError> {
        let now = Utc::now().to_rfc3339();
        let existing: Option<SpeakerVoiceprint> = sqlx::query_as::<_, SpeakerVoiceprint>(
            "SELECT * FROM speaker_voiceprints
             WHERE speaker_profile_id = ? AND embedding_model = ?
             ORDER BY updated_at DESC
             LIMIT 1",
        )
        .bind(profile_id)
        .bind(embedding_model)
        .fetch_optional(pool)
        .await?;

        if let Some(mut voiceprint) = existing {
            voiceprint.embedding = embedding;
            voiceprint.sample_count = sample_count;
            voiceprint.total_duration = total_duration;
            voiceprint.source_meeting_id = source_meeting_id.map(str::to_string);
            voiceprint.source_cluster_label = source_cluster_label.map(str::to_string);
            voiceprint.updated_at = now;

            sqlx::query(
                "UPDATE speaker_voiceprints
                 SET embedding = ?, sample_count = ?, total_duration = ?, source_meeting_id = ?,
                     source_cluster_label = ?, updated_at = ?
                 WHERE id = ?",
            )
            .bind(&voiceprint.embedding)
            .bind(voiceprint.sample_count)
            .bind(voiceprint.total_duration)
            .bind(&voiceprint.source_meeting_id)
            .bind(&voiceprint.source_cluster_label)
            .bind(&voiceprint.updated_at)
            .bind(&voiceprint.id)
            .execute(pool)
            .await?;

            return Ok(voiceprint);
        }

        let voiceprint = SpeakerVoiceprint {
            id: format!("speaker-voiceprint-{}", Uuid::new_v4()),
            speaker_profile_id: profile_id.to_string(),
            embedding_model: embedding_model.to_string(),
            embedding,
            sample_count,
            total_duration,
            source_meeting_id: source_meeting_id.map(str::to_string),
            source_cluster_label: source_cluster_label.map(str::to_string),
            created_at: now.clone(),
            updated_at: now,
        };

        sqlx::query(
            "INSERT INTO speaker_voiceprints
             (id, speaker_profile_id, embedding_model, embedding, sample_count, total_duration,
              source_meeting_id, source_cluster_label, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&voiceprint.id)
        .bind(&voiceprint.speaker_profile_id)
        .bind(&voiceprint.embedding_model)
        .bind(&voiceprint.embedding)
        .bind(voiceprint.sample_count)
        .bind(voiceprint.total_duration)
        .bind(&voiceprint.source_meeting_id)
        .bind(&voiceprint.source_cluster_label)
        .bind(&voiceprint.created_at)
        .bind(&voiceprint.updated_at)
        .execute(pool)
        .await?;

        Ok(voiceprint)
    }
}
