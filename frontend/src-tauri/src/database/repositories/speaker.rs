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

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE speaker_profiles (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                color TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE speaker_voiceprints (
                id TEXT PRIMARY KEY,
                speaker_profile_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 1,
                total_duration REAL NOT NULL DEFAULT 0,
                source_meeting_id TEXT,
                source_cluster_label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    #[tokio::test]
    async fn create_profile_trims_name_and_is_listed() {
        let pool = test_pool().await;
        let created = SpeakerRepository::create_profile(&pool, "  Alice  ", Some("#ff0000"))
            .await
            .unwrap();
        assert_eq!(created.display_name, "Alice");
        assert_eq!(created.color.as_deref(), Some("#ff0000"));
        assert!(created.id.starts_with("speaker-profile-"));
        assert!(created.archived_at.is_none());

        let all = SpeakerRepository::list_profiles(&pool).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].display_name, "Alice");
    }

    #[tokio::test]
    async fn list_profiles_excludes_archived_and_sorts_by_name() {
        let pool = test_pool().await;
        SpeakerRepository::create_profile(&pool, "Bob", None).await.unwrap();
        SpeakerRepository::create_profile(&pool, "Alice", None).await.unwrap();
        let carol = SpeakerRepository::create_profile(&pool, "Carol", None).await.unwrap();
        sqlx::query("UPDATE speaker_profiles SET archived_at = '2026-06-04T00:00:00Z' WHERE id = ?")
            .bind(&carol.id)
            .execute(&pool)
            .await
            .unwrap();

        let names: Vec<String> = SpeakerRepository::list_profiles(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|p| p.display_name)
            .collect();
        // Alphabetical order, archived Carol excluded.
        assert_eq!(names, vec!["Alice".to_string(), "Bob".to_string()]);
    }

    #[tokio::test]
    async fn rename_profile_trims_updates_and_returns_profile() {
        let pool = test_pool().await;
        let p = SpeakerRepository::create_profile(&pool, "Alice", None)
            .await
            .unwrap();
        let renamed = SpeakerRepository::rename_profile(&pool, &p.id, "  Alicia ")
            .await
            .unwrap()
            .expect("rename should return the updated profile");
        assert_eq!(renamed.display_name, "Alicia");
        assert!(renamed.updated_at >= p.updated_at);
    }

    #[tokio::test]
    async fn rename_missing_profile_returns_none() {
        let pool = test_pool().await;
        let result = SpeakerRepository::rename_profile(&pool, "does-not-exist", "X")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn rename_archived_profile_returns_none() {
        let pool = test_pool().await;
        let p = SpeakerRepository::create_profile(&pool, "Alice", None)
            .await
            .unwrap();
        sqlx::query("UPDATE speaker_profiles SET archived_at = '2026-06-04T00:00:00Z' WHERE id = ?")
            .bind(&p.id)
            .execute(&pool)
            .await
            .unwrap();
        let result = SpeakerRepository::rename_profile(&pool, &p.id, "Alicia")
            .await
            .unwrap();
        assert!(result.is_none(), "archived profiles must not be renamable");
    }

    #[tokio::test]
    async fn upsert_voiceprint_updates_in_place_for_same_model() {
        let pool = test_pool().await;
        let p = SpeakerRepository::create_profile(&pool, "Alice", None)
            .await
            .unwrap();

        let first = SpeakerRepository::upsert_voiceprint(
            &pool, &p.id, "ecapa", vec![1, 2, 3], 2, 4.0, Some("m1"), Some("SPEAKER_00"),
        )
        .await
        .unwrap();
        let second = SpeakerRepository::upsert_voiceprint(
            &pool, &p.id, "ecapa", vec![9, 9, 9], 5, 12.5, Some("m2"), Some("SPEAKER_01"),
        )
        .await
        .unwrap();

        // Same row reused (id preserved), values replaced.
        assert_eq!(first.id, second.id);
        assert_eq!(second.embedding, vec![9, 9, 9]);
        assert_eq!(second.sample_count, 5);
        assert_eq!(second.total_duration, 12.5);

        let all = SpeakerRepository::list_voiceprints(&pool).await.unwrap();
        assert_eq!(all.len(), 1, "same profile+model must not create a duplicate row");
        assert_eq!(all[0].embedding, vec![9, 9, 9]);
    }

    #[tokio::test]
    async fn upsert_voiceprint_keeps_separate_rows_per_model() {
        let pool = test_pool().await;
        let p = SpeakerRepository::create_profile(&pool, "Alice", None)
            .await
            .unwrap();
        SpeakerRepository::upsert_voiceprint(&pool, &p.id, "ecapa", vec![1], 1, 1.0, None, None)
            .await
            .unwrap();
        SpeakerRepository::upsert_voiceprint(&pool, &p.id, "wespeaker", vec![2], 1, 1.0, None, None)
            .await
            .unwrap();

        let all = SpeakerRepository::list_voiceprints(&pool).await.unwrap();
        assert_eq!(all.len(), 2, "different embedding models are distinct voiceprints");
    }
}
