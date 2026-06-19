use crate::database::models::SpeakerTurn;
use chrono::Utc;
use sqlx::{Connection, Error as SqlxError, SqlitePool};
use uuid::Uuid;

pub struct SpeakerTurnRepository;

#[derive(Debug, Clone)]
pub struct SpeakerTurnAssignment {
    pub cluster_label: String,
    pub speaker_profile_id: Option<String>,
    pub speaker_label: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: Option<f64>,
    pub confirmed: bool,
    pub assignment_source: String,
}

impl SpeakerTurnRepository {
    pub async fn replace_meeting_turns(
        pool: &SqlitePool,
        meeting_id: &str,
        turns: &[SpeakerTurnAssignment],
    ) -> Result<(), SqlxError> {
        let mut conn = pool.acquire().await?;
        let mut transaction = conn.begin().await?;

        sqlx::query("DELETE FROM speaker_turns WHERE meeting_id = ?")
            .bind(meeting_id)
            .execute(&mut *transaction)
            .await?;

        let now = Utc::now().to_rfc3339();
        for turn in turns {
            sqlx::query(
                "INSERT INTO speaker_turns
                 (id, meeting_id, cluster_label, speaker_profile_id, start_time, end_time,
                  confidence, assignment_source, confirmed, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(format!("speaker-turn-{}", Uuid::new_v4()))
            .bind(meeting_id)
            .bind(&turn.cluster_label)
            .bind(&turn.speaker_profile_id)
            .bind(turn.start_time)
            .bind(turn.end_time)
            .bind(turn.confidence)
            .bind(&turn.assignment_source)
            .bind(if turn.confirmed { 1_i64 } else { 0_i64 })
            .bind(&now)
            .bind(&now)
            .execute(&mut *transaction)
            .await?;
        }

        transaction.commit().await
    }

    pub async fn list_meeting_turns(
        pool: &SqlitePool,
        meeting_id: &str,
    ) -> Result<Vec<SpeakerTurn>, SqlxError> {
        sqlx::query_as::<_, SpeakerTurn>(
            "SELECT * FROM speaker_turns WHERE meeting_id = ? ORDER BY start_time ASC",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await
    }

    pub async fn assign_cluster(
        pool: &SqlitePool,
        meeting_id: &str,
        cluster_label: &str,
        profile_id: &str,
        confidence: Option<f64>,
        confirmed: bool,
        assignment_source: &str,
    ) -> Result<u64, SqlxError> {
        let updated_at = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE speaker_turns
             SET speaker_profile_id = ?, confidence = ?, confirmed = ?, assignment_source = ?, updated_at = ?
             WHERE meeting_id = ? AND cluster_label = ?",
        )
        .bind(profile_id)
        .bind(confidence)
        .bind(if confirmed { 1_i64 } else { 0_i64 })
        .bind(assignment_source)
        .bind(updated_at)
        .bind(meeting_id)
        .bind(cluster_label)
        .execute(pool)
        .await?;

        Ok(result.rows_affected())
    }

    pub async fn assign_transcripts_by_overlap(
        pool: &SqlitePool,
        meeting_id: &str,
        turns: &[SpeakerTurnAssignment],
    ) -> Result<u64, SqlxError> {
        // Run all reads and writes in one transaction so a meeting is never
        // left half-assigned if an update fails partway through the loop.
        let mut tx = pool.begin().await?;

        let transcripts: Vec<(String, Option<f64>, Option<f64>, Option<i64>)> = sqlx::query_as(
            "SELECT id, audio_start_time, audio_end_time, speaker_confirmed FROM transcripts WHERE meeting_id = ?",
        )
        .bind(meeting_id)
        .fetch_all(&mut *tx)
        .await?;

        let mut updated = 0_u64;
        for (transcript_id, start, end, confirmed) in transcripts {
            // Never overwrite a speaker the user manually confirmed. Manual
            // confirmation sets transcripts.speaker_confirmed=1 directly (with
            // no speaker_turns row), so automatic overlap propagation must skip
            // these rows or it would silently destroy user-confirmed labels.
            if confirmed.unwrap_or(0) == 1 {
                continue;
            }

            let (Some(start), Some(end)) = (start, end) else {
                continue;
            };
            if end <= start {
                continue;
            }

            let Some(turn) = best_turn_for_segment(start, end, turns) else {
                continue;
            };

            let result = sqlx::query(
                "UPDATE transcripts
                 SET speaker_profile_id = ?, speaker_label = ?, speaker_confidence = ?, speaker_confirmed = ?
                 WHERE id = ?",
            )
            .bind(&turn.speaker_profile_id)
            .bind(&turn.speaker_label)
            .bind(turn.confidence)
            .bind(if turn.confirmed { 1_i64 } else { 0_i64 })
            .bind(&transcript_id)
            .execute(&mut *tx)
            .await?;

            updated += result.rows_affected();
        }

        tx.commit().await?;

        Ok(updated)
    }
}

pub fn best_turn_for_segment<'a>(
    segment_start: f64,
    segment_end: f64,
    turns: &'a [SpeakerTurnAssignment],
) -> Option<&'a SpeakerTurnAssignment> {
    if segment_end <= segment_start {
        return None;
    }

    turns
        .iter()
        .filter_map(|turn| {
            let overlap_start = segment_start.max(turn.start_time);
            let overlap_end = segment_end.min(turn.end_time);
            let overlap = (overlap_end - overlap_start).max(0.0);
            if overlap > 0.0 {
                Some((turn, overlap))
            } else {
                None
            }
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(turn, _)| turn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(label: &str, start: f64, end: f64) -> SpeakerTurnAssignment {
        SpeakerTurnAssignment {
            cluster_label: label.to_string(),
            speaker_profile_id: None,
            speaker_label: label.to_string(),
            start_time: start,
            end_time: end,
            confidence: None,
            confirmed: false,
            assignment_source: "test".to_string(),
        }
    }

    #[test]
    fn best_turn_for_segment_picks_containing_turn() {
        let turns = vec![turn("SPEAKER_00", 0.0, 10.0), turn("SPEAKER_01", 11.0, 20.0)];

        let result = best_turn_for_segment(2.0, 4.0, &turns).unwrap();

        assert_eq!(result.cluster_label, "SPEAKER_00");
    }

    #[test]
    fn best_turn_for_segment_picks_largest_overlap() {
        let turns = vec![turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 4.0, 10.0)];

        let result = best_turn_for_segment(3.0, 9.0, &turns).unwrap();

        assert_eq!(result.cluster_label, "SPEAKER_01");
    }

    #[test]
    fn best_turn_for_segment_returns_none_without_overlap() {
        let turns = vec![turn("SPEAKER_00", 0.0, 5.0), turn("SPEAKER_01", 10.0, 15.0)];

        let result = best_turn_for_segment(6.0, 9.0, &turns);

        assert!(result.is_none());
    }

    #[test]
    fn best_turn_for_segment_rejects_zero_duration_segment() {
        let turns = vec![turn("SPEAKER_00", 0.0, 5.0)];

        let result = best_turn_for_segment(2.0, 2.0, &turns);

        assert!(result.is_none());
    }
}
