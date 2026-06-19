use crate::database::{
    models::{MeetingModel, SpeakerProfile},
    repositories::{
        speaker::SpeakerRepository,
        speaker_turn::{SpeakerTurnAssignment, SpeakerTurnRepository},
    },
};
use crate::speaker_sidecar::SpeakerSidecarState;
use crate::state::AppState;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::{collections::HashMap, path::PathBuf, time::Duration};
use tracing::warn;

const DEFAULT_SIDECAR_URL: &str = "http://127.0.0.1:8179";
// ECAPA cosine threshold for accepting a speaker match. Empirically, a true
// same-speaker pair across two recordings scores ~0.73 while different speakers
// score <=~0.21, so 0.72 sat right on the same-speaker boundary and rejected
// legitimate matches as "unknown". 0.5 sits in the safe gap (well above the
// inter-speaker ceiling, well below same-speaker). Tunable via the sidecar's
// SPEAKER_RECOGNITION_THRESHOLD env var. TODO: calibrate on real meeting audio.
const DEFAULT_RECOGNITION_THRESHOLD: f64 = 0.5;
const DEFAULT_AMBIGUITY_MARGIN: f64 = 0.05;
// Upper bound on how many other meetings a single speaker assignment will
// re-identify (each re-runs diarization, ~20-40s). Without this, one click
// fans out across the entire meeting history and blocks for minutes/hours.
const MAX_PROPAGATION_MEETINGS: usize = 25;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfileDto {
    pub id: String,
    pub display_name: String,
    pub color: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerTurnDto {
    pub cluster_label: String,
    pub speaker_profile_id: Option<String>,
    pub speaker_label: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: Option<f64>,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerIdentificationResult {
    pub updated_transcript_count: u64,
    pub speaker_turn_count: usize,
    pub speakers: Vec<SpeakerTurnDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerAssignmentResult {
    pub updated_transcript_count: u64,
    pub updated_turn_count: u64,
    pub propagated_meeting_count: u64,
    pub propagated_transcript_count: u64,
    pub propagation_error_count: u64,
    pub profile: SpeakerProfileDto,
}

#[derive(Debug, Default, Clone)]
struct SpeakerPropagationSummary {
    meeting_count: u64,
    transcript_count: u64,
    error_count: u64,
}

#[derive(Debug, Serialize)]
struct SidecarDiarizeRequest {
    audio_path: String,
    num_speakers: Option<i64>,
    min_speakers: Option<i64>,
    max_speakers: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct SidecarDiarizeResponse {
    turns: Vec<SidecarTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SidecarTurn {
    cluster_label: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Serialize)]
struct SidecarProfile {
    profile_id: String,
    display_name: String,
    embedding_model: String,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct SidecarIdentifyRequest {
    audio_path: String,
    turns: Vec<SidecarTurn>,
    profiles: Vec<SidecarProfile>,
    threshold: f64,
    ambiguity_margin: f64,
}

#[derive(Debug, Deserialize)]
struct SidecarIdentifyResponse {
    clusters: Vec<SidecarClusterMatch>,
}

#[derive(Debug, Deserialize)]
struct SidecarClusterMatch {
    cluster_label: String,
    profile_id: Option<String>,
    display_name: Option<String>,
    confidence: Option<f64>,
    ambiguous: bool,
}

#[derive(Debug, Serialize)]
struct SidecarEmbedClusterRequest {
    audio_path: String,
    turns: Vec<SidecarTurn>,
    max_turns: i64,
    min_turn_duration: f64,
}

#[derive(Debug, Deserialize)]
struct SidecarEmbedClusterResponse {
    embedding_model: String,
    embedding: Vec<f32>,
    sample_count: i64,
    total_duration: f64,
}

#[tauri::command]
pub async fn api_list_speaker_profiles(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<SpeakerProfileDto>, String> {
    let pool = state.db_manager.pool();
    let profiles = SpeakerRepository::list_profiles(pool)
        .await
        .map_err(|e| format!("Failed to list speaker profiles: {}", e))?;

    Ok(profiles.into_iter().map(profile_to_dto).collect())
}

#[tauri::command]
pub async fn api_create_speaker_profile(
    display_name: String,
    color: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<SpeakerProfileDto, String> {
    let name = display_name.trim();
    if name.is_empty() {
        return Err("Speaker name cannot be empty".to_string());
    }

    let pool = state.db_manager.pool();
    let profile = SpeakerRepository::create_profile(pool, name, color.as_deref())
        .await
        .map_err(|e| format!("Failed to create speaker profile: {}", e))?;

    Ok(profile_to_dto(profile))
}

#[tauri::command]
pub async fn api_rename_speaker_profile(
    profile_id: String,
    display_name: String,
    state: tauri::State<'_, AppState>,
) -> Result<SpeakerProfileDto, String> {
    let name = display_name.trim();
    if name.is_empty() {
        return Err("Speaker name cannot be empty".to_string());
    }

    let pool = state.db_manager.pool();
    let profile = SpeakerRepository::rename_profile(pool, &profile_id, name)
        .await
        .map_err(|e| format!("Failed to rename speaker profile: {}", e))?
        .ok_or_else(|| "Speaker profile not found".to_string())?;

    reconcile_profile_assignments(pool, &profile_id, &profile.display_name)
        .await
        .map_err(|e| format!("Failed to update transcript speaker labels: {}", e))?;

    Ok(profile_to_dto(profile))
}

#[tauri::command]
pub async fn api_identify_meeting_speakers(
    meeting_id: String,
    state: tauri::State<'_, AppState>,
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<SpeakerIdentificationResult, String> {
    let pool = state.db_manager.pool();
    identify_meeting_speakers(pool, &sidecar, &meeting_id).await
}

async fn identify_meeting_speakers(
    pool: &SqlitePool,
    sidecar: &SpeakerSidecarState,
    meeting_id: &str,
) -> Result<SpeakerIdentificationResult, String> {
    let audio_path = resolve_meeting_audio_path(pool, meeting_id).await?;
    sidecar.ensure_running().await?;
    let client = sidecar_client()?;

    let diarize = post_sidecar::<_, SidecarDiarizeResponse>(
        &client,
        "/diarize",
        &SidecarDiarizeRequest {
            audio_path: audio_path.clone(),
            num_speakers: None,
            min_speakers: None,
            max_speakers: None,
        },
    )
    .await?;

    if diarize.turns.is_empty() {
        return Ok(SpeakerIdentificationResult {
            updated_transcript_count: 0,
            speaker_turn_count: 0,
            speakers: Vec::new(),
        });
    }

    let profiles = SpeakerRepository::list_profiles(pool)
        .await
        .map_err(|e| format!("Failed to load speaker profiles: {}", e))?;
    let voiceprints = SpeakerRepository::list_voiceprints(pool)
        .await
        .map_err(|e| format!("Failed to load speaker voiceprints: {}", e))?;

    let profile_by_id = profiles
        .iter()
        .map(|profile| (profile.id.clone(), profile.clone()))
        .collect::<HashMap<_, _>>();

    let sidecar_profiles = voiceprints
        .into_iter()
        .filter_map(|voiceprint| {
            let profile = profile_by_id.get(&voiceprint.speaker_profile_id)?;
            Some(SidecarProfile {
                profile_id: profile.id.clone(),
                display_name: profile.display_name.clone(),
                embedding_model: voiceprint.embedding_model,
                embedding: bytes_to_f32_vec(&voiceprint.embedding),
            })
        })
        .collect::<Vec<_>>();

    let cluster_matches = if sidecar_profiles.is_empty() {
        Vec::new()
    } else {
        post_sidecar::<_, SidecarIdentifyResponse>(
            &client,
            "/identify",
            &SidecarIdentifyRequest {
                audio_path: audio_path.clone(),
                turns: diarize.turns.clone(),
                profiles: sidecar_profiles,
                threshold: DEFAULT_RECOGNITION_THRESHOLD,
                ambiguity_margin: DEFAULT_AMBIGUITY_MARGIN,
            },
        )
        .await?
        .clusters
    };

    let match_by_cluster = cluster_matches
        .into_iter()
        .map(|item| (item.cluster_label.clone(), item))
        .collect::<HashMap<_, _>>();

    let assignments = diarize
        .turns
        .iter()
        .map(|turn| {
            let cluster_match = match_by_cluster.get(&turn.cluster_label);
            let profile_id = cluster_match.and_then(|m| m.profile_id.clone());
            let speaker_label = match cluster_match {
                Some(m) if m.profile_id.is_some() && !m.ambiguous => m
                    .display_name
                    .clone()
                    .unwrap_or_else(|| display_cluster_label(&turn.cluster_label)),
                Some(m) if m.ambiguous => m
                    .display_name
                    .as_ref()
                    .map(|name| format!("Maybe {}", name))
                    .unwrap_or_else(|| display_cluster_label(&turn.cluster_label)),
                _ => display_cluster_label(&turn.cluster_label),
            };

            SpeakerTurnAssignment {
                cluster_label: turn.cluster_label.clone(),
                speaker_profile_id: profile_id,
                speaker_label,
                start_time: turn.start,
                end_time: turn.end,
                confidence: cluster_match.and_then(|m| m.confidence),
                confirmed: false,
                assignment_source: match cluster_match {
                    Some(m) if m.ambiguous => "ambiguous_recognition".to_string(),
                    Some(m) if m.profile_id.is_some() => "recognition".to_string(),
                    _ => "diarization".to_string(),
                },
            }
        })
        .collect::<Vec<_>>();

    SpeakerTurnRepository::replace_meeting_turns(pool, meeting_id, &assignments)
        .await
        .map_err(|e| format!("Failed to save speaker turns: {}", e))?;
    let updated =
        SpeakerTurnRepository::assign_transcripts_by_overlap(pool, meeting_id, &assignments)
            .await
            .map_err(|e| format!("Failed to assign transcript speakers: {}", e))?;

    Ok(SpeakerIdentificationResult {
        updated_transcript_count: updated,
        speaker_turn_count: assignments.len(),
        speakers: assignments.into_iter().map(assignment_to_dto).collect(),
    })
}

async fn propagate_speaker_recognition_to_other_meetings(
    pool: &SqlitePool,
    sidecar: &SpeakerSidecarState,
    source_meeting_id: &str,
) -> SpeakerPropagationSummary {
    let candidates: Vec<(String, Option<String>, i64)> = match sqlx::query_as(
        "SELECT m.id,
                m.folder_path,
                COALESCE(SUM(CASE WHEN st.confirmed = 1 THEN 1 ELSE 0 END), 0) AS confirmed_turn_count
         FROM meetings m
         LEFT JOIN speaker_turns st ON st.meeting_id = m.id
         GROUP BY m.id, m.folder_path, m.created_at
         ORDER BY m.created_at DESC",
    )
    .fetch_all(pool)
    .await
    {
        Ok(candidates) => candidates,
        Err(error) => {
            warn!(
                "Failed to load speaker propagation candidates after assigning speaker: {}",
                error
            );
            return SpeakerPropagationSummary {
                error_count: 1,
                ..Default::default()
            };
        }
    };

    let mut summary = SpeakerPropagationSummary::default();
    let mut attempted = 0usize;
    for (meeting_id, folder_path, confirmed_turn_count) in candidates {
        if !should_propagate_to_meeting(
            &meeting_id,
            source_meeting_id,
            folder_path.as_deref(),
            confirmed_turn_count,
        ) {
            continue;
        }

        if attempted >= MAX_PROPAGATION_MEETINGS {
            warn!(
                "Speaker propagation capped at {} meetings; remaining eligible meetings were not \
                 auto-identified to avoid a long synchronous re-diarization. Re-run identification \
                 on them individually if needed.",
                MAX_PROPAGATION_MEETINGS
            );
            break;
        }
        attempted += 1;

        match identify_meeting_speakers(pool, sidecar, &meeting_id).await {
            Ok(result) => {
                summary.meeting_count += 1;
                summary.transcript_count += result.updated_transcript_count;
            }
            Err(error) => {
                summary.error_count += 1;
                warn!(
                    meeting_id = %meeting_id,
                    "Failed to propagate speaker recognition after assigning speaker: {}",
                    error
                );
            }
        }
    }

    summary
}

fn should_propagate_to_meeting(
    meeting_id: &str,
    source_meeting_id: &str,
    folder_path: Option<&str>,
    confirmed_turn_count: i64,
) -> bool {
    meeting_id != source_meeting_id
        && folder_path
            .map(|path| !path.trim().is_empty())
            .unwrap_or(false)
        && confirmed_turn_count == 0
}

#[tauri::command]
pub async fn api_assign_speaker_cluster(
    meeting_id: String,
    cluster_label: String,
    profile_id: String,
    learn_voiceprint: bool,
    state: tauri::State<'_, AppState>,
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<SpeakerAssignmentResult, String> {
    let pool = state.db_manager.pool();
    let profile = find_profile(pool, &profile_id).await?;
    let updated_turns = SpeakerTurnRepository::assign_cluster(
        pool,
        &meeting_id,
        &cluster_label,
        &profile_id,
        Some(1.0),
        true,
        "user",
    )
    .await
    .map_err(|e| format!("Failed to assign speaker cluster: {}", e))?;

    let updated_transcripts = assign_current_turns_to_transcripts(pool, &meeting_id).await?;

    let voiceprint_error_count = if learn_voiceprint {
        learn_cluster_voiceprint_best_effort(
            pool,
            &sidecar,
            &meeting_id,
            &cluster_label,
            &profile_id,
        )
        .await
    } else {
        0
    };
    let propagation = if learn_voiceprint {
        propagate_speaker_recognition_to_other_meetings(pool, &sidecar, &meeting_id).await
    } else {
        SpeakerPropagationSummary::default()
    };

    Ok(SpeakerAssignmentResult {
        updated_transcript_count: updated_transcripts,
        updated_turn_count: updated_turns,
        propagated_meeting_count: propagation.meeting_count,
        propagated_transcript_count: propagation.transcript_count,
        propagation_error_count: propagation.error_count + voiceprint_error_count,
        profile: profile_to_dto(profile),
    })
}

#[tauri::command]
pub async fn api_assign_speaker_label(
    meeting_id: String,
    speaker_label: String,
    profile_id: String,
    learn_voiceprint: bool,
    state: tauri::State<'_, AppState>,
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<SpeakerAssignmentResult, String> {
    let pool = state.db_manager.pool();
    let profile = find_profile(pool, &profile_id).await?;
    let turns = SpeakerTurnRepository::list_meeting_turns(pool, &meeting_id)
        .await
        .map_err(|e| format!("Failed to load speaker turns: {}", e))?;
    let profiles = SpeakerRepository::list_profiles(pool)
        .await
        .map_err(|e| format!("Failed to load speaker profiles: {}", e))?;
    let profile_by_id = profiles
        .into_iter()
        .map(|profile| (profile.id.clone(), profile))
        .collect::<HashMap<_, _>>();

    let mut cluster_labels = turns
        .into_iter()
        .filter_map(|turn| {
            let assignment = turn_to_assignment(turn, &profile_by_id);
            (assignment.speaker_label == speaker_label).then_some(assignment.cluster_label)
        })
        .collect::<Vec<_>>();
    cluster_labels.sort();
    cluster_labels.dedup();

    if cluster_labels.is_empty() {
        return Err(
            "Speaker label was not found in this meeting. Run Identify Speakers first.".to_string(),
        );
    }

    let mut updated_turns = 0;
    for cluster_label in &cluster_labels {
        updated_turns += SpeakerTurnRepository::assign_cluster(
            pool,
            &meeting_id,
            cluster_label,
            &profile_id,
            Some(1.0),
            true,
            "user",
        )
        .await
        .map_err(|e| format!("Failed to assign speaker label: {}", e))?;
    }

    let updated_transcripts = assign_current_turns_to_transcripts(pool, &meeting_id).await?;

    let mut voiceprint_error_count = 0;
    if learn_voiceprint {
        for cluster_label in &cluster_labels {
            voiceprint_error_count += learn_cluster_voiceprint_best_effort(
                pool,
                &sidecar,
                &meeting_id,
                cluster_label,
                &profile_id,
            )
            .await;
        }
    }
    let propagation = if learn_voiceprint {
        propagate_speaker_recognition_to_other_meetings(pool, &sidecar, &meeting_id).await
    } else {
        SpeakerPropagationSummary::default()
    };

    Ok(SpeakerAssignmentResult {
        updated_transcript_count: updated_transcripts,
        updated_turn_count: updated_turns,
        propagated_meeting_count: propagation.meeting_count,
        propagated_transcript_count: propagation.transcript_count,
        propagation_error_count: propagation.error_count + voiceprint_error_count,
        profile: profile_to_dto(profile),
    })
}

async fn learn_cluster_voiceprint_best_effort(
    pool: &SqlitePool,
    sidecar: &SpeakerSidecarState,
    meeting_id: &str,
    cluster_label: &str,
    profile_id: &str,
) -> u64 {
    if let Err(error) = sidecar.ensure_running().await {
        warn!(
            cluster_label = %cluster_label,
            "Skipping speaker voiceprint learning because sidecar is unavailable: {}",
            error
        );
        return 1;
    }

    if let Err(error) = learn_cluster_voiceprint(pool, meeting_id, cluster_label, profile_id).await
    {
        warn!(
            cluster_label = %cluster_label,
            "Speaker label was assigned, but voiceprint learning failed: {}",
            error
        );
        return 1;
    }

    0
}

#[tauri::command]
pub async fn api_assign_transcript_speaker(
    transcript_id: String,
    profile_id: String,
    learn_voiceprint: bool,
    state: tauri::State<'_, AppState>,
) -> Result<SpeakerAssignmentResult, String> {
    let pool = state.db_manager.pool();
    let profile = find_profile(pool, &profile_id).await?;

    if learn_voiceprint {
        return Err("Learning from a single transcript segment is not available yet. Assign the full speaker cluster instead.".to_string());
    }

    let result = sqlx::query(
        "UPDATE transcripts
         SET speaker_profile_id = ?, speaker_label = ?, speaker_confidence = ?, speaker_confirmed = ?
         WHERE id = ?",
    )
    .bind(&profile_id)
    .bind(&profile.display_name)
    .bind(1.0_f64)
    .bind(1_i64)
    .bind(&transcript_id)
    .execute(pool)
    .await
    .map_err(|e| format!("Failed to update transcript speaker: {}", e))?;

    Ok(SpeakerAssignmentResult {
        updated_transcript_count: result.rows_affected(),
        updated_turn_count: 0,
        propagated_meeting_count: 0,
        propagated_transcript_count: 0,
        propagation_error_count: 0,
        profile: profile_to_dto(profile),
    })
}

async fn learn_cluster_voiceprint(
    pool: &SqlitePool,
    meeting_id: &str,
    cluster_label: &str,
    profile_id: &str,
) -> Result<(), String> {
    let audio_path = resolve_meeting_audio_path(pool, meeting_id).await?;
    let turns = SpeakerTurnRepository::list_meeting_turns(pool, meeting_id)
        .await
        .map_err(|e| format!("Failed to load speaker turns: {}", e))?
        .into_iter()
        .filter(|turn| turn.cluster_label == cluster_label)
        .map(|turn| SidecarTurn {
            cluster_label: turn.cluster_label,
            start: turn.start_time,
            end: turn.end_time,
        })
        .collect::<Vec<_>>();

    if turns.is_empty() {
        return Err("No speaker turns found for that cluster".to_string());
    }

    let client = sidecar_client()?;
    let embedding = post_sidecar::<_, SidecarEmbedClusterResponse>(
        &client,
        "/embed-cluster",
        &SidecarEmbedClusterRequest {
            audio_path,
            turns,
            max_turns: 8,
            min_turn_duration: 1.5,
        },
    )
    .await?;

    SpeakerRepository::upsert_voiceprint(
        pool,
        profile_id,
        &embedding.embedding_model,
        f32_vec_to_bytes(&embedding.embedding),
        embedding.sample_count,
        embedding.total_duration,
        Some(meeting_id),
        Some(cluster_label),
    )
    .await
    .map_err(|e| format!("Failed to save speaker voiceprint: {}", e))?;

    Ok(())
}

async fn find_profile(pool: &SqlitePool, profile_id: &str) -> Result<SpeakerProfile, String> {
    SpeakerRepository::list_profiles(pool)
        .await
        .map_err(|e| format!("Failed to load speaker profiles: {}", e))?
        .into_iter()
        .find(|profile| profile.id == profile_id)
        .ok_or_else(|| "Speaker profile not found".to_string())
}

async fn assign_current_turns_to_transcripts(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<u64, String> {
    let turns = SpeakerTurnRepository::list_meeting_turns(pool, meeting_id)
        .await
        .map_err(|e| format!("Failed to load speaker turns: {}", e))?;
    let profiles = SpeakerRepository::list_profiles(pool)
        .await
        .map_err(|e| format!("Failed to load speaker profiles: {}", e))?;
    let profile_by_id = profiles
        .into_iter()
        .map(|profile| (profile.id.clone(), profile))
        .collect::<HashMap<_, _>>();
    let assignments = turns
        .into_iter()
        .map(|turn| turn_to_assignment(turn, &profile_by_id))
        .collect::<Vec<_>>();

    SpeakerTurnRepository::assign_transcripts_by_overlap(pool, meeting_id, &assignments)
        .await
        .map_err(|e| format!("Failed to update transcript speakers: {}", e))
}

async fn reconcile_profile_assignments(
    pool: &SqlitePool,
    profile_id: &str,
    display_name: &str,
) -> Result<u64, sqlx::Error> {
    let rows: Vec<(String, Option<String>)> =
        sqlx::query_as("SELECT id, speaker_label FROM transcripts WHERE speaker_profile_id = ?")
            .bind(profile_id)
            .fetch_all(pool)
            .await?;

    let mut updated = 0_u64;
    for (transcript_id, existing_label) in rows {
        let label = reconciled_speaker_label(existing_label.as_deref().unwrap_or(""), display_name);
        let result = sqlx::query(
            "UPDATE transcripts
             SET speaker_label = ?
             WHERE id = ?",
        )
        .bind(label)
        .bind(transcript_id)
        .execute(pool)
        .await?;

        updated += result.rows_affected();
    }

    Ok(updated)
}

async fn resolve_meeting_audio_path(pool: &SqlitePool, meeting_id: &str) -> Result<String, String> {
    let meeting = sqlx::query_as::<_, MeetingModel>(
        "SELECT id, title, created_at, updated_at, folder_path FROM meetings WHERE id = ?",
    )
    .bind(meeting_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| format!("Failed to load meeting: {}", e))?
    .ok_or_else(|| "Meeting not found".to_string())?;

    let folder_path = meeting
        .folder_path
        .filter(|path| !path.trim().is_empty())
        .ok_or_else(|| "Meeting has no recording folder path".to_string())?;
    let folder = PathBuf::from(folder_path);

    if let Some(path) = audio_path_from_metadata(&folder)? {
        return Ok(path.to_string_lossy().to_string());
    }

    for candidate in ["audio.mp4", "audio.m4a", "audio.wav", "audio.mp3"] {
        let path = folder.join(candidate);
        if path.exists() {
            return Ok(path.to_string_lossy().to_string());
        }
    }

    Err(format!(
        "No supported audio file found in {}",
        folder.display()
    ))
}

fn audio_path_from_metadata(folder: &std::path::Path) -> Result<Option<PathBuf>, String> {
    let metadata_path = folder.join("metadata.json");
    if !metadata_path.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read metadata.json: {}", e))?;
    let metadata: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse metadata.json: {}", e))?;
    let Some(audio_file) = metadata.get("audio_file").and_then(|value| value.as_str()) else {
        return Ok(None);
    };
    if audio_file.trim().is_empty() {
        return Ok(None);
    }

    let path = PathBuf::from(audio_file);
    let resolved = if path.is_absolute() {
        path
    } else {
        folder.join(path)
    };

    Ok(resolved.exists().then_some(resolved))
}

fn sidecar_client() -> Result<Client, String> {
    Client::builder()
        .timeout(Duration::from_secs(600))
        .connect_timeout(Duration::from_secs(3))
        .build()
        .map_err(|e| format!("Failed to create speaker sidecar client: {}", e))
}

async fn post_sidecar<T, R>(client: &Client, path: &str, body: &T) -> Result<R, String>
where
    T: Serialize + ?Sized,
    R: for<'de> Deserialize<'de>,
{
    let base_url = std::env::var("MEETILY_DIARIZATION_URL")
        .unwrap_or_else(|_| DEFAULT_SIDECAR_URL.to_string());
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);

    let response = client.post(url).json(body).send().await.map_err(|e| {
        format!(
            "Speaker recognition service is not running or did not respond: {}",
            e
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!(
            "Speaker recognition service failed ({status}): {text}"
        ));
    }

    response
        .json::<R>()
        .await
        .map_err(|e| format!("Failed to parse speaker recognition response: {}", e))
}

fn profile_to_dto(profile: SpeakerProfile) -> SpeakerProfileDto {
    SpeakerProfileDto {
        id: profile.id,
        display_name: profile.display_name,
        color: profile.color,
    }
}

fn assignment_to_dto(assignment: SpeakerTurnAssignment) -> SpeakerTurnDto {
    SpeakerTurnDto {
        cluster_label: assignment.cluster_label,
        speaker_profile_id: assignment.speaker_profile_id,
        speaker_label: assignment.speaker_label,
        start_time: assignment.start_time,
        end_time: assignment.end_time,
        confidence: assignment.confidence,
        confirmed: assignment.confirmed,
    }
}

fn turn_to_assignment(
    turn: crate::database::models::SpeakerTurn,
    profile_by_id: &HashMap<String, SpeakerProfile>,
) -> SpeakerTurnAssignment {
    let profile_label = turn
        .speaker_profile_id
        .as_ref()
        .and_then(|id| profile_by_id.get(id))
        .map(|profile| profile.display_name.clone());
    let speaker_label = match (turn.assignment_source.as_str(), profile_label) {
        ("ambiguous_recognition", Some(label)) => format!("Maybe {}", label),
        (_, Some(label)) => label,
        _ => display_cluster_label(&turn.cluster_label),
    };

    SpeakerTurnAssignment {
        cluster_label: turn.cluster_label,
        speaker_profile_id: turn.speaker_profile_id,
        speaker_label,
        start_time: turn.start_time,
        end_time: turn.end_time,
        confidence: turn.confidence,
        confirmed: turn.confirmed != 0,
        assignment_source: turn.assignment_source,
    }
}

fn display_cluster_label(cluster_label: &str) -> String {
    if let Some(number) = cluster_label.strip_prefix("SPEAKER_") {
        if let Ok(index) = number.parse::<usize>() {
            return format!("Speaker {}", index + 1);
        }
    }
    cluster_label.to_string()
}

fn reconciled_speaker_label(existing_label: &str, new_name: &str) -> String {
    if existing_label.trim_start().starts_with("Maybe ") {
        format!("Maybe {}", new_name)
    } else {
        new_name.to_string()
    }
}

fn f32_vec_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::repositories::speaker_turn::SpeakerTurnRepository;
    use sqlx::sqlite::SqlitePoolOptions;

    #[test]
    fn display_cluster_label_formats_pyannote_labels() {
        assert_eq!(display_cluster_label("SPEAKER_00"), "Speaker 1");
        assert_eq!(display_cluster_label("SPEAKER_12"), "Speaker 13");
        assert_eq!(display_cluster_label("UNKNOWN"), "UNKNOWN");
    }

    #[test]
    fn renamed_profile_label_preserves_ambiguous_prefix() {
        assert_eq!(
            reconciled_speaker_label("Maybe Alice", "Francesco"),
            "Maybe Francesco"
        );
    }

    #[test]
    fn renamed_profile_label_replaces_confirmed_name() {
        assert_eq!(reconciled_speaker_label("Alice", "Francesco"), "Francesco");
    }

    #[test]
    fn renamed_profile_label_replaces_empty_cached_label() {
        assert_eq!(reconciled_speaker_label("", "Francesco"), "Francesco");
    }

    #[test]
    fn f32_bytes_round_trip() {
        let values = vec![0.1_f32, -0.25, 1.5];
        let bytes = f32_vec_to_bytes(&values);

        assert_eq!(bytes_to_f32_vec(&bytes), values);
    }

    #[test]
    fn propagation_candidate_requires_other_unconfirmed_meeting_with_audio_folder() {
        assert!(should_propagate_to_meeting(
            "meeting-old",
            "meeting-current",
            Some("/tmp/recording"),
            0
        ));
        assert!(!should_propagate_to_meeting(
            "meeting-current",
            "meeting-current",
            Some("/tmp/recording"),
            0
        ));
        assert!(!should_propagate_to_meeting(
            "meeting-old",
            "meeting-current",
            Some("   "),
            0
        ));
        assert!(!should_propagate_to_meeting(
            "meeting-old",
            "meeting-current",
            Some("/tmp/recording"),
            1
        ));
    }

    #[test]
    fn turn_to_assignment_preserves_ambiguous_profile_label() {
        let profile = SpeakerProfile {
            id: "profile-1".to_string(),
            display_name: "Alice".to_string(),
            color: None,
            created_at: "2026-06-03T00:00:00Z".to_string(),
            updated_at: "2026-06-03T00:00:00Z".to_string(),
            archived_at: None,
        };
        let profile_by_id = HashMap::from([(profile.id.clone(), profile)]);
        let turn = crate::database::models::SpeakerTurn {
            id: "turn-1".to_string(),
            meeting_id: "meeting-1".to_string(),
            cluster_label: "SPEAKER_00".to_string(),
            speaker_profile_id: Some("profile-1".to_string()),
            start_time: 0.0,
            end_time: 2.0,
            confidence: Some(0.74),
            assignment_source: "ambiguous_recognition".to_string(),
            confirmed: 0,
            created_at: "2026-06-03T00:00:00Z".to_string(),
            updated_at: "2026-06-03T00:00:00Z".to_string(),
        };

        let assignment = turn_to_assignment(turn, &profile_by_id);

        assert_eq!(assignment.speaker_label, "Maybe Alice");
    }

    #[tokio::test]
    async fn profile_rename_reconciles_english_and_italian_transcript_labels_without_repointing_ids(
    ) {
        let pool = speaker_test_pool().await;
        insert_profile(&pool, "profile-english", "Alice").await;
        insert_profile(&pool, "profile-italian", "Luca").await;
        insert_transcript(
            &pool,
            "transcript-english",
            "meeting-1",
            Some("profile-english"),
            Some("Alice"),
            0.0,
            2.0,
        )
        .await;
        insert_transcript(
            &pool,
            "transcript-italian-ambiguous",
            "meeting-1",
            Some("profile-italian"),
            Some("Maybe Luca"),
            2.0,
            4.0,
        )
        .await;

        let updated_english = reconcile_profile_assignments(&pool, "profile-english", "Alicia")
            .await
            .unwrap();
        let updated_italian = reconcile_profile_assignments(&pool, "profile-italian", "Marco")
            .await
            .unwrap();

        assert_eq!(updated_english, 1);
        assert_eq!(updated_italian, 1);
        assert_transcript_speaker(
            &pool,
            "transcript-english",
            Some("profile-english"),
            Some("Alicia"),
        )
        .await;
        assert_transcript_speaker(
            &pool,
            "transcript-italian-ambiguous",
            Some("profile-italian"),
            Some("Maybe Marco"),
        )
        .await;
    }

    #[tokio::test]
    async fn assigning_cluster_to_different_profile_repoints_id_and_label_for_overlapping_transcripts(
    ) {
        let pool = speaker_test_pool().await;
        insert_profile(&pool, "profile-english", "Alice").await;
        insert_profile(&pool, "profile-italian", "Luca").await;
        insert_speaker_turn(
            &pool,
            "turn-english",
            "meeting-1",
            "SPEAKER_00",
            Some("profile-english"),
            0.0,
            5.0,
        )
        .await;
        insert_transcript(
            &pool,
            "transcript-overlap",
            "meeting-1",
            Some("profile-english"),
            Some("Alice"),
            1.0,
            3.0,
        )
        .await;

        let updated_turns = SpeakerTurnRepository::assign_cluster(
            &pool,
            "meeting-1",
            "SPEAKER_00",
            "profile-italian",
            Some(1.0),
            true,
            "user",
        )
        .await
        .unwrap();
        let profile_by_id = SpeakerRepository::list_profiles(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|profile| (profile.id.clone(), profile))
            .collect::<HashMap<_, _>>();
        let assignments = SpeakerTurnRepository::list_meeting_turns(&pool, "meeting-1")
            .await
            .unwrap()
            .into_iter()
            .map(|turn| turn_to_assignment(turn, &profile_by_id))
            .collect::<Vec<_>>();
        let updated_transcripts =
            SpeakerTurnRepository::assign_transcripts_by_overlap(&pool, "meeting-1", &assignments)
                .await
                .unwrap();

        assert_eq!(updated_turns, 1);
        assert_eq!(updated_transcripts, 1);
        assert_transcript_speaker(
            &pool,
            "transcript-overlap",
            Some("profile-italian"),
            Some("Luca"),
        )
        .await;
    }

    #[tokio::test]
    async fn overlap_assignment_never_overwrites_user_confirmed_transcript() {
        let pool = speaker_test_pool().await;
        insert_profile(&pool, "profile-auto", "Alice").await;
        insert_profile(&pool, "profile-manual", "Bob").await;

        // A transcript the user manually confirmed as "Bob" (speaker_confirmed = 1,
        // with no speaker_turns row — exactly how api_assign_transcript_speaker writes it).
        sqlx::query(
            "INSERT INTO transcripts
             (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time,
              speaker_profile_id, speaker_label, speaker_confidence, speaker_confirmed)
             VALUES ('transcript-confirmed', 'meeting-1', '[redacted test text]', '00:00',
                     1.0, 3.0, 'profile-manual', 'Bob', NULL, 1)",
        )
        .execute(&pool)
        .await
        .unwrap();

        // An automatic recognition turn overlapping the same range, pointing elsewhere.
        let assignments = vec![SpeakerTurnAssignment {
            cluster_label: "SPEAKER_00".to_string(),
            speaker_profile_id: Some("profile-auto".to_string()),
            speaker_label: "Alice".to_string(),
            start_time: 0.0,
            end_time: 5.0,
            confidence: Some(0.9),
            confirmed: false,
            assignment_source: "recognition".to_string(),
        }];

        let updated =
            SpeakerTurnRepository::assign_transcripts_by_overlap(&pool, "meeting-1", &assignments)
                .await
                .unwrap();

        // The user-confirmed assignment must survive untouched.
        assert_eq!(updated, 0, "confirmed transcript must not be overwritten");
        assert_transcript_speaker(
            &pool,
            "transcript-confirmed",
            Some("profile-manual"),
            Some("Bob"),
        )
        .await;
    }

    async fn speaker_test_pool() -> SqlitePool {
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
            "CREATE TABLE speaker_turns (
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
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE TABLE transcripts (
                id TEXT PRIMARY KEY,
                meeting_id TEXT NOT NULL,
                transcript TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                audio_start_time REAL,
                audio_end_time REAL,
                speaker_profile_id TEXT,
                speaker_label TEXT,
                speaker_confidence REAL,
                speaker_confirmed INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    async fn insert_profile(pool: &SqlitePool, profile_id: &str, display_name: &str) {
        sqlx::query(
            "INSERT INTO speaker_profiles
             (id, display_name, color, created_at, updated_at, archived_at)
             VALUES (?, ?, NULL, '2026-06-04T00:00:00Z', '2026-06-04T00:00:00Z', NULL)",
        )
        .bind(profile_id)
        .bind(display_name)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_speaker_turn(
        pool: &SqlitePool,
        turn_id: &str,
        meeting_id: &str,
        cluster_label: &str,
        profile_id: Option<&str>,
        start_time: f64,
        end_time: f64,
    ) {
        sqlx::query(
            "INSERT INTO speaker_turns
             (id, meeting_id, cluster_label, speaker_profile_id, start_time, end_time,
              confidence, assignment_source, confirmed, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, NULL, 'recognition', 0, '2026-06-04T00:00:00Z', '2026-06-04T00:00:00Z')",
        )
        .bind(turn_id)
        .bind(meeting_id)
        .bind(cluster_label)
        .bind(profile_id)
        .bind(start_time)
        .bind(end_time)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn insert_transcript(
        pool: &SqlitePool,
        transcript_id: &str,
        meeting_id: &str,
        profile_id: Option<&str>,
        speaker_label: Option<&str>,
        start_time: f64,
        end_time: f64,
    ) {
        sqlx::query(
            "INSERT INTO transcripts
             (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time,
              speaker_profile_id, speaker_label, speaker_confidence, speaker_confirmed)
             VALUES (?, ?, '[redacted test text]', '00:00', ?, ?, ?, ?, NULL, 0)",
        )
        .bind(transcript_id)
        .bind(meeting_id)
        .bind(start_time)
        .bind(end_time)
        .bind(profile_id)
        .bind(speaker_label)
        .execute(pool)
        .await
        .unwrap();
    }

    async fn assert_transcript_speaker(
        pool: &SqlitePool,
        transcript_id: &str,
        expected_profile_id: Option<&str>,
        expected_label: Option<&str>,
    ) {
        let row: (Option<String>, Option<String>) = sqlx::query_as(
            "SELECT speaker_profile_id, speaker_label FROM transcripts WHERE id = ?",
        )
        .bind(transcript_id)
        .fetch_one(pool)
        .await
        .unwrap();

        assert_eq!(row.0.as_deref(), expected_profile_id);
        assert_eq!(row.1.as_deref(), expected_label);
    }
}
