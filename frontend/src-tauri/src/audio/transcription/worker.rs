// audio/transcription/worker.rs
//
// Parallel transcription worker pool and chunk processing logic.

use super::engine::TranscriptionEngine;
use super::provider::TranscriptionError;
use crate::audio::AudioChunk;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Runtime};

const EMPTY_TRANSCRIPT_RETRY_MIN_DURATION_SEC: f64 = 0.75;
const EMPTY_TRANSCRIPT_RETRY_MIN_ENERGY: f32 = 0.00001;
const EMPTY_TRANSCRIPT_RETRY_MAX_DURATION_SEC: f64 = 12.0;
const EMPTY_TRANSCRIPT_FINAL_PADDING_SEC: f64 = 0.8;
const EMPTY_TRANSCRIPT_MAX_GAP_SEC: f64 = 3.0;
const MODEL_LOAD_WAIT_TIMEOUT_SEC: u64 = 120;
const MODEL_LOAD_POLL_MS: u64 = 250;
const ASR_TARGET_RMS: f32 = 0.04;
const ASR_NORMALIZE_MIN_RMS: f32 = 0.001;
const ASR_NORMALIZE_MAX_GAIN: f32 = 16.0;
const ASR_PEAK_LIMIT: f32 = 0.95;

// Sequence counter for transcript updates
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

// Speech detection flag - reset per recording session
static SPEECH_DETECTED_EMITTED: AtomicBool = AtomicBool::new(false);
static STATUS_CHUNKS_QUEUED: AtomicU64 = AtomicU64::new(0);
static STATUS_CHUNKS_COMPLETED: AtomicU64 = AtomicU64::new(0);
static STATUS_ACTIVE: AtomicBool = AtomicBool::new(false);
static STATUS_LAST_ACTIVITY_MS: AtomicU64 = AtomicU64::new(0);

/// Reset the speech detected flag for a new recording session
pub fn reset_speech_detected_flag() {
    SPEECH_DETECTED_EMITTED.store(false, Ordering::SeqCst);
    info!(
        "🔍 SPEECH_DETECTED_EMITTED reset to: {}",
        SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst)
    );
}

pub fn get_transcription_status_snapshot() -> (usize, bool, u64) {
    let queued = STATUS_CHUNKS_QUEUED.load(Ordering::SeqCst);
    let completed = STATUS_CHUNKS_COMPLETED.load(Ordering::SeqCst);
    let pending = queued.saturating_sub(completed) as usize;
    let is_processing = STATUS_ACTIVE.load(Ordering::SeqCst) || pending > 0;
    let last_activity = STATUS_LAST_ACTIVITY_MS.load(Ordering::SeqCst);
    let last_activity_ms = if last_activity == 0 {
        0
    } else {
        now_millis().saturating_sub(last_activity)
    };

    (pending, is_processing, last_activity_ms)
}

fn mark_transcription_activity() {
    STATUS_LAST_ACTIVITY_MS.store(now_millis(), Ordering::SeqCst);
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TranscriptUpdate {
    pub text: String,
    pub timestamp: String, // Wall-clock time for reference (e.g., "14:30:05")
    pub source: String,
    pub sequence_id: u64,
    pub chunk_start_time: f64, // Legacy field, kept for compatibility
    pub is_partial: bool,
    pub confidence: f32,
    // NEW: Recording-relative timestamps for playback sync
    pub audio_start_time: f64, // Seconds from recording start (e.g., 125.3)
    pub audio_end_time: f64,   // Seconds from recording start (e.g., 128.6)
    pub duration: f64,         // Segment duration in seconds (e.g., 3.3)
}

// NOTE: get_transcript_history and get_recording_meeting_name functions
// have been moved to recording_commands.rs where they have access to RECORDING_MANAGER

/// Optimized parallel transcription task ensuring ZERO chunk loss
pub fn start_transcription_task<R: Runtime>(
    app: AppHandle<R>,
    transcription_receiver: tokio::sync::mpsc::UnboundedReceiver<AudioChunk>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("🚀 Starting optimized parallel transcription task - guaranteeing zero chunk loss");
        STATUS_CHUNKS_QUEUED.store(0, Ordering::SeqCst);
        STATUS_CHUNKS_COMPLETED.store(0, Ordering::SeqCst);
        STATUS_ACTIVE.store(true, Ordering::SeqCst);
        mark_transcription_activity();

        // Initialize transcription engine (Whisper or Parakeet based on config)
        let transcription_engine = match super::engine::get_or_init_transcription_engine(&app).await
        {
            Ok(engine) => engine,
            Err(e) => {
                error!("Failed to initialize transcription engine: {}", e);
                STATUS_ACTIVE.store(false, Ordering::SeqCst);
                mark_transcription_activity();
                let _ = app.emit("transcription-error", serde_json::json!({
                    "error": e,
                    "userMessage": "Recording failed: Unable to initialize speech recognition. Please check your model settings.",
                    "actionable": true
                }));
                return;
            }
        };

        // Create parallel workers for faster processing while preserving ALL chunks
        const NUM_WORKERS: usize = 1; // Serial processing ensures transcripts emit in chronological order
        let (work_sender, work_receiver) = tokio::sync::mpsc::unbounded_channel::<AudioChunk>();
        let work_receiver = Arc::new(tokio::sync::Mutex::new(work_receiver));

        // Track completion: AtomicU64 for chunks queued, AtomicU64 for chunks completed
        let chunks_queued = Arc::new(AtomicU64::new(0));
        let chunks_completed = Arc::new(AtomicU64::new(0));
        let input_finished = Arc::new(AtomicBool::new(false));

        info!(
            "📊 Starting {} transcription worker{} (serial mode for ordered emission)",
            NUM_WORKERS,
            if NUM_WORKERS == 1 { "" } else { "s" }
        );

        // Spawn worker tasks
        let mut worker_handles = Vec::new();
        for worker_id in 0..NUM_WORKERS {
            let engine_clone = match &transcription_engine {
                TranscriptionEngine::Whisper(e) => TranscriptionEngine::Whisper(e.clone()),
                TranscriptionEngine::Parakeet(e) => TranscriptionEngine::Parakeet(e.clone()),
                TranscriptionEngine::Provider(p) => TranscriptionEngine::Provider(p.clone()),
            };
            let app_clone = app.clone();
            let work_receiver_clone = work_receiver.clone();
            let chunks_completed_clone = chunks_completed.clone();
            let input_finished_clone = input_finished.clone();
            let chunks_queued_clone = chunks_queued.clone();

            let worker_handle = tokio::spawn(async move {
                info!("👷 Worker {} started", worker_id);

                // PRE-VALIDATE model state to avoid repeated async calls per chunk
                let initial_model_loaded = engine_clone.is_model_loaded().await;
                let current_model = engine_clone
                    .get_current_model()
                    .await
                    .unwrap_or_else(|| "unknown".to_string());

                let engine_name = engine_clone.provider_name();

                if initial_model_loaded {
                    info!(
                        "✅ Worker {} pre-validation: {} model '{}' is loaded and ready",
                        worker_id, engine_name, current_model
                    );
                } else {
                    warn!(
                        "⚠️ Worker {} pre-validation: {} model not loaded - chunks may be skipped",
                        worker_id, engine_name
                    );
                }

                let mut pending_empty_chunk: Option<AudioChunk> = None;

                loop {
                    // Try to get a chunk to process
                    let chunk = {
                        let mut receiver = work_receiver_clone.lock().await;
                        receiver.recv().await
                    };

                    match chunk {
                        Some(chunk) => {
                            // PERFORMANCE OPTIMIZATION: Reduce logging in hot path
                            // Only log every 10th chunk per worker to reduce I/O overhead
                            let should_log_this_chunk = chunk.chunk_id % 10 == 0;

                            if should_log_this_chunk {
                                info!(
                                    "👷 Worker {} processing chunk {} with {} samples",
                                    worker_id,
                                    chunk.chunk_id,
                                    chunk.data.len()
                                );
                            }

                            let incoming_duration = audio_chunk_duration_sec(&chunk);
                            let incoming_energy = audio_chunk_energy(&chunk);
                            emit_transcription_diagnostic(
                                &app_clone,
                                "audio_chunk_received",
                                chunk.chunk_id,
                                chunk.timestamp,
                                chunk.timestamp + incoming_duration,
                                Some("chunk entered transcription worker"),
                                None,
                                Some(incoming_energy),
                                None,
                                None,
                            );

                            // Wait for transient startup/reload instead of dropping speech.
                            if !wait_for_model_loaded(
                                &engine_clone,
                                tokio::time::Duration::from_secs(MODEL_LOAD_WAIT_TIMEOUT_SEC),
                                tokio::time::Duration::from_millis(MODEL_LOAD_POLL_MS),
                            )
                            .await
                            {
                                warn!(
                                    "⚠️ Worker {}: Model was not loaded after {}s; dropping chunk {}",
                                    worker_id, MODEL_LOAD_WAIT_TIMEOUT_SEC, chunk.chunk_id
                                );
                                emit_transcription_diagnostic(
                                    &app_clone,
                                    "model_not_ready_dropped",
                                    chunk.chunk_id,
                                    chunk.timestamp,
                                    chunk.timestamp + incoming_duration,
                                    Some("model not loaded before timeout"),
                                    None,
                                    Some(incoming_energy),
                                    None,
                                    None,
                                );
                                let _ = app_clone.emit(
                                    "transcription-warning",
                                    format!(
                                        "Transcription model was not ready after {} seconds; one audio chunk was skipped.",
                                        MODEL_LOAD_WAIT_TIMEOUT_SEC
                                    ),
                                );
                                chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                STATUS_CHUNKS_COMPLETED.fetch_add(1, Ordering::SeqCst);
                                mark_transcription_activity();
                                continue;
                            }

                            let chunk = if let Some(pending) = pending_empty_chunk.take() {
                                let pending_duration = audio_chunk_duration_sec(&pending);
                                let next_duration = audio_chunk_duration_sec(&chunk);
                                if should_merge_empty_retry_with_next(&pending, &chunk) {
                                    info!(
                                        "Worker {} retrying empty speech chunk {} ({:.2}s) with next chunk {} ({:.2}s)",
                                        worker_id,
                                        pending.chunk_id,
                                        pending_duration,
                                        chunk.chunk_id,
                                        next_duration
                                    );
                                    merge_audio_chunks(pending, chunk)
                                } else {
                                    warn!(
                                        "Worker {} retrying empty speech chunk {} separately because next chunk {} starts after a large gap ({:.2}s)",
                                        worker_id,
                                        pending.chunk_id,
                                        chunk.chunk_id,
                                        audio_gap_sec(&pending, &chunk)
                                    );
                                    flush_pending_empty_chunk(
                                        &engine_clone,
                                        &app_clone,
                                        worker_id,
                                        pending,
                                    )
                                    .await;
                                    chunk
                                }
                            } else {
                                chunk
                            };

                            let chunk_timestamp = chunk.timestamp;
                            let chunk_duration = audio_chunk_duration_sec(&chunk);
                            let chunk_energy = audio_chunk_energy(&chunk);

                            // Transcribe with provider-agnostic approach
                            match transcribe_chunk_with_provider(
                                &engine_clone,
                                chunk.clone(),
                                &app_clone,
                            )
                            .await
                            {
                                Ok((transcript, confidence_opt, is_partial)) => {
                                    let confidence_str = match confidence_opt {
                                        Some(c) => format!("{:.2}", c),
                                        None => "N/A".to_string(),
                                    };

                                    info!("🔍 Worker {} transcription result: text='{}', confidence={}, partial={}",
                                          worker_id, transcript, confidence_str, is_partial);

                                    if should_emit_transcript_text(&transcript, confidence_opt) {
                                        let text_length = transcript.chars().count();
                                        emit_transcription_diagnostic(
                                            &app_clone,
                                            "asr_emitted",
                                            chunk.chunk_id,
                                            chunk_timestamp,
                                            chunk_timestamp + chunk_duration,
                                            Some("non-empty transcript emitted"),
                                            confidence_opt,
                                            Some(chunk_energy),
                                            Some(text_length),
                                            Some(is_partial),
                                        );
                                        emit_transcript_update(
                                            &app_clone,
                                            worker_id,
                                            transcript,
                                            confidence_opt,
                                            is_partial,
                                            chunk_timestamp,
                                            chunk_timestamp + chunk_duration,
                                        );
                                    } else if transcript.trim().is_empty()
                                        && should_retry_empty_transcript(&chunk, chunk_energy)
                                    {
                                        warn!(
                                            "Worker {} preserving empty speech chunk {} for context retry (duration={:.2}s, energy={:.6})",
                                            worker_id,
                                            chunk.chunk_id,
                                            chunk_duration,
                                            chunk_energy
                                        );
                                        emit_transcription_diagnostic(
                                            &app_clone,
                                            "asr_empty_retry_pending",
                                            chunk.chunk_id,
                                            chunk_timestamp,
                                            chunk_timestamp + chunk_duration,
                                            Some("empty transcript for speech-like chunk; preserving for context retry"),
                                            confidence_opt,
                                            Some(chunk_energy),
                                            Some(0),
                                            Some(is_partial),
                                        );
                                        pending_empty_chunk = Some(chunk);
                                    } else if transcript.trim().is_empty() {
                                        emit_transcription_diagnostic(
                                            &app_clone,
                                            "asr_empty_no_retry",
                                            chunk.chunk_id,
                                            chunk_timestamp,
                                            chunk_timestamp + chunk_duration,
                                            Some(
                                                "empty transcript for chunk below retry thresholds",
                                            ),
                                            confidence_opt,
                                            Some(chunk_energy),
                                            Some(0),
                                            Some(is_partial),
                                        );
                                    }
                                }
                                Err(e) => {
                                    // Improved error handling with specific cases
                                    match e {
                                        TranscriptionError::AudioTooShort { .. } => {
                                            // Skip silently, this is expected for very short chunks
                                            info!("Worker {}: {}", worker_id, e);
                                            emit_transcription_diagnostic(
                                                &app_clone,
                                                "audio_too_short_dropped",
                                                chunk.chunk_id,
                                                chunk_timestamp,
                                                chunk_timestamp + chunk_duration,
                                                Some("audio chunk too short for transcription"),
                                                None,
                                                Some(chunk_energy),
                                                None,
                                                None,
                                            );
                                            chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                            STATUS_CHUNKS_COMPLETED.fetch_add(1, Ordering::SeqCst);
                                            mark_transcription_activity();
                                            continue;
                                        }
                                        TranscriptionError::ModelNotLoaded => {
                                            warn!(
                                                "Worker {}: Model unloaded during transcription",
                                                worker_id
                                            );
                                            emit_transcription_diagnostic(
                                                &app_clone,
                                                "model_unloaded_during_transcription",
                                                chunk.chunk_id,
                                                chunk_timestamp,
                                                chunk_timestamp + chunk_duration,
                                                Some("model unloaded during transcription"),
                                                None,
                                                Some(chunk_energy),
                                                None,
                                                None,
                                            );
                                            chunks_completed_clone.fetch_add(1, Ordering::SeqCst);
                                            STATUS_CHUNKS_COMPLETED.fetch_add(1, Ordering::SeqCst);
                                            mark_transcription_activity();
                                            continue;
                                        }
                                        _ => {
                                            warn!(
                                                "Worker {}: Transcription failed: {}",
                                                worker_id, e
                                            );
                                            let reason = e.to_string();
                                            emit_transcription_diagnostic(
                                                &app_clone,
                                                "transcription_error",
                                                chunk.chunk_id,
                                                chunk_timestamp,
                                                chunk_timestamp + chunk_duration,
                                                Some(&reason),
                                                None,
                                                Some(chunk_energy),
                                                None,
                                                None,
                                            );
                                            let _ = app_clone
                                                .emit("transcription-warning", e.to_string());
                                        }
                                    }
                                }
                            }

                            // Mark chunk as completed
                            let completed =
                                chunks_completed_clone.fetch_add(1, Ordering::SeqCst) + 1;
                            STATUS_CHUNKS_COMPLETED.fetch_add(1, Ordering::SeqCst);
                            mark_transcription_activity();
                            let queued = chunks_queued_clone.load(Ordering::SeqCst);

                            // PERFORMANCE: Only log progress every 5th chunk to reduce I/O overhead
                            if completed % 5 == 0 || should_log_this_chunk {
                                info!(
                                    "Worker {}: Progress {}/{} chunks ({:.1}%)",
                                    worker_id,
                                    completed,
                                    queued,
                                    (completed as f64 / queued.max(1) as f64 * 100.0)
                                );
                            }

                            // Emit progress event for frontend
                            let progress_percentage = if queued > 0 {
                                (completed as f64 / queued as f64 * 100.0) as u32
                            } else {
                                100
                            };

                            let _ = app_clone.emit("transcription-progress", serde_json::json!({
                                "worker_id": worker_id,
                                "chunks_completed": completed,
                                "chunks_queued": queued,
                                "progress_percentage": progress_percentage,
                                "message": format!("Worker {} processing... ({}/{})", worker_id, completed, queued)
                            }));
                        }
                        None => {
                            // No more chunks available
                            if input_finished_clone.load(Ordering::SeqCst) {
                                // Double-check that all queued chunks are actually completed
                                let final_queued = chunks_queued_clone.load(Ordering::SeqCst);
                                let final_completed = chunks_completed_clone.load(Ordering::SeqCst);

                                if final_completed >= final_queued {
                                    if let Some(pending) = pending_empty_chunk.take() {
                                        flush_pending_empty_chunk(
                                            &engine_clone,
                                            &app_clone,
                                            worker_id,
                                            pending,
                                        )
                                        .await;
                                    }
                                    info!(
                                        "👷 Worker {} finishing - all {}/{} chunks processed",
                                        worker_id, final_completed, final_queued
                                    );
                                    break;
                                } else {
                                    warn!("👷 Worker {} detected potential chunk loss: {}/{} completed, waiting...", worker_id, final_completed, final_queued);
                                    // AGGRESSIVE POLLING: Reduced from 50ms to 5ms for faster chunk detection during shutdown
                                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                                }
                            } else {
                                // AGGRESSIVE POLLING: Reduced from 10ms to 1ms for faster response during shutdown
                                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            }
                        }
                    }
                }

                info!("👷 Worker {} completed", worker_id);
            });

            worker_handles.push(worker_handle);
        }

        // Main dispatcher: receive chunks and distribute to workers
        let mut receiver = transcription_receiver;
        while let Some(chunk) = receiver.recv().await {
            let queued = chunks_queued.fetch_add(1, Ordering::SeqCst) + 1;
            STATUS_CHUNKS_QUEUED.fetch_add(1, Ordering::SeqCst);
            mark_transcription_activity();
            info!(
                "📥 Dispatching chunk {} to workers (total queued: {})",
                chunk.chunk_id, queued
            );

            if let Err(_) = work_sender.send(chunk) {
                error!("❌ Failed to send chunk to workers - this should not happen!");
                break;
            }
        }

        // Signal that input is finished
        input_finished.store(true, Ordering::SeqCst);
        drop(work_sender); // Close the channel to signal workers

        let total_chunks_queued = chunks_queued.load(Ordering::SeqCst);
        info!("📭 Input finished with {} total chunks queued. Waiting for all {} workers to complete...",
              total_chunks_queued, NUM_WORKERS);

        // Emit final chunk count to frontend
        let _ = app.emit("transcription-queue-complete", serde_json::json!({
            "total_chunks": total_chunks_queued,
            "message": format!("{} chunks queued for processing - waiting for completion", total_chunks_queued)
        }));

        // Wait for all workers to complete
        for (worker_id, handle) in worker_handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                error!("❌ Worker {} panicked: {:?}", worker_id, e);
            } else {
                info!("✅ Worker {} completed successfully", worker_id);
            }
        }

        // Final verification with retry logic to catch any stragglers
        let mut verification_attempts = 0;
        const MAX_VERIFICATION_ATTEMPTS: u32 = 10;

        loop {
            let final_queued = chunks_queued.load(Ordering::SeqCst);
            let final_completed = chunks_completed.load(Ordering::SeqCst);

            if final_queued == final_completed {
                info!(
                    "🎉 ALL {} chunks processed successfully - ZERO chunks lost!",
                    final_completed
                );
                break;
            } else if verification_attempts < MAX_VERIFICATION_ATTEMPTS {
                verification_attempts += 1;
                warn!("⚠️ Chunk count mismatch (attempt {}): {} queued, {} completed - waiting for stragglers...",
                     verification_attempts, final_queued, final_completed);

                // Wait a bit for any remaining chunks to be processed
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            } else {
                error!(
                    "❌ CRITICAL: After {} attempts, chunk loss detected: {} queued, {} completed",
                    MAX_VERIFICATION_ATTEMPTS, final_queued, final_completed
                );

                // Emit critical error event
                let _ = app.emit(
                    "transcript-chunk-loss-detected",
                    serde_json::json!({
                        "chunks_queued": final_queued,
                        "chunks_completed": final_completed,
                        "chunks_lost": final_queued - final_completed,
                        "message": "Some transcript chunks may have been lost during shutdown"
                    }),
                );
                break;
            }
        }

        STATUS_ACTIVE.store(false, Ordering::SeqCst);
        mark_transcription_activity();
        info!("✅ Parallel transcription task completed - all workers finished, ready for model unload");
    })
}

fn audio_chunk_duration_sec(chunk: &AudioChunk) -> f64 {
    if chunk.sample_rate == 0 {
        return 0.0;
    }
    chunk.data.len() as f64 / chunk.sample_rate as f64
}

fn audio_chunk_energy(chunk: &AudioChunk) -> f32 {
    if chunk.data.is_empty() {
        return 0.0;
    }

    chunk.data.iter().map(|sample| sample * sample).sum::<f32>() / chunk.data.len() as f32
}

fn normalize_for_asr(mut samples: Vec<f32>) -> Vec<f32> {
    if samples.is_empty() {
        return samples;
    }

    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < ASR_NORMALIZE_MIN_RMS {
        return samples;
    }

    let gain = (ASR_TARGET_RMS / rms).clamp(1.0, ASR_NORMALIZE_MAX_GAIN);
    if gain <= 1.0 {
        return samples;
    }

    for sample in &mut samples {
        *sample = (*sample * gain).clamp(-ASR_PEAK_LIMIT, ASR_PEAK_LIMIT);
    }

    samples
}

fn should_retry_empty_transcript(chunk: &AudioChunk, energy: f32) -> bool {
    let duration = audio_chunk_duration_sec(chunk);
    duration >= EMPTY_TRANSCRIPT_RETRY_MIN_DURATION_SEC
        && duration <= EMPTY_TRANSCRIPT_RETRY_MAX_DURATION_SEC
        && energy >= EMPTY_TRANSCRIPT_RETRY_MIN_ENERGY
}

fn should_emit_transcript_text(transcript: &str, _confidence: Option<f32>) -> bool {
    // Confidence from local streaming chunks is too noisy to be a loss gate.
    // Keep it as metadata, but never discard non-empty speech text because of it.
    !transcript.trim().is_empty()
}

fn emit_transcription_diagnostic<R: Runtime>(
    app: &AppHandle<R>,
    event: &str,
    chunk_id: u64,
    audio_start_time: f64,
    audio_end_time: f64,
    reason: Option<&str>,
    confidence: Option<f32>,
    energy: Option<f32>,
    text_length: Option<usize>,
    is_partial: Option<bool>,
) {
    let payload = serde_json::json!({
        "event": event,
        "chunk_id": chunk_id,
        "audio_start_time": audio_start_time,
        "audio_end_time": audio_end_time,
        "duration": (audio_end_time - audio_start_time).max(0.0),
        "reason": reason,
        "confidence": confidence,
        "energy": energy,
        "text_length": text_length,
        "is_partial": is_partial,
    });

    if let Err(e) = app.emit("transcription-diagnostic", payload) {
        warn!("Failed to emit transcription diagnostic event: {}", e);
    }
}

fn audio_gap_sec(first: &AudioChunk, second: &AudioChunk) -> f64 {
    let first_end = first.timestamp + audio_chunk_duration_sec(first);
    (second.timestamp - first_end).max(0.0)
}

fn should_merge_empty_retry_with_next(pending: &AudioChunk, next: &AudioChunk) -> bool {
    audio_gap_sec(pending, next) <= EMPTY_TRANSCRIPT_MAX_GAP_SEC
}

fn merge_audio_chunks(mut first: AudioChunk, second: AudioChunk) -> AudioChunk {
    let gap_sec = audio_gap_sec(&first, &second);
    let gap_samples = (gap_sec * first.sample_rate as f64).round() as usize;

    if gap_samples > 0 {
        first.data.extend(std::iter::repeat(0.0).take(gap_samples));
    }

    if second.sample_rate == first.sample_rate {
        first.data.extend(second.data);
    } else {
        let resampled_second = crate::audio::audio_processing::resample_audio(
            &second.data,
            second.sample_rate,
            first.sample_rate,
        );
        first.data.extend(resampled_second);
    }

    first.chunk_id = second.chunk_id;
    first
}

fn pad_audio_chunk(mut chunk: AudioChunk, padding_sec: f64) -> AudioChunk {
    let padding_samples = (padding_sec * chunk.sample_rate as f64).round() as usize;
    chunk
        .data
        .extend(std::iter::repeat(0.0).take(padding_samples));
    chunk
}

async fn wait_for_model_loaded(
    engine: &TranscriptionEngine,
    timeout: tokio::time::Duration,
    poll_interval: tokio::time::Duration,
) -> bool {
    if engine.is_model_loaded().await {
        return true;
    }

    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(poll_interval).await;
        if engine.is_model_loaded().await {
            return true;
        }
    }

    false
}

fn emit_transcript_update<R: Runtime>(
    app: &AppHandle<R>,
    worker_id: usize,
    transcript: String,
    confidence_opt: Option<f32>,
    is_partial: bool,
    audio_start_time: f64,
    audio_end_time: f64,
) {
    let confidence_str = match confidence_opt {
        Some(c) => format!("{:.2}", c),
        None => "N/A".to_string(),
    };

    info!(
        "✅ Worker {} transcribed: {} (confidence: {}, partial: {})",
        worker_id, transcript, confidence_str, is_partial
    );

    let current_flag = SPEECH_DETECTED_EMITTED.load(Ordering::SeqCst);
    info!(
        "🔍 Checking speech-detected flag: current={}, will_emit={}",
        current_flag, !current_flag
    );

    if !current_flag {
        SPEECH_DETECTED_EMITTED.store(true, Ordering::SeqCst);
        match app.emit(
            "speech-detected",
            serde_json::json!({
                "message": "Speech activity detected"
            }),
        ) {
            Ok(_) => {
                info!("🎤 ✅ First speech detected - successfully emitted speech-detected event")
            }
            Err(e) => error!("🎤 ❌ Failed to emit speech-detected event: {}", e),
        }
    } else {
        info!("🔍 Speech already detected in this session, not re-emitting");
    }

    let sequence_id = SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let update = TranscriptUpdate {
        text: transcript,
        timestamp: format_current_timestamp(),
        source: "Audio".to_string(),
        sequence_id,
        chunk_start_time: audio_start_time,
        is_partial,
        confidence: confidence_opt.unwrap_or(0.85),
        audio_start_time,
        audio_end_time,
        duration: (audio_end_time - audio_start_time).max(0.0),
    };

    if let Err(e) = app.emit("transcript-update", &update) {
        error!(
            "Worker {}: Failed to emit transcript update: {}",
            worker_id, e
        );
    }
}

async fn flush_pending_empty_chunk<R: Runtime>(
    engine: &TranscriptionEngine,
    app: &AppHandle<R>,
    worker_id: usize,
    pending: AudioChunk,
) {
    let original_start = pending.timestamp;
    let original_end = pending.timestamp + audio_chunk_duration_sec(&pending);
    let pending_id = pending.chunk_id;
    let padded = pad_audio_chunk(pending, EMPTY_TRANSCRIPT_FINAL_PADDING_SEC);

    info!(
        "Worker {} retrying final empty speech chunk {} with {:.1}s trailing silence",
        worker_id, pending_id, EMPTY_TRANSCRIPT_FINAL_PADDING_SEC
    );

    match transcribe_chunk_with_provider(engine, padded, app).await {
        Ok((transcript, confidence_opt, is_partial)) => {
            if transcript.trim().is_empty() {
                warn!(
                    "Worker {} dropping final speech chunk {} after context retry still returned empty text",
                    worker_id, pending_id
                );
                emit_transcription_diagnostic(
                    app,
                    "asr_empty_retry_dropped",
                    pending_id,
                    original_start,
                    original_end,
                    Some("context retry returned empty transcript"),
                    confidence_opt,
                    None,
                    Some(0),
                    Some(is_partial),
                );
                return;
            }

            let text_length = transcript.chars().count();
            emit_transcription_diagnostic(
                app,
                "asr_retry_emitted",
                pending_id,
                original_start,
                original_end,
                Some("context retry emitted non-empty transcript"),
                confidence_opt,
                None,
                Some(text_length),
                Some(is_partial),
            );
            emit_transcript_update(
                app,
                worker_id,
                transcript,
                confidence_opt,
                is_partial,
                original_start,
                original_end,
            );
        }
        Err(e) => {
            let reason = e.to_string();
            emit_transcription_diagnostic(
                app,
                "asr_retry_error",
                pending_id,
                original_start,
                original_end,
                Some(&reason),
                None,
                None,
                None,
                None,
            );
            warn!(
                "Worker {} final empty speech chunk {} retry failed: {}",
                worker_id, pending_id, e
            );
        }
    }
}

/// Transcribe audio chunk using the appropriate provider (Whisper, Parakeet, or trait-based)
/// Returns: (text, confidence Option, is_partial)
async fn transcribe_chunk_with_provider<R: Runtime>(
    engine: &TranscriptionEngine,
    chunk: AudioChunk,
    app: &AppHandle<R>,
) -> std::result::Result<(String, Option<f32>, bool), TranscriptionError> {
    // Convert to 16kHz mono for transcription
    let transcription_data = if chunk.sample_rate != 16000 {
        crate::audio::audio_processing::resample_audio(&chunk.data, chunk.sample_rate, 16000)
    } else {
        chunk.data
    };

    let speech_samples = normalize_for_asr(transcription_data);

    // Check for empty samples - improved error handling
    if speech_samples.is_empty() {
        warn!(
            "Audio chunk {} is empty, skipping transcription",
            chunk.chunk_id
        );
        return Err(TranscriptionError::AudioTooShort {
            samples: 0,
            minimum: 1600, // 100ms at 16kHz
        });
    }

    // Calculate energy for logging/monitoring only
    let energy: f32 =
        speech_samples.iter().map(|&x| x * x).sum::<f32>() / speech_samples.len() as f32;
    info!(
        "Processing speech audio chunk {} with {} samples (energy: {:.6})",
        chunk.chunk_id,
        speech_samples.len(),
        energy
    );

    // Transcribe using the appropriate engine (with improved error handling)
    match engine {
        TranscriptionEngine::Whisper(whisper_engine) => {
            // Get language preference from global state
            let language = crate::get_language_preference_internal();

            match whisper_engine
                .transcribe_audio_with_confidence(speech_samples, language)
                .await
            {
                Ok((text, confidence, is_partial)) => {
                    let cleaned_text = text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok((String::new(), Some(confidence), is_partial));
                    }

                    info!(
                        "Whisper transcription complete for chunk {}: '{}' (confidence: {:.2}, partial: {})",
                        chunk.chunk_id, cleaned_text, confidence, is_partial
                    );

                    Ok((cleaned_text, Some(confidence), is_partial))
                }
                Err(e) => {
                    error!(
                        "Whisper transcription failed for chunk {}: {}",
                        chunk.chunk_id, e
                    );

                    let transcription_error = TranscriptionError::EngineFailed(e.to_string());
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": transcription_error.to_string(),
                            "userMessage": format!("Transcription failed: {}", transcription_error),
                            "actionable": false
                        }),
                    );

                    Err(transcription_error)
                }
            }
        }
        TranscriptionEngine::Parakeet(parakeet_engine) => {
            match parakeet_engine.transcribe_audio(speech_samples).await {
                Ok(text) => {
                    let cleaned_text = text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok((String::new(), None, false));
                    }

                    info!(
                        "Parakeet transcription complete for chunk {}: '{}'",
                        chunk.chunk_id, cleaned_text
                    );

                    // Parakeet doesn't provide confidence or partial results
                    Ok((cleaned_text, None, false))
                }
                Err(e) => {
                    error!(
                        "Parakeet transcription failed for chunk {}: {}",
                        chunk.chunk_id, e
                    );

                    let transcription_error = TranscriptionError::EngineFailed(e.to_string());
                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": transcription_error.to_string(),
                            "userMessage": format!("Transcription failed: {}", transcription_error),
                            "actionable": false
                        }),
                    );

                    Err(transcription_error)
                }
            }
        }
        TranscriptionEngine::Provider(provider) => {
            // NEW: Trait-based provider (clean, unified interface)
            let language = crate::get_language_preference_internal();

            match provider.transcribe(speech_samples, language).await {
                Ok(result) => {
                    let cleaned_text = result.text.trim().to_string();
                    if cleaned_text.is_empty() {
                        return Ok((String::new(), result.confidence, result.is_partial));
                    }

                    let confidence_str = match result.confidence {
                        Some(c) => format!("confidence: {:.2}", c),
                        None => "no confidence".to_string(),
                    };

                    info!(
                        "{} transcription complete for chunk {}: '{}' ({}, partial: {})",
                        provider.provider_name(),
                        chunk.chunk_id,
                        cleaned_text,
                        confidence_str,
                        result.is_partial
                    );

                    Ok((cleaned_text, result.confidence, result.is_partial))
                }
                Err(e) => {
                    error!(
                        "{} transcription failed for chunk {}: {}",
                        provider.provider_name(),
                        chunk.chunk_id,
                        e
                    );

                    let _ = app.emit(
                        "transcription-error",
                        &serde_json::json!({
                            "error": e.to_string(),
                            "userMessage": format!("Transcription failed: {}", e),
                            "actionable": false
                        }),
                    );

                    Err(e)
                }
            }
        }
    }
}

/// Format current timestamp (wall-clock time)
fn format_current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();

    let hours = (now.as_secs() / 3600) % 24;
    let minutes = (now.as_secs() / 60) % 60;
    let seconds = now.as_secs() % 60;

    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Format recording-relative time as [MM:SS]
#[allow(dead_code)]
fn format_recording_time(seconds: f64) -> String {
    let total_seconds = seconds.floor() as u64;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;

    format!("[{:02}:{:02}]", minutes, secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::recording_state::DeviceType;
    use crate::audio::transcription::provider::{
        TranscriptResult, TranscriptionError, TranscriptionProvider,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn chunk(chunk_id: u64, timestamp: f64, sample_rate: u32, samples: Vec<f32>) -> AudioChunk {
        AudioChunk {
            data: samples,
            sample_rate,
            timestamp,
            chunk_id,
            device_type: DeviceType::Microphone,
        }
    }

    #[test]
    fn retries_empty_transcript_for_energetic_speech_sized_chunk() {
        let audio = chunk(1, 10.0, 16_000, vec![0.1; 16_000]);

        assert!(should_retry_empty_transcript(
            &audio,
            audio_chunk_energy(&audio)
        ));
    }

    #[test]
    fn retries_empty_transcript_for_quiet_non_silent_tail_chunk() {
        let quiet_tail = chunk(4, 32.0, 48_000, vec![0.0033; 48_000 * 6]);

        assert!(should_retry_empty_transcript(
            &quiet_tail,
            audio_chunk_energy(&quiet_tail)
        ));
    }

    #[test]
    fn normalizes_quiet_audio_for_asr_without_amplifying_silence() {
        let quiet = vec![0.0033; 16_000];
        let normalized = normalize_for_asr(quiet);
        let rms = (normalized.iter().map(|sample| sample * sample).sum::<f32>()
            / normalized.len() as f32)
            .sqrt();

        assert!(rms > 0.03, "quiet audio should be boosted for ASR");

        let silence = normalize_for_asr(vec![0.0; 16_000]);
        assert!(silence.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn does_not_retry_empty_transcript_for_tiny_or_silent_chunks() {
        let tiny = chunk(1, 10.0, 16_000, vec![0.1; 2_000]);
        let silent = chunk(2, 10.0, 16_000, vec![0.0; 16_000]);

        assert!(!should_retry_empty_transcript(
            &tiny,
            audio_chunk_energy(&tiny)
        ));
        assert!(!should_retry_empty_transcript(
            &silent,
            audio_chunk_energy(&silent)
        ));
    }

    #[test]
    fn emits_non_empty_transcripts_even_with_low_confidence() {
        assert!(should_emit_transcript_text("ciao", Some(0.01)));
        assert!(should_emit_transcript_text("ciao", None));
        assert!(!should_emit_transcript_text("   ", Some(0.99)));
    }

    #[test]
    fn merge_audio_chunks_preserves_timeline_gap_and_start_time() {
        let first = chunk(10, 1.0, 16_000, vec![0.1; 16_000]);
        let second = chunk(11, 2.5, 16_000, vec![0.2; 8_000]);

        assert!(should_merge_empty_retry_with_next(&first, &second));
        let merged = merge_audio_chunks(first, second);

        assert_eq!(merged.timestamp, 1.0);
        assert_eq!(merged.chunk_id, 11);
        assert_eq!(merged.sample_rate, 16_000);
        assert_eq!(merged.data.len(), 16_000 + 8_000 + 8_000);
    }

    #[test]
    fn does_not_merge_empty_retry_across_large_gap() {
        let first = chunk(10, 1.0, 16_000, vec![0.1; 16_000]);
        let second = chunk(11, 8.5, 16_000, vec![0.2; 8_000]);

        assert_eq!(audio_gap_sec(&first, &second), 6.5);
        assert!(!should_merge_empty_retry_with_next(&first, &second));
    }

    #[test]
    fn final_padding_extends_audio_without_moving_timestamp() {
        let original = chunk(10, 1.0, 16_000, vec![0.1; 16_000]);
        let padded = pad_audio_chunk(original, 0.8);

        assert_eq!(padded.timestamp, 1.0);
        assert_eq!(padded.data.len(), 16_000 + 12_800);
    }

    struct DelayedLoadProvider {
        checks: AtomicUsize,
        ready_after_checks: usize,
    }

    #[async_trait]
    impl TranscriptionProvider for DelayedLoadProvider {
        async fn transcribe(
            &self,
            _audio: Vec<f32>,
            _language: Option<String>,
        ) -> std::result::Result<TranscriptResult, TranscriptionError> {
            Ok(TranscriptResult {
                text: "ready".to_string(),
                confidence: None,
                is_partial: false,
            })
        }

        async fn is_model_loaded(&self) -> bool {
            self.checks.fetch_add(1, Ordering::SeqCst) >= self.ready_after_checks
        }

        async fn get_current_model(&self) -> Option<String> {
            Some("delayed".to_string())
        }

        fn provider_name(&self) -> &'static str {
            "delayed-test-provider"
        }
    }

    #[tokio::test]
    async fn waits_for_model_to_finish_loading_before_dropping_chunk() {
        let provider = Arc::new(DelayedLoadProvider {
            checks: AtomicUsize::new(0),
            ready_after_checks: 2,
        });
        let engine = TranscriptionEngine::Provider(provider);

        assert!(
            wait_for_model_loaded(
                &engine,
                tokio::time::Duration::from_millis(50),
                tokio::time::Duration::from_millis(1),
            )
            .await
        );
    }
}
