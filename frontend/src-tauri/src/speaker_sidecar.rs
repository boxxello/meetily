use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const DEFAULT_URL: &str = "http://127.0.0.1:8179";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8179";
const HEALTH_TIMEOUT_SECS: u64 = 2;
const STARTUP_TIMEOUT_SECS: u64 = 45;
const STARTUP_POLL_MS: u64 = 500;

#[derive(Debug, Clone, Serialize)]
pub struct SpeakerSidecarStatus {
    pub running: bool,
    pub managed: bool,
    pub url: String,
    pub python_path: Option<String>,
    pub sidecar_root: Option<String>,
    pub hf_token_configured: bool,
    pub diarization_loaded: bool,
    pub embedding_loaded: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
struct LaunchConfig {
    python_path: String,
    sidecar_root: PathBuf,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    #[serde(default)]
    hf_token_configured: bool,
    #[serde(default)]
    diarization_loaded: bool,
    #[serde(default)]
    embedding_loaded: bool,
}

#[derive(Clone)]
pub struct SpeakerSidecarState {
    manager: Arc<SpeakerSidecarManager>,
}

impl Default for SpeakerSidecarState {
    fn default() -> Self {
        Self {
            manager: Arc::new(SpeakerSidecarManager::default()),
        }
    }
}

impl SpeakerSidecarState {
    pub async fn status(&self) -> SpeakerSidecarStatus {
        self.manager.status().await
    }

    pub async fn ensure_running(&self) -> Result<SpeakerSidecarStatus, String> {
        self.manager.ensure_running().await
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        self.manager.shutdown().await
    }
}

#[derive(Default)]
struct SpeakerSidecarManager {
    child: Mutex<Option<Child>>,
    launch_config: Mutex<Option<LaunchConfig>>,
}

impl SpeakerSidecarManager {
    async fn status(&self) -> SpeakerSidecarStatus {
        let url = sidecar_url();
        match health_check(&url).await {
            Ok(health) => {
                let config = self.launch_config.lock().await.clone();
                SpeakerSidecarStatus {
                    running: true,
                    managed: self.child.lock().await.is_some(),
                    url,
                    python_path: config.as_ref().map(|c| c.python_path.clone()),
                    sidecar_root: config
                        .as_ref()
                        .map(|c| c.sidecar_root.to_string_lossy().to_string()),
                    hf_token_configured: health.hf_token_configured,
                    diarization_loaded: health.diarization_loaded,
                    embedding_loaded: health.embedding_loaded,
                    message: "Speaker recognition sidecar is running".to_string(),
                }
            }
            Err(error) => SpeakerSidecarStatus {
                running: false,
                managed: self.child.lock().await.is_some(),
                url,
                python_path: None,
                sidecar_root: None,
                hf_token_configured: false,
                diarization_loaded: false,
                embedding_loaded: false,
                message: error,
            },
        }
    }

    async fn ensure_running(&self) -> Result<SpeakerSidecarStatus, String> {
        let url = sidecar_url();
        if let Ok(health) = health_check(&url).await {
            return status_from_health(url, health, None, self.child.lock().await.is_some());
        }

        if std::env::var("MEETILY_DIARIZATION_URL").is_ok() {
            return Err(format!(
                "Speaker recognition service is not reachable at {url}. Check MEETILY_DIARIZATION_URL or start the service manually."
            ));
        }

        let mut child_guard = self.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    log::warn!("Speaker sidecar exited before health check: {}", status);
                    *child_guard = None;
                }
                Ok(None) => {
                    drop(child_guard);
                    let health = wait_for_health(&url).await?;
                    let config = self.launch_config.lock().await.clone();
                    return status_from_health(url, health, config, true);
                }
                Err(error) => {
                    log::warn!("Failed to inspect speaker sidecar process: {}", error);
                    *child_guard = None;
                }
            }
        }

        let config = resolve_launch_config()?;
        log::info!(
            "Starting speaker sidecar with python={} root={}",
            config.python_path,
            config.sidecar_root.display()
        );

        let child = spawn_sidecar(&config)?;
        *child_guard = Some(child);
        *self.launch_config.lock().await = Some(config.clone());
        drop(child_guard);

        let health = wait_for_health(&url).await?;
        status_from_health(url, health, Some(config), true)
    }

    async fn shutdown(&self) -> Result<(), String> {
        let mut child_guard = self.child.lock().await;
        if let Some(mut child) = child_guard.take() {
            if let Err(error) = child.kill().await {
                return Err(format!("Failed to stop speaker sidecar: {error}"));
            }
        }
        *self.launch_config.lock().await = None;
        Ok(())
    }
}

fn status_from_health(
    url: String,
    health: HealthResponse,
    config: Option<LaunchConfig>,
    managed: bool,
) -> Result<SpeakerSidecarStatus, String> {
    if health.status != "ok" {
        return Err(format!(
            "Speaker recognition sidecar returned status '{}'",
            health.status
        ));
    }

    if !health.hf_token_configured {
        return Err(
            "Speaker recognition sidecar is running, but HF_TOKEN or HUGGINGFACE_TOKEN is not configured. Set a Hugging Face token with access to the pyannote model, then restart Meetily or the sidecar."
                .to_string(),
        );
    }

    Ok(SpeakerSidecarStatus {
        running: true,
        managed,
        url,
        python_path: config.as_ref().map(|c| c.python_path.clone()),
        sidecar_root: config
            .as_ref()
            .map(|c| c.sidecar_root.to_string_lossy().to_string()),
        hf_token_configured: health.hf_token_configured,
        diarization_loaded: health.diarization_loaded,
        embedding_loaded: health.embedding_loaded,
        message: "Speaker recognition sidecar is ready".to_string(),
    })
}

async fn wait_for_health(url: &str) -> Result<HealthResponse, String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(STARTUP_TIMEOUT_SECS);
    let mut last_error = "sidecar did not respond yet".to_string();

    while tokio::time::Instant::now() < deadline {
        match health_check(url).await {
            Ok(health) => return Ok(health),
            Err(error) => last_error = error,
        }
        tokio::time::sleep(Duration::from_millis(STARTUP_POLL_MS)).await;
    }

    Err(format!(
        "Speaker recognition sidecar did not become healthy within {STARTUP_TIMEOUT_SECS}s: {last_error}"
    ))
}

async fn health_check(url: &str) -> Result<HealthResponse, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS))
        .connect_timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS))
        .build()
        .map_err(|error| format!("Failed to create health-check client: {error}"))?;
    let health_url = format!("{}/health", url.trim_end_matches('/'));
    let response = client
        .get(health_url)
        .send()
        .await
        .map_err(|error| format!("Speaker recognition sidecar is not running: {error}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Speaker recognition sidecar health check failed with {}",
            response.status()
        ));
    }

    response
        .json::<HealthResponse>()
        .await
        .map_err(|error| format!("Failed to parse speaker sidecar health response: {error}"))
}

fn spawn_sidecar(config: &LaunchConfig) -> Result<Child, String> {
    let mut command = Command::new(&config.python_path);
    command
        .args([
            "-m",
            "uvicorn",
            "backend.diarization_service.main:app",
            "--host",
            DEFAULT_HOST,
            "--port",
            DEFAULT_PORT,
        ])
        .current_dir(&config.sidecar_root)
        .env("PYTHONPATH", &config.sidecar_root)
        .env("MEETILY_DIARIZATION_HOST", DEFAULT_HOST)
        .env("MEETILY_DIARIZATION_PORT", DEFAULT_PORT)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    command
        .spawn()
        .map_err(|error| format!("Failed to start speaker sidecar: {error}"))
}

fn resolve_launch_config() -> Result<LaunchConfig, String> {
    let sidecar_root = resolve_sidecar_root().ok_or_else(|| {
        "Could not find backend/diarization_service. Set MEETILY_DIARIZATION_ROOT to the Meetily repository root or start the sidecar manually.".to_string()
    })?;

    let python_path = resolve_python_path(&sidecar_root).ok_or_else(|| {
        format!(
            "Could not find a Python runtime for speaker recognition. Create the venv at {} or set MEETILY_DIARIZATION_PYTHON.",
            sidecar_root
                .join("backend/diarization_service/.venv/bin/python")
                .display()
        )
    })?;

    Ok(LaunchConfig {
        python_path,
        sidecar_root,
    })
}

fn resolve_sidecar_root() -> Option<PathBuf> {
    if let Ok(root) = std::env::var("MEETILY_DIARIZATION_ROOT") {
        let root = PathBuf::from(root);
        if is_valid_sidecar_root(&root) {
            return Some(root);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.extend(ancestor_candidates(&current_dir));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.extend(ancestor_candidates(exe_dir));
        }
    }
    if let Some(root) = repo_root_from_manifest_dir(Path::new(env!("CARGO_MANIFEST_DIR"))) {
        candidates.push(root);
    }

    candidates
        .into_iter()
        .find(|path| is_valid_sidecar_root(path))
}

fn resolve_python_path(sidecar_root: &Path) -> Option<String> {
    if let Ok(path) = std::env::var("MEETILY_DIARIZATION_PYTHON") {
        if !path.trim().is_empty() {
            return Some(path);
        }
    }

    python_candidates(sidecar_root, None)
        .into_iter()
        .find(|path| Path::new(path).exists())
        .or_else(|| Some("python3".to_string()))
}

fn python_candidates(sidecar_root: &Path, env_python: Option<&str>) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(path) = env_python {
        if !path.trim().is_empty() {
            candidates.push(path.to_string());
        }
    }
    candidates.push(
        sidecar_root
            .join("backend/diarization_service/.venv/bin/python")
            .to_string_lossy()
            .to_string(),
    );
    candidates.push(
        sidecar_root
            .join(".venv/bin/python")
            .to_string_lossy()
            .to_string(),
    );
    candidates
}

fn ancestor_candidates(path: &Path) -> Vec<PathBuf> {
    path.ancestors().map(Path::to_path_buf).collect()
}

fn repo_root_from_manifest_dir(manifest_dir: &Path) -> Option<PathBuf> {
    manifest_dir.parent()?.parent().map(Path::to_path_buf)
}

fn is_valid_sidecar_root(path: &Path) -> bool {
    path.join("backend/diarization_service/main.py").is_file()
}

fn sidecar_url() -> String {
    std::env::var("MEETILY_DIARIZATION_URL").unwrap_or_else(|_| DEFAULT_URL.to_string())
}

#[tauri::command]
pub async fn api_get_speaker_sidecar_status(
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<SpeakerSidecarStatus, String> {
    Ok(sidecar.status().await)
}

#[tauri::command]
pub async fn api_start_speaker_sidecar(
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<SpeakerSidecarStatus, String> {
    sidecar.ensure_running().await
}

#[tauri::command]
pub async fn api_stop_speaker_sidecar(
    sidecar: tauri::State<'_, SpeakerSidecarState>,
) -> Result<(), String> {
    sidecar.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_from_manifest_dir_moves_from_src_tauri_to_repo_root() {
        let root = repo_root_from_manifest_dir(Path::new("/repo/frontend/src-tauri")).unwrap();

        assert_eq!(root, PathBuf::from("/repo"));
    }

    #[test]
    fn python_candidates_prioritize_explicit_python() {
        let candidates = python_candidates(Path::new("/repo"), Some("/custom/python"));

        assert_eq!(candidates[0], "/custom/python");
        assert_eq!(
            candidates[1],
            "/repo/backend/diarization_service/.venv/bin/python"
        );
        assert_eq!(candidates[2], "/repo/.venv/bin/python");
    }

    #[test]
    fn compile_time_repo_root_contains_sidecar_source() {
        let root = repo_root_from_manifest_dir(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();

        assert!(is_valid_sidecar_root(&root));
    }

    #[test]
    fn sidecar_status_from_health_rejects_missing_hf_token() {
        let err = status_from_health(
            DEFAULT_URL.to_string(),
            HealthResponse {
                status: "ok".to_string(),
                hf_token_configured: false,
                diarization_loaded: false,
                embedding_loaded: false,
            },
            None,
            false,
        )
        .unwrap_err();

        assert!(err.contains("HF_TOKEN"));
    }
}
