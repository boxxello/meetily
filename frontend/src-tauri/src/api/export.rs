use log::info as log_info;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Runtime};
use tauri_plugin_dialog::DialogExt;

use crate::api::{MeetingTranscript};
use crate::database::repositories::meeting::MeetingsRepository;
use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportResult {
    pub saved: bool,
    pub path: Option<String>,
    pub segment_count: usize,
    pub skipped_count: Option<usize>,
}

#[tauri::command]
pub async fn api_export_transcript<R: Runtime>(
    app: AppHandle<R>,
    meeting_id: String,
    format: String,
    state: tauri::State<'_, AppState>,
) -> Result<ExportResult, String> {
    log_info!(
        "api_export_transcript: meeting_id={}, format={}",
        meeting_id,
        format
    );

    let pool = state.db_manager.pool();

    let meeting = MeetingsRepository::get_meeting(pool, &meeting_id)
        .await
        .map_err(|e| format!("Failed to load meeting: {}", e))?
        .ok_or_else(|| "Meeting not found".to_string())?;

    if meeting.transcripts.is_empty() {
        return Err("No transcripts to export".to_string());
    }

    let date_str = &meeting.created_at[..10];
    let stem = sanitize_filename_stem(&meeting.title);
    let ext = if format == "vtt" { "vtt" } else { "txt" };
    let default_name = format!("{} - {}.{}", stem, date_str, ext);

    let app_clone = app.clone();
    let ext_owned = ext.to_string();
    let path = tokio::task::spawn_blocking(move || {
        app_clone
            .dialog()
            .file()
            .add_filter("Transcript", &[&ext_owned])
            .set_file_name(&default_name)
            .blocking_save_file()
    })
    .await
    .map_err(|e| format!("Save dialog failed: {}", e))?
    .ok_or_else(|| "Save cancelled".to_string())?;

    let (content, skipped) = match format.as_str() {
        "vtt" => format_vtt(&meeting.transcripts),
        _ => (format_txt(&meeting.transcripts), 0),
    };

    let path_str = path.to_string();
    if let Some(parent) = std::path::Path::new(&path_str).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
    }

    std::fs::write(&path_str, &content)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    log_info!(
        "Exported {} segments to {}",
        meeting.transcripts.len(),
        path_str
    );

    let segment_count = meeting.transcripts.len();
    let skipped_count = if skipped > 0 { Some(skipped) } else { None };

    Ok(ExportResult {
        saved: true,
        path: Some(path_str),
        segment_count,
        skipped_count,
    })
}

fn format_timestamp(seconds: f64) -> String {
    // Round to centiseconds (the display precision) BEFORE splitting into
    // h/m/s. Rounding after the split lets a value like 59.996s render as the
    // invalid ":60.00" because the {:.2} display rounds the seconds field up
    // without carrying into minutes. Carrying here keeps every field in range.
    let total_cs = (seconds.max(0.0) * 100.0).round() as u64;
    let hrs = total_cs / 360_000;
    let mins = (total_cs % 360_000) / 6_000;
    let secs = (total_cs % 6_000) as f64 / 100.0;
    format!("{:02}:{:02}:{:05.2}", hrs, mins, secs)
}

pub fn format_txt(transcripts: &[MeetingTranscript]) -> String {
    let mut sorted: Vec<&MeetingTranscript> = transcripts.iter().collect();
    sorted.sort_by(|a, b| {
        a.audio_start_time
            .partial_cmp(&b.audio_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut output = String::new();
    for t in sorted {
        let text = t.text.replace('\n', " ").replace('\r', "");
        let body = match speaker_display_label(t) {
            Some(label) => format!("{}: {}", label, text),
            None => text,
        };
        if let Some(start) = t.audio_start_time {
            output.push_str(&format!("[{}] {}\n", format_timestamp(start), body));
        } else {
            output.push_str(&body);
            output.push('\n');
        }
    }
    output
}

/// Returns the human-facing speaker label for a transcript segment, if one was
/// assigned and is non-empty. Used to surface diarized/identified speakers in
/// exported transcripts.
fn speaker_display_label(t: &MeetingTranscript) -> Option<&str> {
    t.speaker_label
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

pub fn format_vtt(transcripts: &[MeetingTranscript]) -> (String, usize) {
    let mut sorted: Vec<&MeetingTranscript> = transcripts.iter().collect();
    sorted.sort_by(|a, b| {
        a.audio_start_time
            .partial_cmp(&b.audio_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut output = String::from("WEBVTT\n\n");
    let mut skipped = 0usize;

    for t in &sorted {
        let start = match t.audio_start_time {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };
        let end = t
            .audio_end_time
            .or_else(|| t.duration.map(|d| start + d))
            .unwrap_or(start + 5.0);

        // Escape VTT markup, and collapse newlines: a blank line inside cue
        // text terminates the cue, corrupting/truncating it for parsers.
        let text = t
            .text
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('\r', "")
            .replace('\n', " ");

        let cue_body = match speaker_display_label(t) {
            Some(label) => {
                let voice = label
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                format!("<v {}>{}", voice, text)
            }
            None => text,
        };

        output.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_timestamp(start),
            format_timestamp(end),
            cue_body
        ));
    }

    (output, skipped)
}

pub fn sanitize_filename_stem(title: &str) -> String {
    let sanitized: String = title
        .chars()
        .map(|c| {
            if matches!(c, '\t' | '\n' | '\r' | '\u{000b}' | '\u{000c}') {
                ' '
            } else if matches!(
                c,
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'
            ) || ('\0'..='\u{1f}').contains(&c)
            {
                // Remove these entirely
                '\0'
            } else {
                c
            }
        })
        .filter(|c| *c != '\0')
        .collect();

    let collapsed: Vec<&str> = sanitized.split_whitespace().collect();
    let collapsed = collapsed.join(" ");
    let collapsed = collapsed.trim().trim_matches('.').to_string();

    if collapsed.is_empty() {
        return "meeting".to_string();
    }

    if collapsed.len() > 80 {
        collapsed[..80].to_string()
    } else {
        collapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start: Option<f64>, end: Option<f64>, dur: Option<f64>) -> MeetingTranscript {
        MeetingTranscript {
            id: "t".to_string(),
            text: text.to_string(),
            timestamp: "00:00:00".to_string(),
            audio_start_time: start,
            audio_end_time: end,
            duration: dur,
            speaker_profile_id: None,
            speaker_label: None,
            speaker_confidence: None,
            speaker_confirmed: None,
        }
    }

    #[test]
    fn test_txt_happy_path() {
        let segs = vec![
            seg("Hello world", Some(10.0), None, None),
            seg("Second line", Some(15.0), None, None),
        ];
        let result = format_txt(&segs);
        assert!(result.contains("[00:00:10.00] Hello world"));
        assert!(result.contains("[00:00:15.00] Second line"));
    }

    #[test]
    fn test_txt_null_timestamp() {
        let segs = vec![seg("No timestamp", None, None, None)];
        let result = format_txt(&segs);
        assert_eq!(result, "No timestamp\n");
    }

    #[test]
    fn test_txt_newline_collapse() {
        let segs = vec![seg("Line1\nLine2", Some(1.0), None, None)];
        let result = format_txt(&segs);
        assert_eq!(result, "[00:00:01.00] Line1 Line2\n");
    }

    #[test]
    fn test_txt_ordering() {
        let segs = vec![
            seg("Second", Some(5.0), None, None),
            seg("First", Some(1.0), None, None),
        ];
        let result = format_txt(&segs);
        assert!(result.starts_with("[00:00:01.00] First"));
    }

    #[test]
    fn test_txt_hour_rollover() {
        let segs = vec![seg("Late", Some(3720.0), None, None)];
        let result = format_txt(&segs);
        assert!(result.contains("[01:02:00.00] Late"));
    }

    #[test]
    fn test_txt_empty() {
        assert_eq!(format_txt(&[]), "");
    }

    #[test]
    fn test_vtt_happy_path() {
        let segs = vec![seg("Hello", Some(1.0), Some(3.0), None)];
        let (result, skipped) = format_vtt(&segs);
        assert_eq!(skipped, 0);
        assert!(result.starts_with("WEBVTT"));
        assert!(result.contains("00:00:01.00 --> 00:00:03.00"));
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_vtt_duration_fallback() {
        let segs = vec![seg("Dur", Some(1.0), None, Some(2.5))];
        let (result, _) = format_vtt(&segs);
        assert!(result.contains("00:00:01.00 --> 00:00:03.50"));
    }

    #[test]
    fn test_vtt_default_fallback() {
        let segs = vec![seg("Only start", Some(1.0), None, None)];
        let (result, _) = format_vtt(&segs);
        assert!(result.contains("00:00:01.00 --> 00:00:06.00"));
    }

    #[test]
    fn test_vtt_null_start_skipped() {
        let segs = vec![
            seg("Skip", None, None, None),
            seg("Keep", Some(5.0), Some(7.0), None),
        ];
        let (result, skipped) = format_vtt(&segs);
        assert_eq!(skipped, 1);
        assert!(!result.contains("Skip"));
        assert!(result.contains("Keep"));
    }

    #[test]
    fn test_vtt_escaping() {
        let segs = vec![seg("A & B < C > D", Some(1.0), Some(2.0), None)];
        let (result, _) = format_vtt(&segs);
        assert!(result.contains("A &amp; B &lt; C &gt; D"));
    }

    #[test]
    fn test_vtt_overlapping_cues() {
        let segs = vec![
            seg("First", Some(1.0), Some(3.0), None),
            seg("Second", Some(2.0), Some(4.0), None),
        ];
        let (result, _) = format_vtt(&segs);
        assert!(result.contains("First"));
        assert!(result.contains("Second"));
    }

    #[test]
    fn test_vtt_empty() {
        let (result, skipped) = format_vtt(&[]);
        assert_eq!(skipped, 0);
        assert_eq!(result, "WEBVTT\n\n");
    }

    #[test]
    fn test_timestamp_minute_boundary_never_emits_sixty() {
        // Regression: values just under a minute boundary must carry, not render ":60.00".
        assert_eq!(format_timestamp(59.996), "00:01:00.00");
        assert_eq!(format_timestamp(119.998), "00:02:00.00");
        assert_eq!(format_timestamp(3599.999), "01:00:00.00");
        // Normal values unaffected.
        assert_eq!(format_timestamp(10.0), "00:00:10.00");
        assert_eq!(format_timestamp(59.99), "00:00:59.99");
        assert_eq!(format_timestamp(0.0), "00:00:00.00");
    }

    #[test]
    fn test_txt_speaker_label_prefixed() {
        let mut s = seg("Hello world", Some(1.0), None, None);
        s.speaker_label = Some("Alice".to_string());
        let result = format_txt(&[s]);
        assert_eq!(result, "[00:00:01.00] Alice: Hello world\n");
    }

    #[test]
    fn test_txt_blank_speaker_label_ignored() {
        let mut s = seg("Hi", Some(1.0), None, None);
        s.speaker_label = Some("   ".to_string());
        let result = format_txt(&[s]);
        assert_eq!(result, "[00:00:01.00] Hi\n");
    }

    #[test]
    fn test_vtt_speaker_voice_tag() {
        let mut s = seg("Hello", Some(1.0), Some(3.0), None);
        s.speaker_label = Some("Alice".to_string());
        let (result, _) = format_vtt(&[s]);
        assert!(result.contains("00:00:01.00 --> 00:00:03.00\n<v Alice>Hello"));
    }

    #[test]
    fn test_vtt_cue_newlines_collapsed() {
        // A blank line inside cue text would otherwise terminate the cue.
        let s = seg("Line1\n\nLine2", Some(1.0), Some(2.0), None);
        let (result, _) = format_vtt(&[s]);
        assert!(!result.contains("Line1\n\nLine2"));
        assert!(result.contains("Line1  Line2") || result.contains("Line1 Line2"));
    }

    #[test]
    fn test_sanitize_banned_chars() {
        let result = sanitize_filename_stem("a/b:c*d?e");
        assert!(!result.contains('/'));
        assert!(!result.contains(':'));
        assert!(!result.contains('*'));
    }

    #[test]
    fn test_sanitize_whitespace_collapse() {
        let result = sanitize_filename_stem("a   b\t\tc");
        assert_eq!(result, "a b c");
    }

    #[test]
    fn test_sanitize_empty_fallback() {
        let result = sanitize_filename_stem("");
        assert_eq!(result, "meeting");
    }

    #[test]
    fn test_sanitize_whitespace_only() {
        let result = sanitize_filename_stem("   ");
        assert_eq!(result, "meeting");
    }

    #[test]
    fn test_sanitize_truncation() {
        let long = "a".repeat(100);
        let result = sanitize_filename_stem(&long);
        assert_eq!(result.len(), 80);
    }
}
