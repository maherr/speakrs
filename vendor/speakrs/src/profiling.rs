use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ort::session::Session;
use ort::session::builder::SessionBuilder;
use serde_json::json;

static ENABLED: OnceLock<bool> = OnceLock::new();
static ORT_PROFILE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();
static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);

fn enabled_from_env() -> bool {
    matches!(
        std::env::var("SPEAKRS_PROFILE_JSONL").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("on")
    )
}

pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(enabled_from_env)
}

pub(crate) fn ort_profile_dir() -> Option<&'static PathBuf> {
    ORT_PROFILE_DIR
        .get_or_init(|| std::env::var_os("SPEAKRS_ORT_PROFILE_DIR").map(PathBuf::from))
        .as_ref()
}

pub(crate) fn next_call_id() -> u64 {
    NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn now_ns() -> u64 {
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub(crate) fn emit(value: serde_json::Value) {
    if enabled() {
        eprintln!("{value}");
    }
}

pub(crate) fn emit_stage(
    stage: &str,
    file_id: &str,
    start_ns: u64,
    end_ns: u64,
    extra: serde_json::Value,
) {
    let mut event = serde_json::Map::new();
    event.insert("event".to_string(), json!("stage"));
    event.insert("stage".to_string(), json!(stage));
    event.insert("file_id".to_string(), json!(file_id));
    event.insert("start_ns".to_string(), json!(start_ns));
    event.insert("end_ns".to_string(), json!(end_ns));
    event.insert(
        "duration_us".to_string(),
        json!((end_ns.saturating_sub(start_ns)) as f64 / 1000.0),
    );
    if let Some(extra_obj) = extra.as_object() {
        for (key, value) in extra_obj {
            event.insert(key.clone(), value.clone());
        }
    }
    emit(serde_json::Value::Object(event));
}

pub(crate) fn emit_embedding_call(
    path: &str,
    call_id: u64,
    batch_len: usize,
    chunk_idx: Option<usize>,
    speaker_idx: Option<usize>,
    dispatch_start_ns: u64,
    dispatch_return_ns: u64,
    readback_done_ns: u64,
) {
    emit(json!({
        "event": "embedding_call",
        "path": path,
        "call_id": call_id,
        "batch_len": batch_len,
        "chunk_idx": chunk_idx,
        "speaker_idx": speaker_idx,
        "t_dispatch_start_ns": dispatch_start_ns,
        "t_dispatch_return_ns": dispatch_return_ns,
        "t_readback_done_ns": readback_done_ns,
        "dispatch_window_us": (dispatch_return_ns.saturating_sub(dispatch_start_ns)) as f64 / 1000.0,
        "readback_us": (readback_done_ns.saturating_sub(dispatch_return_ns)) as f64 / 1000.0,
        "total_us": (readback_done_ns.saturating_sub(dispatch_start_ns)) as f64 / 1000.0
    }));
}

fn sanitize_component(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(crate) fn enable_ort_profiling(
    builder: SessionBuilder,
    model_path: &Path,
) -> Result<SessionBuilder, ort::Error> {
    let Some(dir) = ort_profile_dir() else {
        return Ok(builder);
    };
    let stem = model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_component)
        .unwrap_or_else(|| "session".to_string());
    let file_name = format!("profile_{stem}.json");
    Ok(builder.with_profiling(dir.join(file_name))?)
}

pub(crate) fn finish_ort_profile(session: &mut Session, label: &str) {
    if ort_profile_dir().is_none() {
        return;
    }
    match session.end_profiling() {
        Ok(path) => emit(json!({
            "event": "ort_profile_file",
            "label": label,
            "path": path
        })),
        Err(error) => emit(json!({
            "event": "ort_profile_file_error",
            "label": label,
            "error": error.to_string()
        })),
    }
}
