use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use ort::ep;
use ort::session::Session;

#[cfg(unix)]
fn preload_migraphx_tf_stub() -> Result<Option<libloading::os::unix::Library>, String> {
    let path = env::var("MIGRAPHX_TF_STUB_PATH").unwrap_or_else(|_| {
        "/home/maher/.local/share/gpu-diarization-build/migraphx-install/lib/libmigraphx_tf_stub.so"
            .to_string()
    });
    let path = PathBuf::from(path);
    if !path.exists() {
        return Ok(None);
    }

    unsafe {
        libloading::os::unix::Library::open(
            Some(&path),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }
    .map(Some)
    .map_err(|err| format!("failed to preload MIGraphX TF stub {}: {err}", path.display()))
}

#[cfg(not(unix))]
fn preload_migraphx_tf_stub() -> Result<Option<()>, String> {
    Ok(None)
}

fn default_segmentation_model() -> PathBuf {
    PathBuf::from(
        "/home/maher/.cache/huggingface/hub/models--avencera--speakrs-models/snapshots/5d24ffee75f13fb061fa6d10944a64e2dc1d5e6f/segmentation-3.0.onnx",
    )
}

fn init_ort() -> Result<(), String> {
    let Some(path) = env::var_os("ORT_DYLIB_PATH") else {
        return Err("ORT_DYLIB_PATH must point to libonnxruntime.so".to_string());
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(format!("ORT_DYLIB_PATH does not exist: {}", path.display()));
    }

    ort::init_from(&path)
        .map(|builder| {
            builder.commit();
        })
        .map_err(|err| format!("failed to initialize ORT from {}: {err}", path.display()))
}

#[derive(Clone, Copy)]
enum Mode {
    Cpu,
    MiGraphX,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::MiGraphX => "migraphx",
        }
    }
}

fn parse_args(args: &[String]) -> Result<(Mode, PathBuf), String> {
    let mut mode = Mode::MiGraphX;
    let mut model_path: Option<PathBuf> = None;
    let mut idx = 1;

    while idx < args.len() {
        match args[idx].as_str() {
            "--cpu" => mode = Mode::Cpu,
            "--gpu" | "--migraphx" => mode = Mode::MiGraphX,
            "--model" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("missing value after --model".to_string());
                };
                model_path = Some(PathBuf::from(value));
            }
            "--help" | "-h" => return Err(String::new()),
            flag if flag.starts_with('-') => return Err(format!("unknown flag `{flag}`")),
            path => {
                if model_path.is_some() {
                    return Err("expected a single model path".to_string());
                }
                model_path = Some(PathBuf::from(path));
            }
        }
        idx += 1;
    }

    Ok((mode, model_path.unwrap_or_else(default_segmentation_model)))
}

fn print_usage() {
    eprintln!("Usage: ort-migraphx-repro [--cpu|--gpu] [--model path/to/model.onnx]");
    eprintln!("  Default model: speakrs segmentation-3.0.onnx");
}

fn maybe_apply_segmentation_override(
    builder: ort::session::builder::SessionBuilder,
    model_path: &Path,
) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
    let file_name = model_path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    if file_name.starts_with("segmentation-3.0") {
        Ok(builder.with_dimension_override("samples", 160_000)?)
    } else {
        Ok(builder)
    }
}

fn cpu_builder(model_path: &Path) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
    let builder = Session::builder()?
        .with_independent_thread_pool()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_memory_pattern(true)?;
    let builder = maybe_apply_segmentation_override(builder, model_path)?;
    Ok(builder.with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?)
}

fn migraphx_builder(
    model_path: &Path,
) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
    let builder = Session::builder()?
        .with_independent_thread_pool()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_memory_pattern(true)?;
    let builder = maybe_apply_segmentation_override(builder, model_path)?;
    Ok(builder.with_execution_providers([ep::MIGraphX::default()
            .with_device_id(0)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .build()
            .error_on_failure()])?)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (mode, model_path) = match parse_args(&args) {
        Ok(parsed) => parsed,
        Err(message) if message.is_empty() => {
            print_usage();
            process::exit(0);
        }
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            process::exit(1);
        }
    };

    if !Path::new(&model_path).exists() {
        eprintln!("model not found: {}", model_path.display());
        process::exit(1);
    }

    let _stub = match mode {
        Mode::Cpu => None,
        Mode::MiGraphX => match preload_migraphx_tf_stub() {
            Ok(handle) => handle,
            Err(err) => {
                eprintln!("{err}");
                process::exit(1);
            }
        },
    };

    if let Err(err) = init_ort() {
        eprintln!("{err}");
        process::exit(1);
    }

    eprintln!("mode={}", mode.as_str());
    eprintln!("model={}", model_path.display());

    let start = Instant::now();
    let session = match mode {
        Mode::Cpu => cpu_builder(&model_path),
        Mode::MiGraphX => migraphx_builder(&model_path),
    }
    .and_then(|mut builder| builder.commit_from_file(&model_path));

    match session {
        Ok(_) => {
            eprintln!(
                "session init succeeded in {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
        Err(err) => {
            eprintln!(
                "session init failed after {:.3}s: {err}",
                start.elapsed().as_secs_f64()
            );
            process::exit(2);
        }
    }
}
