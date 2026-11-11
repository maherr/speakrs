use serde::Serialize;
use speakrs::pipeline::{FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
use speakrs::{ExecutionMode, ModelManager, OwnedDiarizationPipeline};
use std::env;
use std::process;
use std::time::Instant;
#[cfg(unix)]
use std::path::PathBuf;

#[derive(Serialize)]
struct Segment {
    start: f64,
    end: f64,
    speaker: String,
}

#[derive(Serialize)]
struct DiarizationResult {
    segments: Vec<Segment>,
    speakers: usize,
    duration_secs: f64,
    processing_secs: f64,
    execution_mode: &'static str,
}

#[derive(Clone, Copy)]
enum CliMode {
    Cpu,
    MiGraphX,
    Cuda,
    CudaFast,
}

impl CliMode {
    fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Cpu => ExecutionMode::Cpu,
            Self::MiGraphX => ExecutionMode::MiGraphX,
            Self::Cuda => ExecutionMode::Cuda,
            Self::CudaFast => ExecutionMode::CudaFast,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::MiGraphX => "migraphx",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
        }
    }
}

#[cfg(unix)]
fn preload_migraphx_tf_stub(
    cli_mode: CliMode,
) -> Result<Option<libloading::os::unix::Library>, String> {
    if !matches!(cli_mode, CliMode::MiGraphX) {
        return Ok(None);
    }

    let path = env::var("MIGRAPHX_TF_STUB_PATH").unwrap_or_else(|_| {
        let data = env::var("XDG_DATA_HOME")
            .or_else(|_| env::var("HOME").map(|h| format!("{h}/.local/share")))
            .unwrap_or_default();
        format!("{data}/gpu-diarization-build/migraphx-install/lib/libmigraphx_tf_stub.so")
    });
    let path = PathBuf::from(path);
    if !path.exists() {
        return Ok(None);
    }

    // RTLD_GLOBAL makes the stub's C++ symbol available to libmigraphx_c.so when ORT loads the EP.
    unsafe {
        libloading::os::unix::Library::open(
            Some(&path),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }
    .map(Some)
    .map_err(|err| format!("Error preloading MIGraphX TF stub from {}: {err}", path.display()))
}

#[cfg(not(unix))]
fn preload_migraphx_tf_stub(_cli_mode: CliMode) -> Result<Option<()>, String> {
    Ok(None)
}

fn parse_args(args: &[String]) -> Result<(CliMode, &str), String> {
    let mut mode = CliMode::Cpu;
    let mut wav_path: Option<&str> = None;
    let mut idx = 1;

    while idx < args.len() {
        match args[idx].as_str() {
            "--cpu" => mode = CliMode::Cpu,
            "--gpu" | "--migraphx" => mode = CliMode::MiGraphX,
            "--cuda" => mode = CliMode::Cuda,
            "--cuda-fast" => mode = CliMode::CudaFast,
            "--mode" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("Missing value after --mode".to_string());
                };
                mode = match value.as_str() {
                    "cpu" => CliMode::Cpu,
                    "gpu" | "migraphx" => CliMode::MiGraphX,
                    "cuda" => CliMode::Cuda,
                    "cuda-fast" => CliMode::CudaFast,
                    other => {
                        return Err(format!(
                            "Unsupported mode `{other}`. Use `cpu`, `migraphx`, `cuda`, or `cuda-fast`."
                        ));
                    }
                };
            }
            "--help" | "-h" => {
                return Err(String::new());
            }
            value if value.starts_with('-') => {
                return Err(format!("Unknown flag `{value}`"));
            }
            value => {
                if wav_path.is_some() {
                    return Err("Expected a single WAV file path".to_string());
                }
                wav_path = Some(value);
            }
        }
        idx += 1;
    }

    wav_path
        .map(|path| (mode, path))
        .ok_or_else(|| "Missing WAV file path".to_string())
}

/// The library's file list for the CUDA modes leaves out one model its loader
/// opens, so a first CUDA run on a clean cache could not start.
const CUDA_EXTRA_MODEL_FILES: &[&str] = &["wespeaker-voxceleb-resnet34-tail.onnx"];

/// The CUDA modes download more model files than the others, and the library
/// picks its code path from the files it finds beside the base models. Their
/// own cache keeps a CPU or MIGraphX run on the files it has always had.
fn cuda_model_cache_dir() -> std::path::PathBuf {
    let base = env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"));
    base.join("speakrs-cli").join("cuda-models")
}

fn load_pipeline(cli_mode: CliMode, mode: ExecutionMode) -> Result<OwnedDiarizationPipeline, String> {
    if !matches!(cli_mode, CliMode::Cuda | CliMode::CudaFast) {
        return OwnedDiarizationPipeline::from_pretrained(mode).map_err(|e| e.to_string());
    }
    let manager =
        ModelManager::with_cache_dir(cuda_model_cache_dir()).map_err(|e| e.to_string())?;
    let models_dir = manager.ensure(mode).map_err(|e| e.to_string())?;
    for file in CUDA_EXTRA_MODEL_FILES {
        manager.get(file).map_err(|e| e.to_string())?;
    }
    OwnedDiarizationPipeline::from_dir(models_dir, mode).map_err(|e| e.to_string())
}

fn print_usage() {
    eprintln!(
        "Usage: speakrs-cli [--cpu|--gpu|--cuda|--cuda-fast|--mode cpu|migraphx|cuda|cuda-fast] <wav_file>"
    );
    eprintln!("  --gpu is MIGraphX (AMD). --cuda and --cuda-fast are NVIDIA.");
    eprintln!("  Input must be 16kHz mono WAV.");
    eprintln!("  Outputs JSON with speaker segments to stdout.");
}

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(format!("Expected mono audio, got {} channels", spec.channels).into());
    }
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz sample rate, got {}", spec.sample_rate).into());
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect(),
    };

    Ok(samples)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let (cli_mode, wav_path) = match parse_args(&args) {
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

    // Load audio
    let samples = match load_wav(wav_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading audio: {}", e);
            process::exit(1);
        }
    };

    let duration_secs = samples.len() as f64 / 16000.0;

    let mode = cli_mode.execution_mode();
    eprintln!("Using {} mode", cli_mode.label());

    let _migraphx_tf_stub = match preload_migraphx_tf_stub(cli_mode) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    };

    // Initialize pipeline (downloads models on first run)
    let mut pipeline = match load_pipeline(cli_mode, mode) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error initializing pipeline: {}", e);
            process::exit(1);
        }
    };

    // Run diarization
    let start = Instant::now();
    let result = match pipeline.run(&samples) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error running diarization: {}", e);
            process::exit(1);
        }
    };
    let processing_secs = start.elapsed().as_secs_f64();

    // Extract segments
    let raw_segments =
        result
            .discrete_diarization
            .to_segments(FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS);

    let mut segments: Vec<Segment> = Vec::new();
    let mut speakers = std::collections::HashSet::new();

    for seg in raw_segments {
        speakers.insert(seg.speaker.clone());

        // Merge consecutive segments from same speaker
        if let Some(last) = segments.last_mut() {
            if last.speaker == seg.speaker && (seg.start - last.end).abs() < 0.5 {
                last.end = seg.end;
                continue;
            }
        }

        segments.push(Segment {
            start: seg.start,
            end: seg.end,
            speaker: seg.speaker,
        });
    }

    let output = DiarizationResult {
        segments,
        speakers: speakers.len(),
        duration_secs,
        processing_secs,
        execution_mode: cli_mode.label(),
    };

    println!("{}", serde_json::to_string(&output).unwrap());
}
