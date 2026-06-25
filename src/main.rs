use serde::Serialize;
use speakrs::pipeline::{FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
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
}

impl CliMode {
    fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Cpu => ExecutionMode::Cpu,
            Self::MiGraphX => ExecutionMode::MiGraphX,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::MiGraphX => "migraphx",
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
            "--mode" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err("Missing value after --mode".to_string());
                };
                mode = match value.as_str() {
                    "cpu" => CliMode::Cpu,
                    "gpu" | "migraphx" => CliMode::MiGraphX,
                    other => {
                        return Err(format!(
                            "Unsupported mode `{other}`. Use `cpu` or `migraphx`."
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

fn print_usage() {
    eprintln!("Usage: speakrs-cli [--cpu|--gpu|--mode cpu|migraphx] <wav_file>");
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
    let mut pipeline = match OwnedDiarizationPipeline::from_pretrained(mode) {
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
