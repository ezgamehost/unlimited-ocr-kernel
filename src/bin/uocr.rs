use clap::Parser;
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};
use uocr::{
    cpu::{Backend, CpuContext},
    generation::GenerationOptions,
    model::{Model, Request},
    preprocess::ImageMode,
};

#[derive(Parser)]
#[command(about = "Native Rust CPU inference for Unlimited-OCR")]
struct Args {
    #[arg(long, default_value = "artifacts/upstream")]
    model: PathBuf,
    #[arg(long, required_unless_present = "jsonl")]
    image: Option<PathBuf>,
    #[arg(long,default_value="gundam",value_parser=["base","gundam"])]
    mode: String,
    #[arg(long, default_value = "<image>document parsing.")]
    prompt: String,
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    #[arg(long, default_value_t = 35)]
    ngram_size: usize,
    #[arg(long, default_value_t = 128)]
    ngram_window: usize,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long,default_value="auto",value_parser=["auto","scalar","accelerate"])]
    backend: String,
    #[arg(long, default_value_t = 1024)]
    scratch_mib: usize,
    /// Keep weights resident and read independent request objects from stdin.
    #[arg(long)]
    jsonl: bool,
    /// Print one complete result JSON object instead of just OCR text.
    #[arg(long)]
    json: bool,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let backend = match args.backend.as_str() {
        "scalar" => Backend::Scalar,
        "accelerate" => Backend::Accelerate,
        _ => Backend::Auto,
    };
    let scratch = args
        .scratch_mib
        .checked_mul(1 << 20)
        .ok_or("scratch limit overflow")?;
    let mut cpu = CpuContext::new(backend, args.threads, scratch)?;
    let stamp = Instant::now();
    let model = Model::load(&args.model)?;
    let load = stamp.elapsed().as_secs_f64();
    eprintln!(
        "{}",
        serde_json::json!({"event":"loaded","load_seconds":load,"backend":args.backend,"threads":args.threads})
    );
    if args.jsonl {
        for line in io::stdin().lock().lines() {
            let line = line?;
            let response = match serde_json::from_str::<Request>(&line) {
                Ok(request) => match model.infer(&request, &mut cpu) {
                    Ok(result) => serde_json::json!({"status":"ok","result":result}),
                    Err(error) => serde_json::json!({"status":"error","error":error.to_string()}),
                },
                Err(error) => serde_json::json!({"status":"error","error":error.to_string()}),
            };
            println!("{response}");
            io::stdout().flush()?;
        }
    } else {
        let request = Request {
            image: args.image.ok_or("missing image")?,
            mode: if args.mode == "base" {
                ImageMode::Base
            } else {
                ImageMode::Gundam
            },
            prompt: args.prompt,
            generation: GenerationOptions {
                max_tokens: args.max_tokens,
                ngram_size: args.ngram_size,
                ngram_window: args.ngram_window,
            },
        };
        let result = model.infer(&request, &mut cpu)?;
        if args.json {
            println!("{}", serde_json::to_string(&result)?);
        } else {
            println!("{}", result.text);
            eprintln!("{}", serde_json::to_string(&result.timings)?);
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("uocr: {error}");
        std::process::exit(1);
    }
}
