use std::{fs::File, io::Write, path::PathBuf};

use clap::Parser;
use scribe::{Segment, load_audio, windows};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    audio: PathBuf,
}

fn main() {
    let args = Args::parse();

    // load a context and model
    let ctx = WhisperContext::new_with_params(args.model, WhisperContextParameters::default())
        .expect("failed to load model");

    // create a params object
    let params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });

    let audio_data = load_audio(&args.audio);

    // now we can run the model
    let mut state = ctx.create_state().expect("failed to create state");
    state
        .full(params, &audio_data[..])
        .expect("failed to run model");

    // copy segments out of whisper's state so later stages don't borrow it
    let segments: Vec<Segment> = state
        .as_iter()
        .map(|s| Segment {
            id: s.segment_index() as usize,
            start: s.start_timestamp(),
            end: s.end_timestamp(),
            text: s.to_string().trim().to_owned(),
        })
        .collect();

    let mut file = File::create("transcript.md").expect("Failed to create file");
    for s in &segments {
        let segment_line = format!("[{} - {}] #{}: {}", s.start, s.end, s.id, s.text);
        println!("{}", segment_line);
        writeln!(file, "{}", segment_line).expect("Failed to write line");
    }

    // 45s windows every 15s
    let windows = windows(&segments, 4500, 1500);
    println!("{} windows", windows.len());
}
