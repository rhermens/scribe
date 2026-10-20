use std::{error::Error, fs::File, io::Write, num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use scribe::{Segment, load_audio, sentence_windows, topic_paragraphs};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    audio: PathBuf,

    #[arg(long)]
    lang: String,

    /// Split at context similarity valleys below this value; higher values split more often.
    #[arg(long, default_value_t = 0.6, value_parser = parse_topic_threshold, allow_hyphen_values = true)]
    topic_threshold: f32,

    /// Minimum words on either side of a subject boundary; never forces a split.
    #[arg(long, default_value = "80")]
    min_paragraph_words: NonZeroUsize,
}

fn parse_topic_threshold(value: &str) -> Result<f32, String> {
    let threshold: f32 = value
        .parse()
        .map_err(|_| "expected a number between -1 and 1")?;
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err("topic threshold must be between -1 and 1".to_owned());
    }
    Ok(threshold)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // load a context and model
    let ctx = WhisperContext::new_with_params(args.model, WhisperContextParameters::default())
        .expect("failed to load model");

    // create a params object
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    params.set_language(Some(&args.lang));

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

    let sentences = sentence_windows(&segments);
    // Empty and single-sentence transcripts need neither a model download nor inference.
    let paragraphs = if sentences.len() < 2 {
        sentences
    } else {
        let mut model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::ParaphraseMLMiniLML12V2)
                .with_show_download_progress(true),
        )?;
        let texts: Vec<&str> = sentences
            .iter()
            .map(|sentence| sentence.text.as_str())
            .collect();
        let embeddings = model.embed(texts, None)?;
        topic_paragraphs(
            &sentences,
            &embeddings,
            args.topic_threshold,
            args.min_paragraph_words.get(),
        )?
    };

    // Finish segmentation before replacing the output file, so inference errors keep it intact.
    let mut file = File::create("transcript.md")?;
    for paragraph in &paragraphs {
        let first = &segments[paragraph.segments.start];
        let last = &segments[paragraph.segments.end - 1];
        let markdown = format!(
            "## [{} - {} cs] #{}-#{}\n\n{}\n",
            first.start, last.end, first.id, last.id, paragraph.text,
        );
        println!("{markdown}");
        writeln!(file, "{markdown}")?;
    }
    eprintln!("{} topic paragraphs", paragraphs.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topic_options_are_validated_before_transcription() {
        let args = Args::try_parse_from([
            "scribe",
            "--model",
            "model.bin",
            "--audio",
            "audio.wav",
            "--lang",
            "nl",
        ])
        .unwrap();
        assert_eq!(args.topic_threshold, 0.6);
        assert_eq!(args.min_paragraph_words.get(), 80);

        for value in ["NaN", "inf", "1.1", "bad"] {
            assert!(
                Args::try_parse_from([
                    "scribe",
                    "--model",
                    "model.bin",
                    "--audio",
                    "audio.wav",
                    "--lang",
                    "nl",
                    "--topic-threshold",
                    value,
                ])
                .is_err()
            );
        }
        assert!(
            Args::try_parse_from([
                "scribe",
                "--model",
                "model.bin",
                "--audio",
                "audio.wav",
                "--lang",
                "nl",
                "--min-paragraph-words",
                "0",
            ])
            .is_err()
        );
        let args = Args::try_parse_from([
            "scribe",
            "--model",
            "model.bin",
            "--audio",
            "audio.wav",
            "--lang",
            "nl",
            "--topic-threshold",
            "-0.5",
            "--min-paragraph-words",
            "20",
        ])
        .unwrap();
        assert_eq!(args.topic_threshold, -0.5);
        assert_eq!(args.min_paragraph_words.get(), 20);
        assert_eq!(parse_topic_threshold("-1").unwrap(), -1.0);
        assert_eq!(parse_topic_threshold("1").unwrap(), 1.0);
    }
}
