use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use clap::Parser;
use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use laya::{Agent, Question};
use scribe::{
    Segment,
    audio::load_audio,
    download::download_models,
    topics::{sentence_windows, topic_paragraphs},
};
use serde_json::json;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
    WhisperState,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "transcript.md")]
    output: PathBuf,

    #[arg(long, default_value = "ggml-medium")]
    whisper_model: String,

    #[arg(long)]
    audio: PathBuf,

    #[arg(long, default_value = "nl")]
    lang: String,

    #[arg(long, default_value = "Which action is required?")]
    prompt: String,

    #[arg(long)]
    options: Vec<String>,

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

fn load_embedding_model(dir: &Path) -> Result<TextEmbedding, Box<dyn Error>> {
    let read = |file: &str| fs::read(dir.join(file));
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };
    // Match the pooling and quantization TextEmbedding::try_new would pick.
    let mut model = UserDefinedEmbeddingModel::new(read("onnx/model.onnx")?, tokenizer_files)
        .with_quantization(TextEmbedding::get_quantization_mode(
            &EmbeddingModel::ParaphraseMLMiniLML12V2,
        ));
    if let Some(pooling) =
        TextEmbedding::get_default_pooling_method(&EmbeddingModel::ParaphraseMLMiniLML12V2)
    {
        model = model.with_pooling(pooling);
    }
    Ok(TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined::new(),
    )?)
}

fn load_whisper_model(
    model_path: PathBuf,
    lang: &'_ str,
) -> Result<(WhisperState, FullParams<'_, '_>), WhisperError> {
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .expect("failed to load model");

    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    params.set_language(Some(lang));
    ctx.create_state().map(|x| (x, params))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let (whisper_model_path, embedding_model_dir, laya_model_dir) =
        download_models(&args.whisper_model).await?;

    let (mut whisper_state, params) = load_whisper_model(whisper_model_path, &args.lang)?;
    let audio_data = load_audio(&args.audio);

    whisper_state
        .full(params, &audio_data[..])
        .expect("failed to run model");

    let laya_agent = Agent::from_dir(laya_model_dir, Default::default())?;
    let laya_questions = vec![(
        "action".to_owned(),
        Question::choice(&args.prompt, &args.options),
    )];

    // copy segments out of whisper's state so later stages don't borrow it
    let segments: Vec<Segment> = whisper_state
        .as_iter()
        .map(|s| Segment {
            id: s.segment_index() as usize,
            start: s.start_timestamp(),
            end: s.end_timestamp(),
            text: s.to_string().trim().to_owned(),
        })
        .collect();

    let sentences = sentence_windows(&segments);
    // Empty and single-sentence transcripts need no embedding inference.
    let paragraphs = if sentences.len() < 2 {
        sentences
    } else {
        let mut model = load_embedding_model(&embedding_model_dir)?;
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
        )
    };

    let actions = paragraphs
        .iter()
        .map(|paragraph| {
            let response = laya_agent.system_one(&json!(paragraph.text), &laya_questions)?;
            let action = response.answers["action"]["choice"]
                .as_str()
                .ok_or("laya returned no action choice")?
                .to_owned();
            Ok(action)
        })
        .collect::<Result<Vec<String>, Box<dyn Error>>>()?;

    // Finish segmentation and classification before replacing the output file,
    // so inference errors keep it intact.
    let mut file = File::create(args.output)?;
    for (paragraph, action) in paragraphs.iter().zip(&actions) {
        let first = &segments[paragraph.segments.start];
        let last = &segments[paragraph.segments.end - 1];
        let markdown = format!(
            "## [{} - {} cs] #{}-#{} {} ({})\n\n{}\n",
            first.start, last.end, first.id, last.id, &args.prompt, action, paragraph.text,
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
