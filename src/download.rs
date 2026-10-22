use std::{collections::HashMap, error::Error, path::PathBuf, sync::Mutex};

use hf_hub::{
    HFClient,
    progress::{DownloadEvent, ProgressEvent, ProgressHandler},
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

const EMBEDDING_REPO: (&str, &str) = ("Xenova", "paraphrase-multilingual-MiniLM-L12-v2");
const EMBEDDING_FILES: [&str; 5] = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

const WHISPER_REPO: (&str, &str) = ("ggerganov", "whisper.cpp");

const LAYA_REPO: (&str, &str) = ("convaiinnovations", "laya");
const LAYA_FILES: [&str; 5] = [
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "rl_agent_config.json",
];

/// Sums per-file progress so one bar can follow a multi-file snapshot download.
struct DownloadBar {
    bar: ProgressBar,
    files: Mutex<HashMap<String, u64>>,
}

impl DownloadBar {
    fn new(bars: &MultiProgress, message: &str) -> Result<Self, Box<dyn Error>> {
        let bar = bars.add(
            ProgressBar::no_length().with_style(ProgressStyle::with_template(
                "{msg} {wide_bar} {bytes}/{total_bytes} {bytes_per_sec} {eta}",
            )?),
        );
        bar.set_message(message.to_owned());
        Ok(Self {
            bar,
            files: Mutex::default(),
        })
    }
}

impl ProgressHandler for DownloadBar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => self.bar.set_length(*total_bytes),
            DownloadEvent::Progress { files } => {
                let mut completed = self.files.lock().unwrap();
                for file in files {
                    completed.insert(file.filename.clone(), file.bytes_completed);
                }
                self.bar.set_position(completed.values().sum());
            }
            // Cache hits report no bytes, so fill the bar on completion.
            DownloadEvent::Complete => self.bar.finish(),
            DownloadEvent::AggregateProgress { .. } => {}
        }
    }
}

pub async fn download_models(
    whisper_model: &str,
) -> Result<(PathBuf, PathBuf, PathBuf), Box<dyn std::error::Error>> {
    let hf_client = HFClient::new()?;

    let bars = MultiProgress::new();
    let whisper_repo = hf_client.model(WHISPER_REPO.0, WHISPER_REPO.1);
    let embedding_repo = hf_client.model(EMBEDDING_REPO.0, EMBEDDING_REPO.1);
    let laya_repo = hf_client.model(LAYA_REPO.0, LAYA_REPO.1);

    // Download both models in parallel before any local work starts.
    let (whisper_model_path, embedding_model_dir, laya_model_dir) = tokio::try_join!(
        whisper_repo
            .download_file()
            .filename(format!("{}.bin", whisper_model))
            .progress(DownloadBar::new(&bars, whisper_model)?)
            .send(),
        embedding_repo
            .snapshot_download()
            .allow_patterns(EMBEDDING_FILES.map(String::from).to_vec())
            .progress(DownloadBar::new(&bars, EMBEDDING_REPO.1)?)
            .send(),
        laya_repo
            .snapshot_download()
            .allow_patterns(EMBEDDING_FILES.map(String::from).to_vec())
            .progress(DownloadBar::new(&bars, EMBEDDING_REPO.1)?)
            .send(),
    )?;

    Ok((whisper_model_path, embedding_model_dir, laya_model_dir))
}
