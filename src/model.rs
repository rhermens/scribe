use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use laya::Agent;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
    WhisperState,
};

pub fn load_embedding_model(dir: &Path) -> Result<TextEmbedding, Box<dyn Error>> {
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

pub fn load_transcript_model(
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

pub fn load_classification_model(dir: PathBuf) -> Result<Agent, Box<dyn std::error::Error>> {
    Agent::from_dir(dir, Default::default()).map_err(|e| e.into())
}
