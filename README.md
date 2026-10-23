# Scribe

Transcribe a 16 kHz WAV file into topic-separated Markdown paragraphs:

```sh
nix develop --impure
cargo run --release -- --model models/ggml-medium.bin --audio recording.wav --lang nl
```

Use a multilingual Whisper model (without `.en`) for Dutch.

Scribe writes `transcript.md` and prints the same paragraphs to stdout. Each paragraph includes the original Whisper segment IDs and start/end timestamps in centiseconds.

## Topic segmentation

Whisper segments are grouped at sentence endings, or after roughly 60 words without punctuation. Groups keep whole segments to preserve timestamps. A segment containing several sentences stays intact.

FastEmbed embeds these groups on the CPU with `ParaphraseMLMiniLML12V2`, a multilingual model supporting Dutch. Scribe compares the average embeddings of eight sentence groups on each side of a possible boundary. Short transcripts use smaller contexts.

Splits require a local similarity valley below the threshold, with a drop of at least `0.05` relative to nearby scores. Nearby candidates compete: the lowest-similarity boundary wins. Brief questions and ordinary sentence changes no longer trigger a split by themselves.

```sh
cargo run --release -- --model models/ggml-medium.bin --audio recording.wav --lang nl \
  --topic-threshold 0.6 --min-paragraph-words 80
```

- `--topic-threshold`: cross-context cosine similarity cutoff, from -1 to 1. Higher values admit more boundaries; lower values group broader subjects. Defaults to `0.6`.
- `--min-paragraph-words`: minimum resulting paragraph length; defaults to `80`. This only suppresses nearby boundaries; reaching it never forces a split. A transcript shorter than this remains one paragraph.

Tune both settings against your recordings. Topic boundaries remain embedding-based heuristics, not guaranteed subject classifications. Sentence endings use punctuation heuristics rather than a language-aware parser.

The embedding model downloads on first use and is cached in `.fastembed_cache/`. Empty and single-group transcripts skip embedding inference.

## GPU transcription

GPU acceleration is opt-in through Cargo features. Plain `cargo build` and `nix build` use the CPU on every platform. FastEmbed remains on the CPU.

`nix develop --impure` automatically selects the development toolchain:

- Linux with an NVIDIA driver: CUDA. Detection checks `/run/opengl-driver/lib/libcuda.so` or `/proc/driver/nvidia/version`.
- macOS: Metal.
- Other systems: CPU.

The existing `.envrc` enables host detection for direnv. Pure `nix develop` cannot inspect host driver paths and selects CPU on Linux. Use `nix develop .#cpu`, `.#cuda`, or `.#metal` to override detection.

Shell selection supplies the toolchain only; it does not enable Cargo features.

Select `--features cuda` or `--features metal` to enable the backend for Whisper and Laya:

```sh
cargo build --release --features cuda
cargo build --release --features metal
```

Pass the same feature to `cargo run`, `cargo test`, and `cargo clippy` when using a GPU backend. Backend toolchain and hardware requirements still apply.

### CUDA (Linux)

Use `nix develop --impure` to detect CUDA, or `nix develop .#cuda` to select it explicitly. Pass `--features cuda` to Cargo. Use `nix build .#cuda` for a CUDA-enabled package.

```sh
nix develop --impure
cargo run --release --features cuda -- --model models/ggml-medium.bin --audio recording.wav --lang nl
```

The CUDA Nix shell supplies CUDA 13.2, cuBLAS, and a compatible host compiler. NVIDIA's CUDA packages require allowing unfree packages; the flake enables this locally. The build targets compute capability `89` for the RTX 4070 SUPER. Change both `CMAKE_CUDA_ARCHITECTURES` and `CUDA_COMPUTE_CAP` in `flake.nix` for a different GPU.

An installed NVIDIA driver is required at runtime. The shell exposes the NixOS driver libraries through `/run/opengl-driver/lib`. Re-enter the shell or run `direnv reload` after changing the flake. CUDA compilation can take substantially longer than Metal builds.

### Metal (macOS)

The default development shell selects Metal on macOS. Enable `--features metal` in Cargo, or use `nix build .#metal`. The Metal shaders are embedded in the binary.

## Checks

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

Unit tests use synthetic embeddings and do not download a model.
