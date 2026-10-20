use std::{ops::Range, path::Path};

use hound::{SampleFormat, WavReader};

/// Reads a WAV file into 32-bit float, 16KHz, mono samples as expected by whisper.
pub fn load_audio(path: &Path) -> Vec<f32> {
    let mut reader = WavReader::open(path).expect("failed to open audio file");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16000, "audio must be sampled at 16KHz");

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .expect("failed to read audio samples"),
        SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()
                .expect("failed to read audio samples")
        }
    };

    // downmix to mono by averaging the channels of each frame
    let channels = spec.channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub struct Segment {
    pub id: usize,
    /// Centiseconds, as returned by whisper.
    pub start: i64,
    pub end: i64,
    pub text: String,
}

pub struct Window {
    /// Indices into the segment slice.
    pub segments: Range<usize>,
    pub text: String,
}

/// Groups segments into overlapping windows of `size` centiseconds, advancing by `stride`.
pub fn windows(segments: &[Segment], size: i64, stride: i64) -> Vec<Window> {
    let Some(last) = segments.last() else {
        return Vec::new();
    };

    let mut out: Vec<Window> = Vec::new();
    let mut lo = 0;
    let mut t = segments[0].start;
    while t < last.end {
        // skip segments that ended before this window starts
        while lo < segments.len() && segments[lo].end <= t {
            lo += 1;
        }
        let hi = lo + segments[lo..].partition_point(|s| s.start < t + size);

        // silence gaps can produce the same range twice
        if hi > lo && out.last().map(|w| &w.segments) != Some(&(lo..hi)) {
            let text = segments[lo..hi]
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            out.push(Window {
                segments: lo..hi,
                text,
            });
        }

        if hi == segments.len() {
            break;
        }
        t += stride;
    }
    out
}
