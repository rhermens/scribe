use std::ops::Range;

pub mod audio;
pub mod download;
pub mod embedding;
pub mod model;
pub mod topics;

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
