use serde::{Deserialize, Serialize};

use super::encoder_committee::{EncoderCommittee, EncoderIndex};

/// Modality is an enum that contains a modality marker. This is used to ensure type
/// safety while allowing an encoder to handle and interact with multiple identities.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Modality {
    /// Text Modality
    Text(TextMarker),
    /// Image modality
    Image(ImageMarker),
    /// Audio Modality
    Audio(AudioMarker),
    /// Video modality
    Video(VideoMarker),
}

impl Modality {
    /// returns a text marked modality
    pub(crate) const fn text() -> Self {
        Self::Text(TextMarker)
    }
    /// returns a image marked modality
    pub(crate) const fn image() -> Self {
        Self::Image(ImageMarker)
    }
    /// returns a audio marked modality
    pub(crate) const fn audio() -> Self {
        Self::Audio(AudioMarker)
    }
    /// returns a video marked modality
    pub(crate) const fn video() -> Self {
        Self::Video(VideoMarker)
    }
}

/// Modality committee wraps the marked encoder committees
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ModalityCommittee {
    /// Text Committee
    Text(EncoderCommittee<TextMarker>),
    /// Image Committee
    Image(EncoderCommittee<ImageMarker>),
    /// Audio Committee
    Audio(EncoderCommittee<AudioMarker>),
    /// Video Committee
    Video(EncoderCommittee<VideoMarker>),
}

/// Modality index wraps each marked encoder index
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ModalityIndex {
    /// Text Index
    Text(EncoderIndex<TextMarker>),
    /// Image Index
    Image(EncoderIndex<ImageMarker>),
    /// Audio Index
    Audio(EncoderIndex<AudioMarker>),
    /// Video Index
    Video(EncoderIndex<VideoMarker>),
}

/// Define a trait for modality markers
pub(crate) trait ModalityMarker {}

/// Marker for Text Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TextMarker;
impl ModalityMarker for TextMarker {}

/// Marker for Image Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ImageMarker;
impl ModalityMarker for ImageMarker {}

/// Marker for Audio Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AudioMarker;
impl ModalityMarker for AudioMarker {}

/// Marker for Video Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VideoMarker;
impl ModalityMarker for VideoMarker {}
