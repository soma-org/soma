use serde::{Deserialize, Serialize};

use super::{
    authority_committee::Epoch,
    encoder_committee::{Encoder, EncoderCommittee, EncoderIndex},
};

/// Modality is an enum that contains a modality marker. This is used to ensure type
/// safety while allowing an encoder to handle and interact with multiple identities.
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone)]
pub(crate) struct ModalityCommittee {
    epoch: Epoch,
    text_committee: Option<EncoderCommittee<TextMarker>>,
    image_committee: Option<EncoderCommittee<ImageMarker>>,
    audio_committee: Option<EncoderCommittee<AudioMarker>>,
    video_committee: Option<EncoderCommittee<VideoMarker>>,
}

impl ModalityCommittee {
    pub(crate) fn new(
        epoch: Epoch,
        text_committee: Option<EncoderCommittee<TextMarker>>,
        image_committee: Option<EncoderCommittee<ImageMarker>>,
        audio_committee: Option<EncoderCommittee<AudioMarker>>,
        video_committee: Option<EncoderCommittee<VideoMarker>>,
    ) -> Self {
        Self {
            epoch,
            text_committee,
            image_committee,
            audio_committee,
            video_committee,
        }
    }

    pub(crate) fn get_committee<M: ModalityMarker>(&self) -> Option<&EncoderCommittee<M>>
    where
        Self: GetCommittee<M>,
    {
        <Self as GetCommittee<M>>::get_committee(self).as_ref()
    }

    pub(crate) fn get_encoder<M: ModalityMarker>(
        &self,
        index: EncoderIndex<M>,
    ) -> Option<&Encoder<M>>
    where
        Self: GetCommittee<M>,
    {
        self.get_committee::<M>()
            .and_then(|committee| Some(committee.encoder(index)))
    }
}

/// Define a trait for modality markers
pub(crate) trait ModalityMarker {}

/// Marker for Text Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TextMarker;
impl ModalityMarker for TextMarker {}

/// Marker for Image Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ImageMarker;
impl ModalityMarker for ImageMarker {}

/// Marker for Audio Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct AudioMarker;
impl ModalityMarker for AudioMarker {}

/// Marker for Video Modality
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct VideoMarker;
impl ModalityMarker for VideoMarker {}

pub(crate) trait GetCommittee<M: ModalityMarker> {
    fn get_committee(&self) -> &Option<EncoderCommittee<M>>;
}

impl GetCommittee<TextMarker> for ModalityCommittee {
    fn get_committee(&self) -> &Option<EncoderCommittee<TextMarker>> {
        &self.text_committee
    }
}

impl GetCommittee<ImageMarker> for ModalityCommittee {
    fn get_committee(&self) -> &Option<EncoderCommittee<ImageMarker>> {
        &self.image_committee
    }
}

impl GetCommittee<AudioMarker> for ModalityCommittee {
    fn get_committee(&self) -> &Option<EncoderCommittee<AudioMarker>> {
        &self.audio_committee
    }
}

impl GetCommittee<VideoMarker> for ModalityCommittee {
    fn get_committee(&self) -> &Option<EncoderCommittee<VideoMarker>> {
        &self.video_committee
    }
}

#[derive(Clone)]
pub(crate) struct OwnModalityIndices {
    pub text_index: Option<EncoderIndex<TextMarker>>,
    pub image_index: Option<EncoderIndex<ImageMarker>>,
    pub audio_index: Option<EncoderIndex<AudioMarker>>,
    pub video_index: Option<EncoderIndex<VideoMarker>>,
}
