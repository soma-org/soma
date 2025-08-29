pub mod recorder;
pub mod rotary;
pub mod v1;

use burn::tensor::{backend::Backend, Int, Tensor};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use v1::decoder::DecoderV1;
use v1::predictor::PredictorV1;

#[enum_dispatch]
pub trait SerializedProbeAPI<B: Backend> {
    fn to_probe(self, device: &B::Device) -> Probe<B>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SerializedProbeV1 {
    decoder: Bytes,
    predictor: Bytes,
}

impl<B: Backend> SerializedProbeAPI<B> for SerializedProbeV1 {
    fn to_probe(self, device: &B::Device) -> Probe<B> {
        Probe::V1(ProbeV1 {
            decoder: DecoderV1::from_bytes(device, self.decoder),
            predictor: PredictorV1::from_bytes(device, self.predictor),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[enum_dispatch(SerializedProbeAPI)]
pub enum SerializedProbe {
    V1(SerializedProbeV1),
}

#[enum_dispatch]
pub trait ProbeAPI<B: Backend> {
    fn call_decoder(
        &self,
        byte_ids: Tensor<B, 2, Int>, // list of byte ids
        patch_embeds: Tensor<B, 3>,  // single embedding
                                     // returns the byte representations to be passed into the predictor
    ) -> Tensor<B, 3>;
    fn call_predictor(
        &self,
        byte_embeds: Tensor<B, 2>, // takes the byte representations
                                   // returns the predicted logits of vocab size
    ) -> Tensor<B, 2>;
    fn serialize(self) -> SerializedProbe;
}

#[derive(Clone, Debug)]
pub(crate) struct ProbeV1<B: Backend> {
    decoder: DecoderV1<B>,
    predictor: PredictorV1<B>,
}

impl<B: Backend> ProbeV1<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            decoder: DecoderV1::init(device),
            predictor: PredictorV1::init(device),
        }
    }
}

impl<B: Backend> ProbeAPI<B> for ProbeV1<B> {
    fn call_decoder(
        &self,
        byte_ids: Tensor<B, 2, Int>, // list of byte ids
        patch_embedding: Tensor<B, 3>, // single embedding
                                     // returns the byte representations to be passed into the predictor
    ) -> Tensor<B, 3> {
        self.decoder.forward(byte_ids, patch_embedding)
    }
    fn call_predictor(
        &self,
        byte_representation: Tensor<B, 2>, // takes the byte representations
                                           // returns the predicted logits of vocab size
    ) -> Tensor<B, 2> {
        self.predictor.forward(byte_representation)
    }
    fn serialize(self) -> SerializedProbe {
        SerializedProbe::V1(SerializedProbeV1 {
            decoder: self.decoder.to_bytes(),
            predictor: self.predictor.to_bytes(),
        })
    }
}

#[derive(Clone, Debug)]
#[enum_dispatch(ProbeAPI<B>)]
pub enum Probe<B: Backend> {
    V1(ProbeV1<B>),
}
