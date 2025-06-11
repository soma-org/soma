use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    record::{HalfPrecisionSettings, Recorder},
    tensor::{backend::Backend, Tensor},
};
use bytes::Bytes;

use crate::modules::recorder::BcsRecorder;

use super::{BYTE_EMBEDDING_DIM, VOCAB_SIZE};

#[derive(Module, Debug)]
pub struct PredictorV1<B: Backend> {
    layer_norm: LayerNorm<B>,
    to_logits: Linear<B>,
}

impl<B: Backend> PredictorV1<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            layer_norm: LayerNormConfig::new(BYTE_EMBEDDING_DIM).init(device),
            to_logits: LinearConfig::new(BYTE_EMBEDDING_DIM, VOCAB_SIZE).init(device),
        }
    }

    pub fn forward(
        &self,
        byte_representations: Tensor<B, 1>, // takes the byte representations
    ) -> Tensor<B, 1> {
        let [byte_embedding_dim] = byte_representations.dims();
        assert_eq!(
            byte_embedding_dim, BYTE_EMBEDDING_DIM,
            "Input embedding dimensions must be {BYTE_EMBEDDING_DIM}"
        );
        let x = self.layer_norm.forward(byte_representations);
        self.to_logits.forward(x)
    }

    pub(crate) fn from_bytes(device: &B::Device, bytes: Bytes) -> Self {
        let recorder = BcsRecorder::<HalfPrecisionSettings>::new();
        let record = recorder.load(bytes.into(), device).unwrap();
        Self::init(device).load_record(record)
        //TODO: must perform architecture validation
    }
    pub(crate) fn to_bytes(self) -> Bytes {
        let recorder = BcsRecorder::<HalfPrecisionSettings>::new();
        Bytes::from(recorder.record(self.into_record(), ()).unwrap())
    }
}
