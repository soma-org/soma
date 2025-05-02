use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        activation::{relu, sigmoid},
        backend::Backend,
        Tensor,
    },
};
use bytes::Bytes;

#[derive(Module, Debug)]
pub struct ByteDecoderV1<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
}

impl<B: Backend> ByteDecoderV1<B> {
    pub(crate) fn init(device: &B::Device) -> Self {
        Self {
            // 1024 -> 2048
            layer1: LinearConfig::new(1024, 2048).init(device),
            // 2048 -> 4096
            layer2: LinearConfig::new(2048, 4096).init(device),
        }
    }

    pub(crate) fn from_bytes(device: &B::Device, bytes: Bytes) -> Self {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(bytes.into(), device).unwrap();
        Self::init(device).load_record(record)
        //TODO: should perform validation
    }

    pub(crate) fn forward(&self, embedding: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_batch_size, embedding_dim] = embedding.dims();
        assert_eq!(
            embedding_dim, 1024,
            "Input embedding dimensions must be 1024"
        );
        // Forward pass through dense layers
        let x = relu(self.layer1.forward(embedding));
        sigmoid(self.layer2.forward(x)) * 255.0
    }

    pub(crate) fn to_bytes(&self) -> Bytes {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        Bytes::from(recorder.record(self.clone().into_record(), ()).unwrap())
    }
}
