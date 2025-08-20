use burn::{
    module::Module,
    nn::{
        attention::{
            generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
        },
        Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    record::{HalfPrecisionSettings, Recorder},
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use bytes::Bytes;

use crate::modules::recorder::BcsRecorder;

use super::{
    BYTE_EMBEDDING_DIM, HIDDEN_MULTIPLIER, MAX_SEQ_LEN, NUM_HEADS, NUM_LAYERS, PATCH_EMBEDDING_DIM,
    VOCAB_SIZE,
};

#[derive(Module, Debug)]
pub struct FeedForwardV1<B: Backend> {
    pub linear_1: Linear<B>,
    pub linear_2: Linear<B>,
    pub gelu: Gelu,
}

impl<B: Backend> FeedForwardV1<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear_1: LinearConfig::new(BYTE_EMBEDDING_DIM, BYTE_EMBEDDING_DIM * HIDDEN_MULTIPLIER)
                .init(device),
            linear_2: LinearConfig::new(BYTE_EMBEDDING_DIM * HIDDEN_MULTIPLIER, BYTE_EMBEDDING_DIM)
                .init(device),
            gelu: Gelu::new(),
        }
    }
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_1.forward(input);
        let x = self.gelu.forward(x);
        self.linear_2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct DecoderLayerV1<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    patch_projection: Linear<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    ff: FeedForwardV1<B>,
}

impl<B: Backend> DecoderLayerV1<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            self_attn: MultiHeadAttentionConfig::new(BYTE_EMBEDDING_DIM, NUM_HEADS)
                .with_dropout(0.0)
                .init(device),
            patch_projection: LinearConfig::new(PATCH_EMBEDDING_DIM, BYTE_EMBEDDING_DIM)
                .init(device),
            norm_1: LayerNormConfig::new(BYTE_EMBEDDING_DIM).init(device),
            norm_2: LayerNormConfig::new(BYTE_EMBEDDING_DIM).init(device),
            ff: FeedForwardV1::new(device),
        }
    }
    fn forward(
        &self,
        byte_reps: Tensor<B, 3>,
        patch_reps: Tensor<B, 3>,
        mask_attn: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let x = byte_reps;
        let residual_path = self.norm_1.forward(x.clone());

        let input_mhs = MhaInput::self_attn(residual_path).mask_attn(mask_attn);
        let residual_path = self.self_attn.forward(input_mhs).context;

        let x = x + residual_path;

        let projection = self.patch_projection.forward(patch_reps);
        let fused = x + projection;

        let residual_path = self.norm_2.forward(fused.clone());

        let residual_path = self.ff.forward(residual_path);
        let x = fused + residual_path;

        x
    }
}

#[derive(Module, Debug)]
pub struct DecoderV1<B: Backend> {
    token_embeds: Embedding<B>,
    pos_embeds: Embedding<B>,
    layers: Vec<DecoderLayerV1<B>>,
}

impl<B: Backend> DecoderV1<B> {
    pub fn init(device: &B::Device) -> Self {
        let layers = (0..NUM_LAYERS)
            .map(|_| DecoderLayerV1::<B>::new(device))
            .collect::<Vec<_>>();
        Self {
            token_embeds: EmbeddingConfig::new(VOCAB_SIZE, BYTE_EMBEDDING_DIM).init(device),
            pos_embeds: EmbeddingConfig::new(MAX_SEQ_LEN, BYTE_EMBEDDING_DIM).init(device),
            layers,
        }
    }

    pub fn forward(&self, byte_ids: Tensor<B, 2, Int>, patch_reps: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = &self.token_embeds.devices()[0];
        let [batch_size, seq_length] = byte_ids.dims();
        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let pos_embeds = self.pos_embeds.forward(index_positions);
        let token_embeds = self.token_embeds.forward(byte_ids);
        let mut x = token_embeds + pos_embeds;
        for layer in self.layers.iter() {
            x = layer.forward(x, patch_reps.clone(), mask_attn.clone());
        }
        x
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
