use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    tensor::{Int, Tensor, backend::Backend},
};

use crate::v1::{
    V1_EMBEDDING_DIM, V1_MAX_WAVELENGTH, V1_NUM_HEADS, V1_NUM_LAYERS, V1_PWFF_HIDDEN_DIM,
    V1_SCALE_FACTOR, modules::layer::Layer,
};

#[derive(Config, Debug)]
pub struct EncoderConfig {
    /// The size of the input and output features.
    #[config(default = "V1_EMBEDDING_DIM")]
    pub embedding_dim: usize,
    /// The size of the hidden inner pwff features.
    #[config(default = "V1_PWFF_HIDDEN_DIM")]
    pub pwff_hidden_dim: usize,
    /// The number of transformer layers.
    #[config(default = "V1_NUM_LAYERS")]
    pub num_layers: usize,
    /// The number of transformer heads.
    #[config(default = "V1_NUM_HEADS")]
    pub num_heads: usize,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
    /// The max wavelength for RoPE.
    #[config(default = "V1_MAX_WAVELENGTH")]
    pub max_wavelength: f32,
    /// The RoPE scale factor.
    #[config(default = "V1_SCALE_FACTOR")]
    pub scale_factor: f32,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<Layer<B>>,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let layers =
            (0..self.num_layers).map(|_| Layer::<B>::new(self, device)).collect::<Vec<_>>();
        Encoder { layers }
    }
}
impl<B: Backend> Encoder<B> {
    pub fn forward(
        &self,
        context: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut x = context;
        for layer in self.layers.iter() {
            x = layer.forward(x, positions.clone());
        }
        x
    }
}
