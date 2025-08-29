use burn::{
    module::Module,
    prelude::Backend,
    tensor::{Int, Tensor},
};
use std::marker::PhantomData;

const MAX_WAVELENGTH: f32 = 10_000.0;

#[derive(Module, Debug)]
pub struct RotaryEncoding<B: Backend> {
    marker: PhantomData<B>,
}

impl<B: Backend> RotaryEncoding<B> {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }

    pub fn apply_rope(
        &self,
        inputs: Tensor<B, 4>, // [batch_size, seq_len, num_heads, head_dim]
        positions: Tensor<B, 2, Int>, // [batch_size, seq_len]
        head_dim: usize,
        max_wavelength: f32,
        scale_factor: f32,
    ) -> Tensor<B, 4> {
        if scale_factor < 1.0 {
            panic!("scale_factor must be >= 1.0, got {}", scale_factor);
        }
        if head_dim % 2 != 0 {
            panic!("head_dim must be even, got {}", head_dim);
        }

        // Calculate frequencies: 1 / (max_wavelength ^ (2i / head_dim))
        let device = inputs.device();

        let fraction = Tensor::<B, 1, Int>::arange(0..(head_dim / 2) as i64, &device)
            .float()
            .mul_scalar(2)
            .div_scalar(head_dim as i64);
        println!("fraction: {}", fraction);
        let base_tensor = Tensor::full_like(&fraction, max_wavelength);
        let timescale = base_tensor.powf(fraction);
        println!("timescale: {}", timescale);

        // Prepare sinusoid input: positions * theta
        let positions_float = positions.float().unsqueeze_dim::<3>(2); // [batch_size, seq_len] -> [batch_size, seq_len, 1]
        println!("positions_float: {}", positions_float);
        let theta = timescale.unsqueeze::<2>().unsqueeze::<3>(); // [head_dim/2] -> [1, head_dim/2] -> [1, 1, head_dim/2]
        println!("theta: {}", theta);

        let sinusoid_inp = positions_float.div(theta);

        println!("sin inp 1: {}", sinusoid_inp);

        let sinusoid_inp = sinusoid_inp.unsqueeze_dim::<4>(2);
        println!("sin inp 2 {}", sinusoid_inp);

        let sinusoid_inp = sinusoid_inp.div_scalar(scale_factor);
        println!("final sin inp {}", sinusoid_inp);
        // println!("sin imp: {}", sinusoid_inp);

        let sin = sinusoid_inp.clone().sin();
        println!("sin {}", sin);
        let cos = sinusoid_inp.cos();
        println!("cos {}", cos);

        let chunks = inputs.chunk(2, 3);
        if chunks.len() != 2 {
            panic!(
                "Expected 2 chunks from splitting head_dim, got {}",
                chunks.len()
            );
        }
        let first_half = chunks[0].clone();
        let second_half = chunks[1].clone();

        let first_part = first_half.clone() * cos.clone() - second_half.clone() * sin.clone();
        let second_part = second_half * cos + first_half * sin;

        let out = Tensor::cat(vec![first_part, second_part], 3);
        out
    }
}
#[cfg(test)]
mod tests {
    use super::{RotaryEncoding, MAX_WAVELENGTH};
    use burn::backend::NdArray;
    use burn::tensor::{Int, Shape, Tensor, TensorData};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rotary_encoding_forward() {
        let device = Default::default();

        let batch = 2;
        let seq_len = 2;
        let num_heads = 1;
        let embedding_dim = 4;
        let head_dim = embedding_dim / num_heads;

        let re = RotaryEncoding::<TestBackend>::new();

        // Create input tensor with shape [batch_size, seq_len, num_heads, head_dim]
        let shape = Shape::new([batch, seq_len, num_heads, head_dim]);
        let input: Tensor<TestBackend, 4> = Tensor::ones(shape, &device);

        let positions: Tensor<TestBackend, 2, Int> = Tensor::from_ints(
            TensorData::new(vec![0, 1, 0, 1], Shape::new([batch, seq_len])),
            &device,
        );
        println!("{}", positions);

        // Apply RoPE
        let output = re.apply_rope(input, positions, head_dim, MAX_WAVELENGTH, 1.0);
        let output: Tensor<NdArray, 3> =
            output.reshape(Shape::new([batch, seq_len, embedding_dim]));
        println!("Output: {}", output);
    }
}
