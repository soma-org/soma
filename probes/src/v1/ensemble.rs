use burn::{
    module::Module,
    tensor::{Int, Tensor, backend::Backend},
};

use crate::v1::probe::Probe;

#[derive(Module, Debug)]
pub struct Ensemble<B: Backend> {
    probes: Vec<Probe<B>>,
    weights: Tensor<B, 3>,
    capacity: usize,
}

impl<B: Backend> Ensemble<B> {
    pub fn new(probes: Vec<Probe<B>>, weights: Tensor<B, 3>) -> Self {
        assert_eq!(probes.len(), weights.dims()[0]);
        let capacity = probes.len();
        Self {
            probes,
            weights,
            capacity,
        }
    }
    pub fn forward(&self, context: Tensor<B, 3>, positions: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits: Vec<_> = self
            .probes
            .iter()
            .map(|p| p.forward(context.clone(), positions.clone()))
            .collect();

        let stacked = Tensor::stack(logits, 0); // [P, B, V]
        println!("{:?}", stacked.dims());

        println!("{:?}", self.weights.dims());
        (stacked * self.weights.clone())
            .sum_dim(0)
            .squeeze_dim::<2>(0) // → [B, V]
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::{Int, Tolerance, ops::FloatElem},
    };

    use crate::v1::{
        V1_EMBEDDING_DIM,
        ensemble::Ensemble,
        probe::{Probe, ProbeConfig},
    };

    type TestBackend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<TestBackend>;
    // type TestBackend = burn::backend::Metal;

    #[test]
    fn test_v1_ensemble() {
        let num_probes = 5;
        let batch_size = 2;
        let seq_len = 32;
        let embedding_dim = V1_EMBEDDING_DIM;

        let config = ProbeConfig::new();

        let device = Default::default();
        let probe: Probe<TestBackend> = config.init(&device);
        let probes: Vec<Probe<TestBackend>> = (0..num_probes).map(|_| probe.clone()).collect();

        let weights_flat: Tensor<TestBackend, 1> =
            Tensor::full([num_probes], 1.0f32 / num_probes as f32, &device);
        let weights: Tensor<TestBackend, 3> = weights_flat.reshape([num_probes, 1, 1]); // [P,1,1]

        let ensemble = Ensemble::new(probes, weights);

        let input_tensor =
            Tensor::<TestBackend, 3>::ones([batch_size, seq_len, embedding_dim], &device);

        let positions = Tensor::<TestBackend, 2, Int>::ones([batch_size, seq_len], &device);

        let output = ensemble.forward(input_tensor.clone(), positions.clone());
        let probe_output = probe.forward(input_tensor, positions);
        output
            .to_data()
            .assert_approx_eq::<FT>(&probe_output.to_data(), Tolerance::default());
    }
}
