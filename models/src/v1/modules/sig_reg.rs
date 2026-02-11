use burn::{
    config::Config,
    module::Module,
    tensor::{Int, Tensor, backend::Backend, linalg::l2_norm, s},
};

use crate::v1::{V1_SIG_REG_POINTS, V1_SIG_REG_SLICES, V1_SIG_REG_T_MAX};

#[derive(Config, Debug)]
pub struct SIGRegConfig {
    #[config(default = "V1_SIG_REG_T_MAX")]
    pub t_max: f64,
    #[config(default = "V1_SIG_REG_POINTS")]
    pub points: usize,
    #[config(default = "V1_SIG_REG_SLICES")]
    pub slices: usize,
}

#[derive(Module, Debug)]
pub struct SIGReg<B: Backend> {
    pub t: Tensor<B, 1>,
    pub phi: Tensor<B, 1>,
    pub weights: Tensor<B, 1>,
    pub slices: usize,
}

impl SIGRegConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SIGReg<B> {
        let indices = Tensor::<B, 1, Int>::arange(0..self.points as i64, device); // step defaults to 1
        let t = indices.float() * (self.t_max / ((self.points - 1) as f64));
        let dt = self.t_max / ((self.points - 1) as f64);

        let weights = Tensor::full([self.points], dt, device);
        let weights =
            weights.slice_assign(s![1..-1], Tensor::full([self.points - 2], dt * 2.0, device));
        let phi: Tensor<B, 1> = (-t.clone().square() / 2.0).exp();
        let weights = weights * phi.clone();
        SIGReg { t, phi, weights, slices: self.slices }
    }
}

impl<B: Backend> SIGReg<B> {
    pub fn forward(&self, input: Tensor<B, 3>, noise: Tensor<B, 2>) -> Tensor<B, 1> {
        let a = noise.clone() / l2_norm(noise, 0);
        let x = input.matmul(a.unsqueeze());
        let n = *x.shape().last().unwrap() as f32;
        let x_t: Tensor<B, 4> = x.unsqueeze_dim(3);
        let x_t = x_t * self.t.clone().unsqueeze();

        let cos_mean: Tensor<B, 3> = x_t.clone().cos().mean_dim(2).squeeze_dim(2);
        let sin_mean: Tensor<B, 3> = x_t.sin().mean_dim(2).squeeze_dim(2);

        let err = ((cos_mean - self.phi.clone().unsqueeze()).powf_scalar(2.0))
            + (sin_mean.powf_scalar(2.0));

        let integrated = err * self.weights.clone().unsqueeze();
        let integrated = integrated.sum_dim(2) * n;

        integrated.mean()
    }
}
