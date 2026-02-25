// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::{Tensor, prelude::Backend, tensor::linalg::l2_norm};

pub fn cosine_distance<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let a_norm = l2_norm(a.clone(), 0);
    let b_norm = l2_norm(b.clone(), 0);
    let a_unit = a.div(a_norm);
    let b_unit = b.div(b_norm);
    a_unit.dot(b_unit).neg().add_scalar(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn cosine_distance_identical_vectors() {
        let device = Default::default();
        let a: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let b: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let dist = cosine_distance(a, b);
        let val = dist.into_data().to_vec::<f32>().unwrap()[0];
        assert!(val.abs() < 1e-5, "identical vectors should have distance ~0, got {val}");
    }

    #[test]
    fn cosine_distance_opposite_vectors() {
        let device = Default::default();
        let a: Tensor<B, 1> = Tensor::from_floats([1.0, 0.0, 0.0], &device);
        let b: Tensor<B, 1> = Tensor::from_floats([-1.0, 0.0, 0.0], &device);
        let dist = cosine_distance(a, b);
        let val = dist.into_data().to_vec::<f32>().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-5, "opposite vectors should have distance ~2, got {val}");
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let device = Default::default();
        let a: Tensor<B, 1> = Tensor::from_floats([1.0, 0.0], &device);
        let b: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0], &device);
        let dist = cosine_distance(a, b);
        let val = dist.into_data().to_vec::<f32>().unwrap()[0];
        assert!((val - 1.0).abs() < 1e-5, "orthogonal vectors should have distance ~1, got {val}");
    }

    #[test]
    fn cosine_distance_zero_vector_returns_nan() {
        let device = Default::default();
        let a: Tensor<B, 1> = Tensor::from_floats([0.0, 0.0, 0.0], &device);
        let b: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let dist = cosine_distance(a, b);
        let val = dist.into_data().to_vec::<f32>().unwrap()[0];
        assert!(val.is_nan(), "zero vector should produce NaN distance, got {val}");
    }

    #[test]
    fn cosine_distance_both_zero_vectors_returns_nan() {
        let device = Default::default();
        let a: Tensor<B, 1> = Tensor::from_floats([0.0, 0.0], &device);
        let b: Tensor<B, 1> = Tensor::from_floats([0.0, 0.0], &device);
        let dist = cosine_distance(a, b);
        let val = dist.into_data().to_vec::<f32>().unwrap()[0];
        assert!(val.is_nan(), "both zero vectors should produce NaN distance, got {val}");
    }
}
