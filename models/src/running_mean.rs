// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::{Tensor, prelude::Backend};

pub struct RunningMean<B: Backend, const D: usize> {
    running_mean: Option<Tensor<B, D>>,
    count: usize,
}

impl<B: Backend, const D: usize> Default for RunningMean<B, D> {
    fn default() -> Self {
        Self { running_mean: None, count: 0 }
    }
}

impl<B: Backend, const D: usize> RunningMean<B, D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, tensor: Tensor<B, D>) {
        self.count += 1;
        self.running_mean = Some(match self.running_mean.take() {
            Some(mean) => mean.clone() + (tensor - mean) / self.count as f32,
            None => tensor,
        });
    }

    pub fn value(self) -> Option<Tensor<B, D>> {
        self.running_mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn to_vec(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    #[test]
    fn running_mean_single_value() {
        let device = Default::default();
        let mut acc = RunningMean::new();
        acc.add(Tensor::from_floats([2.0, 4.0], &device));
        let result = to_vec(acc.value().unwrap());
        assert_eq!(result, vec![2.0, 4.0]);
    }

    #[test]
    fn running_mean_of_two() {
        let device = Default::default();
        let mut acc = RunningMean::new();
        acc.add(Tensor::from_floats([1.0, 3.0], &device));
        acc.add(Tensor::from_floats([3.0, 5.0], &device));
        let result = to_vec(acc.value().unwrap());
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn running_mean_of_many() {
        let device = Default::default();
        let mut acc = RunningMean::new();
        for i in 1..=100 {
            acc.add(Tensor::from_floats([i as f32], &device));
        }
        let result = to_vec(acc.value().unwrap());
        assert!((result[0] - 50.5).abs() < 1e-3);
    }

    #[test]
    fn running_mean_empty_returns_none() {
        let acc: RunningMean<B, 1> = RunningMean::new();
        assert!(acc.value().is_none());
    }
}
