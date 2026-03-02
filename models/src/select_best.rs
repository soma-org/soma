// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::Tensor;
use burn::prelude::Backend;

use crate::ModelOutput;

/// Returns the index of the model output with the lowest loss.
pub fn select_best_model<B: Backend>(outputs: &[ModelOutput<B>]) -> Option<usize> {
    if outputs.is_empty() {
        return None;
    }
    let losses: Vec<_> = outputs.iter().map(|o| o.loss.clone().unsqueeze::<2>()).collect();
    let stacked = Tensor::cat(losses, 0).flatten::<1>(0, 1);
    let data = stacked.argmin(0).into_data();
    // Backend may return i32 or i64 for argmin indices; handle both.
    let idx = data.to_vec::<i64>().unwrap_or_else(|_| {
        data.to_vec::<i32>()
            .map(|v| v.into_iter().map(|x| x as i64).collect())
            .expect("argmin should return i32 or i64")
    })[0];
    Some(idx as usize)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type B = NdArray;

    #[test]
    fn select_best_model_picks_lowest_loss() {
        let device = Default::default();
        let outputs = vec![
            ModelOutput {
                embedding: Tensor::<B, 1>::from_floats([1.0, 0.0], &device),
                loss: Tensor::<B, 1>::from_floats([0.5], &device),
            },
            ModelOutput {
                embedding: Tensor::<B, 1>::from_floats([0.0, 1.0], &device),
                loss: Tensor::<B, 1>::from_floats([0.1], &device),
            },
            ModelOutput {
                embedding: Tensor::<B, 1>::from_floats([1.0, 1.0], &device),
                loss: Tensor::<B, 1>::from_floats([0.9], &device),
            },
        ];

        assert_eq!(select_best_model(&outputs), Some(1));
    }

    #[test]
    fn select_best_model_empty_returns_none() {
        let outputs: Vec<ModelOutput<B>> = vec![];
        assert!(select_best_model(&outputs).is_none());
    }

    #[test]
    fn select_best_model_single_entry() {
        let device = Default::default();
        let outputs = vec![ModelOutput {
            embedding: Tensor::<B, 1>::from_floats([1.0, 2.0], &device),
            loss: Tensor::<B, 1>::from_floats([0.42], &device),
        }];
        assert_eq!(select_best_model(&outputs), Some(0));
    }
}
