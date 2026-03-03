// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
#[cfg(feature = "python")]
use numpy::{PyArrayDyn, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;

pub fn uniform_array(seed: u64, shape: &[usize], min: f32, max: f32) -> ArrayD<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(min, max).unwrap();
    let array_shape = IxDyn(shape);
    ArrayD::random_using(array_shape, dist, &mut rng)
}

pub fn normal_array(seed: u64, shape: &[usize], mean: f32, std_dev: f32) -> ArrayD<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(mean, std_dev).unwrap();
    let array_shape = IxDyn(shape);
    ArrayD::random_using(array_shape, dist, &mut rng)
}

/// Returns `(flat_vec, shape)` without exposing ndarray types to callers.
pub fn normal_array_raw(
    seed: u64,
    shape: &[usize],
    mean: f32,
    std_dev: f32,
) -> (Vec<f32>, Vec<usize>) {
    let arr = normal_array(seed, shape, mean, std_dev);
    let s = arr.shape().to_vec();
    let (v, _) = arr.into_raw_vec_and_offset();
    (v, s)
}

/// Returns `(flat_vec, shape)` without exposing ndarray types to callers.
pub fn uniform_array_raw(seed: u64, shape: &[usize], min: f32, max: f32) -> (Vec<f32>, Vec<usize>) {
    let arr = uniform_array(seed, shape, min, max);
    let s = arr.shape().to_vec();
    let (v, _) = arr.into_raw_vec_and_offset();
    (v, s)
}

pub fn constant_array(shape: &[usize], value: f32) -> ArrayD<f32> {
    let array_shape = IxDyn(shape);
    ArrayD::from_elem(array_shape, value)
}

/// Returns `(flat_vec, shape)` without exposing ndarray types to callers.
pub fn constant_array_raw(shape: &[usize], value: f32) -> (Vec<f32>, Vec<usize>) {
    let arr = constant_array(shape, value);
    let s = arr.shape().to_vec();
    let (v, _) = arr.into_raw_vec_and_offset();
    (v, s)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "uniform_array")]
fn py_uniform_array(
    py: Python<'_>,
    seed: u64,
    shape: Vec<usize>,
    min: f32,
    max: f32,
) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    let array = uniform_array(seed, &shape, min, max);
    Ok(array.to_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "normal_array")]
fn py_normal_array(
    py: Python<'_>,
    seed: u64,
    shape: Vec<usize>,
    mean: f32,
    std_dev: f32,
) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    let array = normal_array(seed, &shape, mean, std_dev);
    Ok(array.to_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "constant_array")]
fn py_constant_array(
    py: Python<'_>,
    shape: Vec<usize>,
    value: f32,
) -> PyResult<Bound<'_, PyArrayDyn<f32>>> {
    let array = constant_array(&shape, value);
    Ok(array.to_pyarray(py))
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn arrgen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_uniform_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_normal_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_constant_array, m)?)?;
    Ok(())
}
