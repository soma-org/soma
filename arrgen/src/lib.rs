use ndarray::ArrayD;
use ndarray_rand::{rand_distr::Normal, RandomExt};
use numpy::{IxDyn, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use rand::{distributions::Uniform, rngs::StdRng, SeedableRng};

pub fn uniform_array(seed: u64, shape: Vec<usize>, min: f64, max: f64) -> ArrayD<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(min, max);
    let array_shape = IxDyn(&shape);
    let array = ArrayD::random_using(array_shape, dist, &mut rng);
    array
}

pub fn normal_array(seed: u64, shape: Vec<usize>, mean: f64, std_dev: f64) -> ArrayD<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(mean, std_dev).unwrap();
    let array_shape = IxDyn(&shape);
    let array = ArrayD::random_using(array_shape, dist, &mut rng);
    array
}

pub fn constant_array(shape: Vec<usize>, value: f64) -> ArrayD<f64> {
    let array_shape = IxDyn(&shape);
    ArrayD::from_elem(array_shape, value)
}

#[pyfunction]
#[pyo3(name = "uniform_array")]
fn py_uniform_array(
    py: Python<'_>,
    seed: u64,
    shape: Vec<usize>,
    min: f64,
    max: f64,
) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let array = uniform_array(seed, shape, min, max);
    Ok(array.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "normal_array")]
fn py_normal_array(
    py: Python<'_>,
    seed: u64,
    shape: Vec<usize>,
    mean: f64,
    std_dev: f64,
) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let array = normal_array(seed, shape, mean, std_dev);
    Ok(array.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "constant_array")]
fn py_constant_array(
    py: Python<'_>,
    shape: Vec<usize>,
    value: f64,
) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let array = constant_array(shape, value);
    Ok(array.to_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn arrgen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_uniform_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_normal_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_constant_array, m)?)?;
    Ok(())
}
