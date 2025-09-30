use ndarray::ArrayD;
use ndarray_rand::RandomExt;
use numpy::{IxDyn, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use rand::{distributions::Uniform, rngs::StdRng, SeedableRng};

#[pyfunction]
pub fn generate_array(
    py: Python<'_>,
    shape: Vec<usize>,
    seed: u64,
    min: f64,
    max: f64,
) -> PyResult<Bound<'_, PyArrayDyn<f64>>> {
    let array = generate_array_rust(shape, seed, min, max);
    Ok(array.to_pyarray(py))
}

pub fn generate_array_rust(shape: Vec<usize>, seed: u64, min: f64, max: f64) -> ArrayD<f64> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Define the uniform distribution for the range [min, max)
    let dist = Uniform::new(min, max);

    // Create a multi-dimensional ndarray with the given shape and random values
    let array_shape = IxDyn(&shape);
    let array = ArrayD::random_using(array_shape, dist, &mut rng);
    array
}

/// A Python module implemented in Rust.
#[pymodule]
fn arrgen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_array, m)?)?;
    Ok(())
}
