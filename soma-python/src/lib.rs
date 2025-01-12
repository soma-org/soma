use encoder::REGISTERED_MODULE_ATTR;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{prelude::*, types::PyType};

#[pyclass(subclass)]
pub struct Module {}

#[pymethods]
impl Module {
    #[new]
    fn new() -> Self {
        Self {}
    }
    #[allow(unused_variables)]
    fn __call__<'py>(
        &self,
        input: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Soma Module subclasses must implement call",
        ))
    }
}

#[pyfunction]
fn register<'py, 'a>(class: &'a Bound<'py, PyType>) -> PyResult<&'a Bound<'py, PyType>> {
    if !class.is_subclass_of::<Module>()? {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Class must be a subclass of soma Module",
        ));
    }

    let module_name = class.getattr("__module__")?.extract::<String>()?;
    let module = class.py().import_bound(&*module_name)?;
    module.setattr(REGISTERED_MODULE_ATTR, class)?;
    Ok(class)
}

#[pymodule]
fn soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Module>()?;
    m.add_function(wrap_pyfunction!(register, m)?)?;
    Ok(())
}
