use pyo3::{
    sync::GILOnceCell,
    types::{PyAnyMethods, PyModule},
    PyObject, Python,
};
use std::{
    env,
    path::{Path, PathBuf},
};

use numpy::{
    ndarray::{Array, ArrayD},
    IxDyn, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods,
};

use super::Model;
use crate::error::{ShardError, ShardResult};
use async_trait::async_trait;

pub const REGISTERED_MODULE_ATTR: &str = "__REGISTERED_MODULE__";

pub(crate) struct PythonInterpreter {}

impl PythonInterpreter {
    fn new(
        virtual_environment: &Path,
        project_root: &Path,
        python_interpreter: &Path,
    ) -> ShardResult<Self> {
        let site_packages_path = find_site_packages_path(virtual_environment).ok_or_else(|| {
            ShardError::PathError("Invalid UTF-8 in site-packages path".to_string())
        })?;

        let python_path = format!(
            "{}:{}",
            site_packages_path
                .to_str()
                .ok_or_else(|| ShardError::PathError(
                    "Invalid UTF-8 in site-packages path".to_string()
                ))?,
            project_root.to_str().ok_or_else(|| ShardError::PathError(
                "Invalid UTF-8 in project root path".to_string()
            ))?
        );
        // SAFETY: setting environment variables in multi-threaded programs is considered unsafe.
        // since this is called prior to any multi-threaded execution of the program, usage is acceptable.
        #[allow(unsafe_code)]
        unsafe {
            // sets the python interpreter to the interpreter in the virtual environment
            env::set_var("PYO3_PYTHON", &python_interpreter);
            // sets where the interpreter should look when importing in python
            env::set_var("PYTHONPATH", python_path);
        }
        // initialize py03
        pyo3::prepare_freethreaded_python();

        Ok(Self {})
    }

    fn new_module(&self, entry_point: &Path) -> ShardResult<PythonModule> {
        let entry_point_code = std::fs::read_to_string(&entry_point).map_err(|e| {
            ShardError::FailedLoadingPythonModule(format!("Failed to read entry point file: {}", e))
        })?;

        let module: GILOnceCell<PyObject> = GILOnceCell::new();

        Python::with_gil(|py| -> ShardResult<()> {
            module.get_or_try_init(py, || -> Result<PyObject, ShardError> {
                // TODO: change file name module to match the file provided by the 3rd party developer
                let py_module = PyModule::from_code_bound(py, &entry_point_code, "", "main")
                    .map_err(|e| {
                        ShardError::FailedLoadingPythonModule(format!(
                            "Failed to load module: {}",
                            e
                        ))
                    })?;

                let attr = py_module.getattr(REGISTERED_MODULE_ATTR).map_err(|e| {
                    ShardError::FailedLoadingPythonModule(format!("Failed to get attribute: {}", e))
                })?;

                let result = attr.call0().map_err(|e| {
                    ShardError::FailedLoadingPythonModule(format!("Constructor failed: {}", e))
                })?;

                Ok(result.into())
            })?;

            Ok(())
        })?;

        Ok(PythonModule { module })
    }
}
pub struct PythonModule {
    module: GILOnceCell<PyObject>,
}

#[async_trait]
impl Model for PythonModule {
    async fn call(&self, input: &ArrayD<f32>) -> ShardResult<ArrayD<f32>> {
        // Clone or get what we need before the spawn_blocking
        let input = input.clone(); // Clone the input

        // TODO: figure out how to improve this? I probably shouldn't aquire the gil twice?
        // potentially run this using a dedicated OS thread and not using tokio spawn blocking


        let module = Python::with_gil(|py| -> ShardResult<PyObject> {
            Ok(self
                .module
                .get(py)
                .ok_or_else(|| {
                    ShardError::FailedCallingPythonModule(
                        "Could not get registered module from GIL".to_string(),
                    )
                })?
                .clone_ref(py))
        })?;

        let result = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| -> ShardResult<ArrayD<f32>> {
                // Use the cloned module inside the blocking task
                let py_input = PyArrayDyn::from_array_bound(py, &input);

                let result = module
                    .call_method1(py, "__call__", (py_input,))
                    .map_err(|e| {
                        ShardError::FailedCallingPythonModule(format!("call_method1 failed: {}", e))
                    })?;

                let array = result.downcast_bound::<PyArrayDyn<f32>>(py).map_err(|e| {
                    ShardError::FailedCallingPythonModule(format!(
                        "Failed to convert result: {}",
                        e
                    ))
                })?;

                let input_batch_size = input.shape().get(0).ok_or(ShardError::ArrayShapeError)?;
                let output_batch_size = array.shape().get(0).ok_or(ShardError::ArrayShapeError)?;

                if input_batch_size != output_batch_size {
                    return Err(ShardError::BatchSizeMismatch(format!(
                        "got: {}, expected: {}",
                        output_batch_size, input_batch_size
                    )));
                }

                Ok(array.to_owned_array())
            })
        })
        .await
        .map_err(|e| ShardError::ThreadError(e.to_string()))??;

        Ok(result)
    }
}

fn find_site_packages_path(venv_path: &Path) -> Option<PathBuf> {
    let lib_path = venv_path.join("lib");
    let entries = std::fs::read_dir(&lib_path).ok()?;

    for entry in entries {
        let entry = entry.ok()?;
        if entry.file_type().ok()?.is_dir() {
            let site_packages_path = entry.path().join("site-packages");
            if site_packages_path.exists() {
                return Some(site_packages_path);
            }
        }
    }

    None
}
