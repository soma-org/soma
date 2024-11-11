use std::env;
use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use pyo3::{sync::GILOnceCell, PyAny};

use crate::error::{ShardError, ShardResult};

use super::{DataRefs, Model};

/// Selects the storage backend to be used
pub(crate) enum StorageBackend {
    /// contains the specific path to use
    Filesystem(String),
}

/// Python related details for interfacing uniquely
pub(crate) struct PythonModel {
    /// Specifies where data is physically stored. Since data is passed via
    /// the storage backend, it is crucial that both the rust and python programs
    /// have the same storage backend.
    storage_backend: StorageBackend,
    /// The path to the .venv directory
    virtual_environment: PathBuf,
    /// The root of the python project
    project_root: PathBuf,
    /// The file that contains the correctly wrapped python class
    entry_point: PathBuf,
    /// The python interpreter using the specified virtual env
    python_interpreter: PathBuf,
    /// contains an active module wrapper after initialization
    module_wrapper: Option<GILOnceCell<Option<Py<PyAny>>>>,
}

impl PythonModel {
    /// creates a new python model adapter
    fn new(
        storage_backend: StorageBackend,
        virtual_environment: PathBuf,
        project_root: PathBuf,
        entry_point: PathBuf,
        python_interpreter: PathBuf,
    ) -> Self {
        // TODO: find site packages using the .venv location and add to python path
        let python_path = &project_root;

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

        Self {
            storage_backend,
            virtual_environment,
            project_root,
            entry_point,
            python_interpreter,
            module_wrapper: None,
        }
    }
}

// TODO: improve the concrete typing of the python class / methods / args
// TODO: add more specific shard error types
// TODO: define the attribute / method names as constants defined outside of the functions
impl Model for PythonModel {
    fn init(&mut self) -> ShardResult<()> {
        let entry_point_code = std::fs::read_to_string(&self.entry_point)
            .map_err(|e| ShardError::FailedLoadingPythonCode(
                format!("Failed to read entry point file: {}", e)
            ))?;

        let module_wrapper = GILOnceCell::new();
        
        Python::with_gil(|py| -> ShardResult<()> {
            module_wrapper.get_or_init(py, || {
                // Use Option to represent potential failure
                PyModule::from_code_bound(py, &entry_point_code, "", "")
                    .and_then(|py_module| py_module.getattr("Wrapper"))
                    .and_then(|wrapper| wrapper.call0())
                    .map(|instance| Some(instance.into()))
                    .unwrap_or(None)
            });

            // Check if we got a valid wrapper
            match module_wrapper.get(py).and_then(|w| w.as_ref()) {
                Some(_) => {
                    self.module_wrapper = Some(module_wrapper);
                    Ok(())
                }
                None => Err(ShardError::FailedLoadingPythonCode(
                    "Module wrapper initialization failed".to_string()
                ))
            }
        })
    }

    fn call(&self, input: DataRefs) -> ShardResult<DataRefs> {

        Python::with_gil(|py| {

            //TODO: have concrete typing rather than using the PyAny
            let _py_result = self.module_wrapper
                .as_ref()
                .ok_or_else(|| ShardError::FailedLoadingPythonCode("Module wrapper not initialized".into()))?
                .get(py)
                .and_then(|wrapper| wrapper.as_ref())
                .ok_or_else(|| ShardError::FailedLoadingPythonCode("Python instance not available".into()))?
                //TODO: change method and add args
                .call_method0(py, "process_data")
                .map_err(|e| ShardError::FailedLoadingPythonCode(format!("Failed to call Python method: {}", e)))?;


            // map the result to data refs

            // Ok(result)
            Ok(DataRefs::default())

        })

    }
}
