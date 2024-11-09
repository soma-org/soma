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
    module_wrapper: Option<GILOnceCell<Py<PyAny>>>,
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

impl Model for PythonModel {
    fn init(&mut self) -> ShardResult<()> {
        let module_wrapper = GILOnceCell::new();
        Python::with_gil(|py| -> ShardResult<()> {
            let entry_point_code = std::fs::read_to_string(&self.entry_point)
                .map_err(|e| ShardError::FailedLoadingPythonCode(e.to_string()))?;

            module_wrapper.get_or_init(py, || {
                // TODO: figure out how to improve error handling here
                let py_module = PyModule::from_code_bound(py, &entry_point_code, "", "").unwrap();
                // TODO: type the module wrapper better, and figure out whether the attribute name should be configurable
                let module_wrapper: Py<PyAny> = py_module.getattr("Wrapper").unwrap().into();
                // TODO: pass in the backend configuration to the python wrapper
                module_wrapper.call0(py).unwrap()
            });

            Ok(())
        })?;
        self.module_wrapper = Some(module_wrapper);
        Ok(())
    }

    fn call(&self, input: DataRefs) -> ShardResult<DataRefs> {
        Ok(DataRefs::default())
        // TODO: improve error handling and return a typed response.

        // if let module_wrapper= Some(self.module_wrapper) {

        // }
        // module_wrapper.call_method0().unwrap();
    }
}
