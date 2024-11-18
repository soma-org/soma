pub(crate) mod python;

use crate::{error::ShardResult, types::checksum::Checksum};

/// The `Model` trait establishes a standard set of operations that can be taken
/// on any implementation of a model. There are two primary functions, one to initialize
/// the model, and call to trigger processing.
pub(crate) trait Model {
    /// Init allows the underlying model to perform any necessary initializations prior
    /// to being called. Additionally, init may pass an implementation specific config file
    /// to the underlying model.
    fn init(&mut self) -> ShardResult<()>;
    /// call takes some data references, sends the data to the model, and receives data references
    /// for the model output in return. Model implementations are expected to interface with the storage
    /// backend directly and only supply references in return.
    fn call(&self, input: DataRefs) -> ShardResult<DataRefs>;
}

/// Refers to data by the id of the data, not the actual data itself. `DataRefs` are used
/// to communicate without
#[derive(Default)]
pub(crate) struct DataRefs {
    /// List of file checksums
    checksums: Vec<Checksum>,
}
