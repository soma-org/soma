use objects::storage::{filesystem::FilesystemObjectStorage, memory::MemoryObjectStore};

pub mod evaluation;
pub mod inference;

pub trait RuntimeStorage: Send + Sync + Sized {}

impl RuntimeStorage for FilesystemObjectStorage {}

impl RuntimeStorage for MemoryObjectStore {}
