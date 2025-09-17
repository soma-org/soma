use std::fs::File;

use async_trait::async_trait;
use bytes::Bytes;
use memmap2::{Mmap, MmapOptions};
use objects::storage::{
    filesystem::FilesystemObjectStorage, memory::MemoryObjectStore, ObjectPath,
};
use types::error::{EvaluationError, EvaluationResult};

pub trait SafetensorBuffer: Send + Sync + Sized {
    type Buffer: AsRef<[u8]> + Send + Sync;
    fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer>;
}

impl SafetensorBuffer for FilesystemObjectStorage {
    type Buffer = Mmap;
    fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer> {
        let file = File::open(self.get_full_path(&path))
            .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
        Ok(mmap)
    }
}

#[async_trait]
impl SafetensorBuffer for MemoryObjectStore {
    type Buffer = Bytes;
    fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer> {
        self.store
            .read()
            .get(&path)
            .cloned()
            .ok_or_else(|| EvaluationError::StorageFailure("object not found in path".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use objects::storage::{
        filesystem::FilesystemObjectStorage, memory::MemoryObjectStore, ObjectPath, ObjectStorage,
    };
    use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
    use tempdir::TempDir;

    use crate::core::safetensor_buffer::SafetensorBuffer;

    // Basic async test
    #[tokio::test]
    async fn test_file_object_store() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: Vec<u8> = data.into_iter().flat_map(|x| x.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![4], &data_bytes).unwrap();
        let tensors = vec![("test_tensor".to_string(), tensor)];

        let path = ObjectPath::new("test".to_string()).unwrap();

        let serialized = serialize(tensors, &None).unwrap();

        let temp_dir = TempDir::new("test_dir").expect("Failed to create temp directory");
        let temp_path = temp_dir.path();

        let store = FilesystemObjectStorage::new(temp_path);
        store
            .put_object(&path, Bytes::from(serialized))
            .await
            .unwrap();

        let buffer = store.safetensor_buffer(path).unwrap();

        let t = SafeTensors::deserialize(&buffer).unwrap();
        let x = t.tensor("test_tensor").unwrap();
        assert_eq!(x.shape(), vec![4]);
        assert_eq!(x.dtype(), Dtype::F32);
        assert_eq!(x.data(), data_bytes);
    }

    #[tokio::test]
    async fn test_memory_object_store() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: Vec<u8> = data.into_iter().flat_map(|x| x.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![4], &data_bytes).unwrap();

        // Create a map of tensors (e.g., a single tensor named "my_tensor")
        let tensors = vec![("test_tensor".to_string(), tensor)];

        let serialized = serialize(tensors, &None).unwrap();

        let store = MemoryObjectStore::new_for_test();
        let path = ObjectPath::new("test".to_string()).unwrap();
        store
            .put_object(&path, Bytes::from(serialized))
            .await
            .unwrap();

        let buffer = store.safetensor_buffer(path).unwrap();

        let t = SafeTensors::deserialize(&buffer).unwrap();
        let x = t.tensor("test_tensor").unwrap();
        assert_eq!(x.shape(), vec![4]);
        assert_eq!(x.dtype(), Dtype::F32);
        assert_eq!(x.data(), data_bytes);
    }
}
