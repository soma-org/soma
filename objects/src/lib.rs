use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use std::sync::Arc;
use types::{
    crypto::NetworkKeyPair,
    error::{ObjectError, ObjectResult},
    metadata::{
        DownloadMetadata, Metadata, MetadataV1, MtlsDownloadMetadata, MtlsDownloadMetadataV1,
        ObjectPath,
    },
};
use url::Url;

pub mod networking;

#[async_trait]
pub trait PersistentStore: Send + Sync + Sized + 'static {
    fn object_store(&self) -> &Arc<dyn ObjectStore>;
    async fn download_metadata(&self, path: ObjectPath) -> ObjectResult<DownloadMetadata>;
}

#[async_trait]
pub trait EphemeralStore: Send + Sync + Sized + 'static {
    type Buffer: AsRef<[u8]> + Send + Sync;
    fn object_store(&self) -> &Arc<dyn ObjectStore>;
    async fn buffer_object(&self, path: ObjectPath) -> ObjectResult<Self::Buffer>;
}

#[derive(Clone)]
pub struct PersistentInMemoryStore {
    object_store: Arc<dyn ObjectStore>,
    object_server_base_url: Url,
    network_keypair: NetworkKeyPair,
}

impl PersistentInMemoryStore {
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        object_server_base_url: Url,
        network_keypair: NetworkKeyPair,
    ) -> Self {
        Self {
            object_store,
            object_server_base_url,
            network_keypair,
        }
    }
}

#[async_trait]
impl PersistentStore for PersistentInMemoryStore {
    fn object_store(&self) -> &Arc<dyn ObjectStore> {
        &self.object_store
    }
    async fn download_metadata(&self, path: ObjectPath) -> ObjectResult<DownloadMetadata> {
        if let Ok(meta) = self.object_store.head(&path.path()).await {
            let url = self.object_server_base_url.clone();
            //TODO: need to fix with signed params....
            let metadata = Metadata::V1(MetadataV1::new(path.checksum(), meta.size));
            Ok(DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(
                MtlsDownloadMetadataV1::new(self.network_keypair.public(), url, metadata),
            )))
        } else {
            Err(ObjectError::NotFound("object not found".to_string()))
        }
    }
}

#[derive(Clone)]
pub struct EphemeralInMemoryStore {
    object_store: Arc<dyn ObjectStore>,
}

impl EphemeralInMemoryStore {
    pub fn new(object_store: Arc<dyn ObjectStore>) -> Self {
        Self { object_store }
    }
}

#[async_trait]
impl EphemeralStore for EphemeralInMemoryStore {
    type Buffer = Bytes;
    fn object_store(&self) -> &Arc<dyn ObjectStore> {
        &self.object_store
    }
    async fn buffer_object(&self, path: ObjectPath) -> ObjectResult<Self::Buffer> {
        self.object_store
            .get(&path.path())
            .await
            .map_err(|e| ObjectError::StorageFailure(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| ObjectError::StorageFailure(e.to_string()))
    }
}

// #[async_trait]
// impl SafetensorBuffer for LocalFileSystem {
//     type Buffer = Mmap;
//     async fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer> {
//         let fs_path = self
//             .path_to_filesystem(&path.path())
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         let file =
//             File::open(fs_path).map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         let mmap = unsafe { MmapOptions::new().map(&file) }
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?;
//         Ok(mmap)
//     }
// }

// #[async_trait]
// impl SafetensorBuffer for InMemory {
//     type Buffer = Bytes;
//     async fn safetensor_buffer(&self, path: ObjectPath) -> EvaluationResult<Self::Buffer> {
//         self.get(&path.path())
//             .await
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))?
//             .bytes()
//             .await
//             .map_err(|e| EvaluationError::StorageFailure(e.to_string()))
//     }
// }

// #[cfg(test)]
// mod tests {
//     use bytes::Bytes;
//     use object_store::{local::LocalFileSystem, memory::InMemory, ObjectStore, PutPayload};
//     use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
//     use tempdir::TempDir;
//     use types::{checksum::Checksum, metadata::ObjectPath};

//     use crate::evaluation::core::safetensor_buffer::SafetensorBuffer;

//     // Basic async test
//     #[tokio::test]
//     async fn test_file_object_store() {
//         let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//         let data_bytes: Vec<u8> = data.into_iter().flat_map(|x| x.to_le_bytes()).collect();
//         let tensor = TensorView::new(Dtype::F32, vec![4], &data_bytes).unwrap();
//         let tensors = vec![("test_tensor".to_string(), tensor)];

//         let checksum = Checksum::new_from_bytes(&data_bytes);
//         let object_path = ObjectPath::Uploads(checksum);

//         let serialized = serialize(tensors, &None).unwrap();

//         let temp_dir = TempDir::new("test_dir").expect("Failed to create temp directory");
//         let temp_path = temp_dir.path();

//         let store = LocalFileSystem::new();

//         store
//             .put(
//                 &object_path.path(),
//                 PutPayload::from_bytes(Bytes::from(serialized)),
//             )
//             .await
//             .unwrap();

//         let buffer = store.safetensor_buffer(object_path).await.unwrap();

//         let t = SafeTensors::deserialize(&buffer).unwrap();
//         let x = t.tensor("test_tensor").unwrap();
//         assert_eq!(x.shape(), vec![4]);
//         assert_eq!(x.dtype(), Dtype::F32);
//         assert_eq!(x.data(), data_bytes);
//     }

//     #[tokio::test]
//     async fn test_memory_object_store() {
//         let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//         let data_bytes: Vec<u8> = data.into_iter().flat_map(|x| x.to_le_bytes()).collect();
//         let tensor = TensorView::new(Dtype::F32, vec![4], &data_bytes).unwrap();

//         let checksum = Checksum::new_from_bytes(&data_bytes);
//         // Create a map of tensors (e.g., a single tensor named "my_tensor")
//         let tensors = vec![("test_tensor".to_string(), tensor)];

//         let serialized = serialize(tensors, &None).unwrap();

//         let store = InMemory::new();
//         let object_path = ObjectPath::Uploads(checksum);
//         store
//             .put(
//                 &object_path.path(),
//                 PutPayload::from_bytes(Bytes::from(serialized)),
//             )
//             .await
//             .unwrap();

//         let buffer = store.safetensor_buffer(object_path).await.unwrap();

//         let t = SafeTensors::deserialize(&buffer).unwrap();
//         let x = t.tensor("test_tensor").unwrap();
//         assert_eq!(x.shape(), vec![4]);
//         assert_eq!(x.dtype(), Dtype::F32);
//         assert_eq!(x.data(), data_bytes);
//     }
// }
