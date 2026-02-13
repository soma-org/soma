use crate::{BlobPath, loader::BlobLoader};
use async_trait::async_trait;
use burn::store::SafetensorsStore;
use memmap2::{Mmap, MmapOptions};
use object_store::local::LocalFileSystem;
use tokio::fs::File;
use types::error::{BlobError, BlobResult};

#[async_trait]
impl BlobLoader for LocalFileSystem {
    type Buffer = Mmap;

    async fn get_buffer(&self, path: &BlobPath) -> BlobResult<Self::Buffer> {
        let fs_path = self
            .path_to_filesystem(&path.path())
            .map_err(|e| BlobError::StorageFailure(e.to_string()))?;
        let file =
            File::open(fs_path).await.map_err(|e| BlobError::StorageFailure(e.to_string()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| BlobError::StorageFailure(e.to_string()))?;
        Ok(mmap)
    }

    async fn safetensor_store(&self, path: &BlobPath) -> BlobResult<SafetensorsStore> {
        let fs_path = self
            .path_to_filesystem(&path.path())
            .map_err(|e| BlobError::StorageFailure(e.to_string()))?;
        Ok(SafetensorsStore::from_file(fs_path))
    }
}
