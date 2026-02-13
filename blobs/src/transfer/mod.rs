use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use object_store::path::Path;
use tokio::sync::Semaphore;
use types::error::{BlobError, BlobResult};
use types::metadata::Metadata;

use crate::BlobPath;
use crate::engine::{BlobEngine, BlobReader};

#[async_trait]
pub trait BlobTransfer: Send + Sync + 'static {
    async fn transfer(&self, blob_path: BlobPath, metadata: Metadata) -> BlobResult<()>;
}

pub struct StoreTransfer<Src: ObjectStore, Dst: ObjectStore> {
    source: Arc<Src>,
    dest: Arc<Dst>,
    engine: BlobEngine,
}

impl<Src: ObjectStore, Dst: ObjectStore> StoreTransfer<Src, Dst> {
    pub fn new(
        source: Arc<Src>,
        dest: Arc<Dst>,
        semaphore: Arc<Semaphore>,
        chunk_size: u64,
        ns_per_byte: u16,
    ) -> BlobResult<Self> {
        let engine = BlobEngine::new(semaphore, chunk_size, ns_per_byte)?;
        Ok(Self { source, dest, engine })
    }
}

struct StoreReader<S: ObjectStore> {
    store: Arc<S>,
    path: Path,
}

#[async_trait]
impl<S: ObjectStore> BlobReader for StoreReader<S> {
    async fn get_full(&self, timeout: Duration) -> BlobResult<Bytes> {
        match tokio::time::timeout(timeout, async {
            let get_result =
                self.store.get(&self.path).await.map_err(BlobError::ObjectStoreError)?;
            get_result.bytes().await.map_err(BlobError::ObjectStoreError)
        })
        .await
        {
            Ok(Ok(bytes)) => Ok(bytes),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(BlobError::Timeout),
        }
    }
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> BlobResult<Bytes> {
        tokio::time::timeout(timeout, self.store.get_range(&self.path, range))
            .await
            .map_err(|_| BlobError::Timeout)?
            .map_err(BlobError::ObjectStoreError)
    }
}

#[async_trait]
impl<Src: ObjectStore, Dst: ObjectStore> BlobTransfer for StoreTransfer<Src, Dst> {
    async fn transfer(&self, blob_path: BlobPath, metadata: Metadata) -> BlobResult<()> {
        let reader = Arc::new(StoreReader { store: self.source.clone(), path: blob_path.path() });
        self.engine.download(reader, self.dest.clone(), blob_path, metadata).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MIN_PART_SIZE;
    use fastcrypto::hash::HashFunction;
    use object_store::memory::InMemory;
    use types::{
        checksum::Checksum,
        crypto::DefaultHash,
        metadata::{Metadata, MetadataV1},
    };

    fn setup_stores_and_data(data: &[u8]) -> (Arc<InMemory>, Arc<InMemory>, Checksum, Metadata) {
        let mut hasher = DefaultHash::new();
        hasher.update(data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let source = Arc::new(InMemory::new());
        let dest = Arc::new(InMemory::new());
        (source, dest, checksum, metadata)
    }

    #[tokio::test]
    async fn transfer_small_object() {
        let chunk_size = MIN_PART_SIZE;
        let data = vec![1u8; chunk_size as usize / 2];
        let (source, dest, checksum, metadata) = setup_stores_and_data(&data);
        let blob_path = BlobPath::Data(0, checksum);

        source.put(&blob_path.path(), data.clone().into()).await.unwrap();

        let transfer =
            StoreTransfer::new(source, dest.clone(), Arc::new(Semaphore::new(5)), chunk_size, 40)
                .unwrap();

        transfer.transfer(blob_path.clone(), metadata).await.unwrap();

        let result = dest.get(&blob_path.path()).await.unwrap();
        assert_eq!(result.bytes().await.unwrap().to_vec(), data);
    }

    #[tokio::test]
    async fn transfer_large_object_multipart() {
        let chunk_size = MIN_PART_SIZE;
        let data = vec![2u8; (chunk_size as usize) * 3];
        let (source, dest, checksum, metadata) = setup_stores_and_data(&data);
        let blob_path = BlobPath::Data(1, checksum);

        source.put(&blob_path.path(), data.clone().into()).await.unwrap();

        let transfer =
            StoreTransfer::new(source, dest.clone(), Arc::new(Semaphore::new(3)), chunk_size, 40)
                .unwrap();

        transfer.transfer(blob_path.clone(), metadata).await.unwrap();

        let result = dest.get(&blob_path.path()).await.unwrap();
        assert_eq!(result.bytes().await.unwrap().to_vec(), data);
    }

    #[tokio::test]
    async fn transfer_existing_object_skips() {
        let chunk_size = MIN_PART_SIZE;
        let data = vec![3u8; chunk_size as usize];
        let (source, dest, checksum, metadata) = setup_stores_and_data(&data);
        let blob_path = BlobPath::Data(2, checksum);

        source.put(&blob_path.path(), data.clone().into()).await.unwrap();
        dest.put(&blob_path.path(), data.clone().into()).await.unwrap();

        let transfer =
            StoreTransfer::new(source, dest.clone(), Arc::new(Semaphore::new(2)), chunk_size, 40)
                .unwrap();

        transfer.transfer(blob_path.clone(), metadata).await.unwrap();

        let result = dest.get(&blob_path.path()).await.unwrap();
        assert_eq!(result.bytes().await.unwrap().to_vec(), data);
    }

    #[tokio::test]
    async fn transfer_checksum_mismatch_fails() {
        let chunk_size = MIN_PART_SIZE;
        let data = vec![4u8; chunk_size as usize];

        let wrong_checksum = Checksum::new_from_hash(DefaultHash::digest(b"wrong").into());
        let metadata = Metadata::V1(MetadataV1::new(wrong_checksum, data.len()));
        let blob_path = BlobPath::Data(3, wrong_checksum);

        let source = Arc::new(InMemory::new());
        source.put(&blob_path.path(), data.into()).await.unwrap();
        let dest = Arc::new(InMemory::new());

        let transfer =
            StoreTransfer::new(source, dest, Arc::new(Semaphore::new(2)), chunk_size, 40).unwrap();

        let err = transfer.transfer(blob_path, metadata).await.unwrap_err();
        assert!(matches!(err, BlobError::ChecksumMismatch { .. }));
    }
}
