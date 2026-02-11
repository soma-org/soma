use crate::{BlobPath, MAX_PART_SIZE, MIN_PART_SIZE, readers::BlobReader};
use bytes::Bytes;
use fastcrypto::hash::HashFunction;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use std::{collections::HashMap, ops::Range, sync::Arc, time::Duration};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    task::{JoinHandle, JoinSet},
};
use types::{checksum::Checksum, crypto::DefaultHash, error::BlobError, metadata::Metadata};
use types::{error::BlobResult, metadata::MetadataAPI};

struct MultipartGuard {
    upload: Option<Box<dyn MultipartUpload>>,
}

impl MultipartGuard {
    fn new(upload: Box<dyn MultipartUpload>) -> Self {
        Self { upload: Some(upload) }
    }

    fn upload_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        self.upload.as_mut().expect("multipart already consumed")
    }

    async fn complete(mut self) -> Result<(), BlobError> {
        let mut upload = self.upload.take().expect("multipart already consumed");
        upload.complete().await.map_err(BlobError::ObjectStoreError)?;
        Ok(())
    }
}

impl Drop for MultipartGuard {
    fn drop(&mut self) {
        if let Some(mut upload) = self.upload.take() {
            tokio::spawn(async move {
                let _ = upload.abort().await;
            });
        }
    }
}

pub struct BlobDownloader {
    semaphore: Arc<Semaphore>,
    chunk_size: u64,
    ns_per_byte: u16,
}

impl BlobDownloader {
    pub fn new(semaphore: Arc<Semaphore>, chunk_size: u64, ns_per_byte: u16) -> BlobResult<Self> {
        if !(MIN_PART_SIZE..=MAX_PART_SIZE).contains(&chunk_size) {
            return Err(BlobError::InvalidChunkSize {
                size: chunk_size,
                min: MIN_PART_SIZE,
                max: MAX_PART_SIZE,
            });
        }
        Ok(Self { semaphore, chunk_size, ns_per_byte })
    }

    fn compute_timeout(num_bytes: u64, ns_per_byte: u16) -> Duration {
        let nanos = num_bytes.saturating_mul(ns_per_byte as u64);
        Duration::from_nanos(nanos)
    }

    fn generate_ranges(total_size: u64, chunk_size: u64) -> BlobResult<Vec<Range<usize>>> {
        if total_size > usize::MAX as u64 {
            return Err(BlobError::FileTooLarge);
        }
        let total = total_size as usize;
        let chunk = chunk_size as usize;
        let num_chunks = total.div_ceil(chunk);
        let mut ranges = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk;
            let end = ((i + 1) * chunk).min(total);
            ranges.push(start..end);
        }
        Ok(ranges)
    }

    pub async fn download(
        &self,
        reader: Arc<dyn BlobReader>,
        storage: Arc<dyn ObjectStore>,
        blob_path: BlobPath,
        metadata: Metadata,
    ) -> BlobResult<()> {
        if storage.head(&blob_path.path()).await.is_ok() {
            return Ok(());
        }

        let mut hasher = DefaultHash::new();

        if metadata.size() as u64 <= self.chunk_size {
            let bytes = reader
                .get_full(Self::compute_timeout(metadata.size() as u64, self.ns_per_byte))
                .await?;
            hasher.update(&bytes);
            let computed_checksum = Checksum::new_from_hash(hasher.finalize().into());

            if bytes.len() != metadata.size() {
                return Err(BlobError::SizeMismatch {
                    expected: metadata.size() as u64,
                    actual: bytes.len() as u64,
                });
            }
            if computed_checksum != metadata.checksum() {
                return Err(BlobError::ChecksumMismatch {
                    expected: metadata.checksum().to_string(),
                    actual: computed_checksum.to_string(),
                });
            }

            storage
                .put(&blob_path.path(), bytes.into())
                .await
                .map_err(BlobError::ObjectStoreError)?;
        } else {
            let ranges = Self::generate_ranges(metadata.size() as u64, self.chunk_size)?;
            let num_parts = ranges.len();
            let concurrency = self.semaphore.available_permits().max(1) * 2;
            let (tx, mut rx) = mpsc::channel::<(
                usize,
                usize,
                OwnedSemaphorePermit,
                JoinHandle<BlobResult<Bytes>>,
            )>(concurrency);
            let semaphore = self.semaphore.clone();
            let reader_clone = reader.clone();
            let ns_per_byte = self.ns_per_byte;
            let driver = tokio::spawn(async move {
                for (idx, range) in ranges.into_iter().enumerate() {
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|e| BlobError::ReadError(e.to_string()))?;

                    let reader = reader_clone.clone();
                    let expected_len = range.end - range.start;
                    let timeout = Self::compute_timeout(expected_len as u64, ns_per_byte);
                    let range_clone = range.clone();
                    let get_handle =
                        tokio::spawn(async move { reader.get_range(range_clone, timeout).await });

                    if tx.send((idx, expected_len, permit, get_handle)).await.is_err() {
                        return Err(BlobError::ReadError("Receiver dropped".to_string()));
                    }
                }
                Ok(())
            });

            let mut guard = MultipartGuard::new(
                storage
                    .put_multipart(&blob_path.path())
                    .await
                    .map_err(BlobError::ObjectStoreError)?,
            );

            let result = Self::run_multipart_transfer(
                &mut guard,
                &mut rx,
                hasher,
                num_parts,
                &metadata,
            )
            .await;

            driver.abort();

            if let Err(e) = result {
                drop(guard);
                let _ = driver.await;
                return Err(e);
            }

            guard.complete().await?;
            let _ = driver.await;
        }

        Ok(())
    }

    async fn run_multipart_transfer(
        guard: &mut MultipartGuard,
        rx: &mut mpsc::Receiver<(usize, usize, OwnedSemaphorePermit, JoinHandle<BlobResult<Bytes>>)>,
        mut hasher: DefaultHash,
        num_parts: usize,
        metadata: &Metadata,
    ) -> BlobResult<()> {
        let mut buffer: HashMap<
            usize,
            (usize, OwnedSemaphorePermit, JoinHandle<BlobResult<Bytes>>),
        > = HashMap::new();
        let mut next_idx = 0;
        let mut total_downloaded = 0u64;
        let mut put_join_set: JoinSet<BlobResult<()>> = JoinSet::new();

        while let Some((idx, expected_len, permit, get_handle)) = rx.recv().await {
            buffer.insert(idx, (expected_len, permit, get_handle));

            while let Some((expected_len, permit, get_handle)) = buffer.remove(&next_idx) {
                let bytes = match get_handle.await {
                    Ok(Ok(b)) => b,
                    Ok(Err(e)) => {
                        put_join_set.abort_all();
                        for (_, (_, _, h)) in buffer.drain() {
                            h.abort();
                        }
                        return Err(e);
                    }
                    Err(join_err) => {
                        put_join_set.abort_all();
                        for (_, (_, _, h)) in buffer.drain() {
                            h.abort();
                        }
                        return Err(BlobError::ReadError(join_err.to_string()));
                    }
                };

                if bytes.len() != expected_len {
                    put_join_set.abort_all();
                    for (_, (_, _, h)) in buffer.drain() {
                        h.abort();
                    }
                    return Err(BlobError::SizeMismatch {
                        expected: expected_len as u64,
                        actual: bytes.len() as u64,
                    });
                }

                hasher.update(&bytes);
                total_downloaded += bytes.len() as u64;
                let put_fut = guard.upload_mut().put_part(PutPayload::from_bytes(bytes));
                put_join_set.spawn(async move {
                    let result = put_fut.await.map_err(BlobError::ObjectStoreError);
                    drop(permit);
                    result
                });

                next_idx += 1;
            }
        }

        if next_idx != num_parts {
            put_join_set.abort_all();
            return Err(BlobError::ReadError("Missing parts at end of transfer".into()));
        }

        let computed_checksum = Checksum::new_from_hash(hasher.finalize().into());
        if total_downloaded != metadata.size() as u64 {
            put_join_set.abort_all();
            return Err(BlobError::SizeMismatch {
                expected: metadata.size() as u64,
                actual: total_downloaded,
            });
        }
        if computed_checksum != metadata.checksum() {
            put_join_set.abort_all();
            return Err(BlobError::ChecksumMismatch {
                expected: metadata.checksum().to_string(),
                actual: computed_checksum.to_string(),
            });
        }

        while let Some(res) = put_join_set.join_next().await {
            match res {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    put_join_set.abort_all();
                    return Err(e);
                }
                Err(join_err) => {
                    put_join_set.abort_all();
                    return Err(BlobError::ReadError(join_err.to_string()));
                }
            }
        }

        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use crate::{BlobPath, readers::store::BlobStoreReader};

    use super::*;
    use fastcrypto::hash::HashFunction;
    use object_store::memory::InMemory;
    use std::sync::Arc;
    use types::{
        checksum::Checksum,
        crypto::DefaultHash,
        metadata::{Metadata, MetadataV1},
    };

    #[tokio::test]
    async fn download_small_object() {
        let concurrency = Arc::new(Semaphore::new(5));
        let chunk_size = MIN_PART_SIZE;
        let ns_per_byte = 40;

        // Create test data (small: fits in one chunk)
        let data = vec![1u8; chunk_size as usize / 2];
        let mut hasher = DefaultHash::new();
        hasher.update(&data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len()));

        let blob_path = BlobPath::Data(0, checksum);

        // Source store (with data)
        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store.put(&blob_path.path(), data.clone().into()).await.unwrap();

        // Destination store (empty)
        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Reader from source
        let reader = Arc::new(BlobStoreReader::new(source_store.clone(), &blob_path));

        // Downloader
        let downloader = BlobDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Perform download
        downloader
            .download(reader, dest_store.clone(), blob_path.clone(), metadata.clone())
            .await
            .unwrap();

        // Verify object exists in destination
        let result = dest_store.get(&blob_path.path()).await.unwrap();
        let downloaded_bytes = result.bytes().await.unwrap();
        assert_eq!(downloaded_bytes.to_vec(), data);
    }

    #[tokio::test]
    async fn download_large_object_multipart() {
        let concurrency = Arc::new(Semaphore::new(3));
        let chunk_size = MIN_PART_SIZE;
        let ns_per_byte = 40;

        // Large data: 3.5 chunks
        let data_size = (chunk_size as usize) * 3;
        let data = vec![2u8; data_size];

        let mut hasher = DefaultHash::new();
        hasher.update(&data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());

        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let blob_path = BlobPath::Data(1, checksum);

        // Source store
        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store.put(&blob_path.path(), data.clone().into()).await.unwrap();

        // Destination store
        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Reader
        let reader = Arc::new(BlobStoreReader::new(source_store.clone(), &blob_path));

        // Downloader
        let downloader = BlobDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Download
        downloader
            .download(reader, dest_store.clone(), blob_path.clone(), metadata.clone())
            .await
            .unwrap();

        // Verify
        let result = dest_store.get(&blob_path.path()).await.unwrap();
        let downloaded_bytes = result.bytes().await.unwrap();
        assert_eq!(downloaded_bytes.to_vec(), data);
    }

    #[tokio::test]
    async fn download_existing_object_skips() {
        let chunk_size = MIN_PART_SIZE;
        let ns_per_byte = 40;
        let data = vec![3u8; chunk_size as usize];

        let mut hasher = DefaultHash::new();
        hasher.update(&data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len()));
        let blob_path = BlobPath::Data(2, checksum);

        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store.put(&blob_path.path(), data.clone().into()).await.unwrap();

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        // Pre-populate destination so download should skip
        dest_store.put(&blob_path.path(), data.clone().into()).await.unwrap();

        let reader = Arc::new(BlobStoreReader::new(source_store.clone(), &blob_path));
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = BlobDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Should succeed and skip download
        downloader.download(reader, dest_store.clone(), blob_path.clone(), metadata).await.unwrap();

        // Data should still be correct
        let result = dest_store.get(&blob_path.path()).await.unwrap();
        assert_eq!(result.bytes().await.unwrap().to_vec(), data);
    }

    #[tokio::test]
    async fn download_checksum_mismatch_fails() {
        let chunk_size = MIN_PART_SIZE;
        let ns_per_byte = 40;
        let data = vec![4u8; chunk_size as usize];

        let mut hasher = DefaultHash::new();
        hasher.update(&data);
        let _real_checksum = Checksum::new_from_hash(hasher.finalize().into());

        // Use wrong checksum
        let wrong_checksum = Checksum::new_from_hash(DefaultHash::digest(b"wrong").into());
        let metadata = Metadata::V1(MetadataV1::new(wrong_checksum, data.len()));
        let blob_path = BlobPath::Data(3, wrong_checksum);

        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store.put(&blob_path.path(), data.into()).await.unwrap();

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let reader = Arc::new(BlobStoreReader::new(source_store.clone(), &blob_path));
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = BlobDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        let err =
            downloader.download(reader, dest_store.clone(), blob_path, metadata).await.unwrap_err();

        assert!(matches!(err, BlobError::ChecksumMismatch { .. }));
    }

    #[test]
    fn compute_timeout_scales_linearly() {
        let d = BlobDownloader::compute_timeout(1_000, 40);
        assert_eq!(d, Duration::from_nanos(40_000));

        let d = BlobDownloader::compute_timeout(0, 40);
        assert_eq!(d, Duration::ZERO);

        let d = BlobDownloader::compute_timeout(1, 1);
        assert_eq!(d, Duration::from_nanos(1));
    }

    #[test]
    fn compute_timeout_saturates_on_overflow() {
        let d = BlobDownloader::compute_timeout(u64::MAX, u16::MAX);
        assert_eq!(d, Duration::from_nanos(u64::MAX));
    }

    struct ShortReader {
        data: Vec<u8>,
    }

    #[async_trait::async_trait]
    impl BlobReader for ShortReader {
        async fn get_full(&self, _timeout: Duration) -> BlobResult<Bytes> {
            Ok(Bytes::from(self.data.clone()))
        }
        async fn get_range(&self, range: Range<usize>, _timeout: Duration) -> BlobResult<Bytes> {
            let end = range.end.min(self.data.len());
            let start = range.start.min(end);
            Ok(Bytes::from(self.data[start..end].to_vec()))
        }
    }

    #[tokio::test]
    async fn download_size_mismatch_on_short_read() {
        let real_data = vec![6u8; 100];
        let short_data = vec![6u8; 50];

        let mut hasher = DefaultHash::new();
        hasher.update(&real_data);
        let checksum = Checksum::new_from_hash(hasher.finalize().into());
        let metadata = Metadata::V1(MetadataV1::new(checksum, real_data.len()));
        let blob_path = BlobPath::Data(5, checksum);

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let reader: Arc<dyn BlobReader> = Arc::new(ShortReader { data: short_data });
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = BlobDownloader::new(concurrency, MIN_PART_SIZE, 40).unwrap();

        let err = downloader.download(reader, dest_store, blob_path, metadata).await.unwrap_err();
        assert!(
            matches!(err, BlobError::SizeMismatch { expected: 100, actual: 50 }),
            "expected SizeMismatch, got {err:?}"
        );
    }
}
