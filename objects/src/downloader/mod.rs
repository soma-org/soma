use crate::{readers::ObjectReader, MAX_PART_SIZE, MIN_PART_SIZE};
use bytes::Bytes;
use fastcrypto::hash::HashFunction;
use object_store::{ObjectStore, PutPayload};
use std::{collections::HashMap, ops::Range, sync::Arc, time::Duration};
use tokio::{
    sync::{mpsc, OwnedSemaphorePermit, Semaphore},
    task::{JoinHandle, JoinSet},
};
use types::{
    checksum::Checksum,
    crypto::DefaultHash,
    error::ObjectError,
    metadata::{Metadata, ObjectPath},
};
use types::{error::ObjectResult, metadata::MetadataAPI};

pub struct ObjectDownloader {
    semaphore: Arc<Semaphore>,
    chunk_size: u64,
    ns_per_byte: u16,
}

impl ObjectDownloader {
    pub fn new(semaphore: Arc<Semaphore>, chunk_size: u64, ns_per_byte: u16) -> ObjectResult<Self> {
        if chunk_size < MIN_PART_SIZE || chunk_size > MAX_PART_SIZE {
            return Err(ObjectError::VerificationError(
                "invalid chunk size".to_string(),
            ));
        }
        Ok(Self {
            semaphore,
            chunk_size,
            ns_per_byte,
        })
    }

    fn compute_timeout(num_bytes: u64, ns_per_byte: u16) -> Duration {
        let nanos = num_bytes.saturating_mul(ns_per_byte as u64);
        Duration::from_nanos(nanos)
    }

    fn generate_ranges(total_size: u64, chunk_size: u64) -> Vec<Range<u64>> {
        let num_chunks = ((total_size + chunk_size - 1) / chunk_size) as usize;
        let mut ranges = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = (i as u64) * chunk_size;
            let end = ((i as u64 + 1) * chunk_size).min(total_size);
            ranges.push(start..end);
        }
        ranges
    }

    pub async fn download(
        &self,
        reader: Arc<dyn ObjectReader>,
        storage: Arc<dyn ObjectStore>,
        object_path: ObjectPath,
        metadata: Metadata,
    ) -> ObjectResult<()> {
        if storage.head(&object_path.path()).await.is_ok() {
            return Ok(());
        }

        let mut hasher = DefaultHash::new();

        if metadata.size() <= self.chunk_size as u64 {
            println!("this ran");
            let bytes = reader
                .get_full(Self::compute_timeout(metadata.size(), self.ns_per_byte))
                .await?;
            hasher.update(&bytes);
            let computed_checksum = Checksum::new_from_hash(hasher.finalize().into());

            println!("this ran");
            println!("{}", bytes.len());
            println!("{}", metadata.size());
            if computed_checksum != metadata.checksum() || bytes.len() as u64 != metadata.size() {
                return Err(ObjectError::VerificationError(
                    "verification failed".to_string(),
                ));
            }

            println!("this ran");
            storage
                .put(&object_path.path(), bytes.into())
                .await
                .map_err(ObjectError::ObjectStoreError)?;
        } else {
            let mut total_downloaded = 0u64;
            let ranges = Self::generate_ranges(metadata.size(), self.chunk_size);
            let num_parts = ranges.len();
            let (tx, mut rx) =
                mpsc::channel::<(usize, OwnedSemaphorePermit, JoinHandle<ObjectResult<Bytes>>)>(
                    ranges.len(),
                );
            let semaphore = self.semaphore.clone();
            let reader_clone = reader.clone();
            let ns_per_byte = self.ns_per_byte.clone();
            let driver = tokio::spawn(async move {
                for (idx, range) in ranges.into_iter().enumerate() {
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|e| ObjectError::ReadError(e.to_string()))?;

                    let reader = reader_clone.clone();
                    let num_bytes = range.end - range.start;
                    let timeout = Self::compute_timeout(num_bytes, ns_per_byte);
                    let range_clone = range.clone();
                    let get_handle =
                        tokio::spawn(async move { reader.get_range(range_clone, timeout).await });

                    if tx.send((idx, permit, get_handle)).await.is_err() {
                        return Err(ObjectError::ReadError("Receiver dropped".to_string()));
                    }
                }
                Ok(())
            });

            let mut multipart = storage
                .put_multipart(&object_path.path())
                .await
                .map_err(ObjectError::ObjectStoreError)?;

            let mut buffer: HashMap<
                usize,
                (OwnedSemaphorePermit, JoinHandle<ObjectResult<Bytes>>),
            > = HashMap::new();
            let mut next_idx = 0;
            let mut put_join_set = JoinSet::new();

            while let Some((idx, permit, get_handle)) = rx.recv().await {
                buffer.insert(idx, (permit, get_handle));

                while let Some((permit, get_handle)) = buffer.remove(&next_idx) {
                    let bytes = get_handle
                        .await
                        .map_err(|e| ObjectError::ReadError(e.to_string()))?
                        .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

                    hasher.update(&bytes);
                    total_downloaded += bytes.len() as u64;
                    let put_fut = multipart.put_part(PutPayload::from_bytes(bytes));
                    put_join_set.spawn(async move {
                        let result = put_fut.await.map_err(ObjectError::ObjectStoreError);
                        drop(permit);
                        result
                    });

                    next_idx += 1;
                }
            }

            if next_idx != num_parts {
                let _ = multipart.abort().await;
                return Err(ObjectError::ReadError(
                    "Missing parts at end of transfer".into(),
                ));
            }

            let computed_checksum = Checksum::new_from_hash(hasher.finalize().into());
            println!("{}", total_downloaded);
            println!("{}", metadata.size());
            println!("{}", computed_checksum);
            println!("{}", metadata.checksum());
            if computed_checksum != metadata.checksum() || total_downloaded != metadata.size() {
                let _ = multipart.abort().await;
                return Err(ObjectError::VerificationError(
                    "verification failed".to_string(),
                ));
            }

            while let Some(res) = put_join_set.join_next().await {
                match res {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => {
                        let _ = multipart.abort().await;
                        return Err(e);
                    }
                    Err(join_err) => {
                        let _ = multipart.abort().await;
                        return Err(ObjectError::ReadError(join_err.to_string()));
                    }
                }
            }

            multipart
                .complete()
                .await
                .map_err(ObjectError::ObjectStoreError)?;

            driver
                .await
                .map_err(|e| ObjectError::ReadError(e.to_string()))??;
        }

        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use crate::readers::store::ObjectStoreReader;

    use super::*;
    use fastcrypto::hash::HashFunction;
    use object_store::memory::InMemory;
    use std::sync::Arc;
    use types::{
        checksum::Checksum,
        crypto::DefaultHash,
        metadata::{Metadata, MetadataV1, ObjectPath},
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
        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len() as u64));

        let object_path = ObjectPath::Probes(0, checksum);

        // Source store (with data)
        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store
            .put(&object_path.path(), data.clone().into())
            .await
            .unwrap();

        // Destination store (empty)
        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Reader from source
        let reader = Arc::new(ObjectStoreReader::new(
            source_store.clone(),
            object_path.clone(),
        ));

        // Downloader
        let downloader = ObjectDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Perform download
        downloader
            .download(
                reader,
                dest_store.clone(),
                object_path.clone(),
                metadata.clone(),
            )
            .await
            .unwrap();

        // Verify object exists in destination
        let result = dest_store.get(&object_path.path()).await.unwrap();
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

        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len() as u64));
        let object_path = ObjectPath::Probes(1, checksum);

        // Source store
        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store
            .put(&object_path.path(), data.clone().into())
            .await
            .unwrap();

        // Destination store
        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Reader
        let reader = Arc::new(ObjectStoreReader::new(
            source_store.clone(),
            object_path.clone(),
        ));

        // Downloader
        let downloader = ObjectDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Download
        downloader
            .download(
                reader,
                dest_store.clone(),
                object_path.clone(),
                metadata.clone(),
            )
            .await
            .unwrap();

        // Verify
        let result = dest_store.get(&object_path.path()).await.unwrap();
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
        let metadata = Metadata::V1(MetadataV1::new(checksum, data.len() as u64));
        let object_path = ObjectPath::Probes(2, checksum);

        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store
            .put(&object_path.path(), data.clone().into())
            .await
            .unwrap();

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        // Pre-populate destination so download should skip
        dest_store
            .put(&object_path.path(), data.clone().into())
            .await
            .unwrap();

        let reader = Arc::new(ObjectStoreReader::new(source_store, object_path.clone()));
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = ObjectDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        // Should succeed and skip download
        downloader
            .download(reader, dest_store.clone(), object_path.clone(), metadata)
            .await
            .unwrap();

        // Data should still be correct
        let result = dest_store.get(&object_path.path()).await.unwrap();
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
        let metadata = Metadata::V1(MetadataV1::new(wrong_checksum, data.len() as u64));
        let object_path = ObjectPath::Probes(3, wrong_checksum);

        let source_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        source_store
            .put(&object_path.path(), data.into())
            .await
            .unwrap();

        let dest_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let reader = Arc::new(ObjectStoreReader::new(source_store, object_path.clone()));
        let concurrency = Arc::new(Semaphore::new(2));
        let downloader = ObjectDownloader::new(concurrency, chunk_size, ns_per_byte).unwrap();

        let err = downloader
            .download(reader, dest_store.clone(), object_path, metadata)
            .await
            .unwrap_err();

        assert!(matches!(err, ObjectError::VerificationError(_)));
    }
}
