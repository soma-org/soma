use bytes::Bytes;
use fastcrypto::hash::HashFunction;
use object_store::{MultipartUpload, ObjectStore, PutPayload};
use reqwest::{header, Client, StatusCode};
use std::{sync::Arc, time::Duration};
use types::{
    checksum::Checksum,
    crypto::DefaultHash as DefaultHashFunction,
    error::ObjectError,
    metadata::{Metadata, MetadataAPI, ObjectPath},
};
use url::Url;

use types::error::ObjectResult;

use crate::networking::MIN_PART_SIZE;

pub struct Download {
    client: Client,
    url: Url,
    storage: Arc<dyn ObjectStore>,
    storage_path: ObjectPath,
    metadata: Metadata,
    ns_per_byte: u64,
    max_size: u64,
    backoff: Duration,
    max_retries: usize,
    downloaded_bytes: u64,
    hasher: DefaultHashFunction,
    multipart_upload: Option<Box<dyn MultipartUpload>>,
    buffer: Vec<u8>,
}

impl Download {
    pub async fn new(
        client: Client,
        url: Url,
        storage: Arc<dyn ObjectStore>,
        storage_path: ObjectPath,
        metadata: Metadata,
        ns_per_byte: u64,
        max_size: u64,
    ) -> ObjectResult<()> {
        if storage_path.checksum() != metadata.checksum() {
            return Err(ObjectError::VerificationError(
                "Checksum mismatch".to_string(),
            ));
        }
        if metadata.size() > max_size {
            return Err(ObjectError::VerificationError("File too large".to_string()));
        }
        if storage.head(&storage_path.path()).await.is_ok() {
            // if exists, skip early
            return Ok(());
        }

        let download = Self {
            client,
            url,
            storage,
            storage_path,
            metadata,
            ns_per_byte,
            max_size,
            backoff: Duration::from_secs(1),
            max_retries: 3,
            downloaded_bytes: 0,
            hasher: DefaultHashFunction::new(),
            multipart_upload: None,
            buffer: Vec::new(),
        };

        download.retry_loop().await
    }

    async fn retry_loop(mut self) -> ObjectResult<()> {
        let mut success: Option<()> = None;
        for attempt in 0..=self.max_retries {
            match self.download_and_upload().await {
                Ok(()) => {
                    success = Some(());
                    break;
                }
                Err(e) => {
                    self.cleanup_after_error().await;
                    let sleep_duration = self.sleep_duration(attempt);
                    tokio::time::sleep(sleep_duration).await;
                }
            }
        }

        match success {
            Some(_) => {
                self.verify_and_complete().await?;
            }
            None => {
                self.abort().await?;
            }
        }

        // handle abort
        Ok(())
    }

    async fn download_and_upload(&mut self) -> ObjectResult<()> {
        let bytes_to_download = self.metadata.size().saturating_sub(self.downloaded_bytes);
        let timeout = Duration::from_nanos(self.ns_per_byte * bytes_to_download);

        let mut req = self.client.get(self.url.clone()).timeout(timeout);
        if self.downloaded_bytes > 0 {
            req = req.header(header::RANGE, format!("bytes={}-", self.downloaded_bytes));
        }

        let mut response = req
            .send()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?;

        let status = response.status();
        if (self.downloaded_bytes > 0 && status != StatusCode::PARTIAL_CONTENT)
            || (self.downloaded_bytes == 0 && !status.is_success())
            || status.is_server_error()
        {
            return Err(ObjectError::NetworkRequest(format!(
                "Unexpected status: {}",
                status
            )));
        }

        while let Some(chunk_result) = response
            .chunk()
            .await
            .map_err(|e| ObjectError::NetworkRequest(e.to_string()))?
        {
            let chunk = match chunk_result {
                chunk if !chunk.is_empty() => chunk,
                _ => break, // Empty chunk signals end of stream
            };

            self.buffer.extend_from_slice(&chunk);

            // Flush buffer to parts as it grows
            while self.buffer.len() >= MIN_PART_SIZE as usize {
                let part: Vec<u8> = self.buffer.drain(..MIN_PART_SIZE as usize).collect();
                let part = Bytes::from(part);
                self.push_to_storage(part).await?;
            }
        }

        if !self.buffer.is_empty() {
            let part: Vec<u8> = self.buffer.drain(..).collect();
            let part = Bytes::from(part);
            self.push_to_storage(part).await?;
        }

        Ok(())
    }

    async fn push_to_storage(&mut self, part: Bytes) -> ObjectResult<()> {
        self.validate_size_before_upload(&part)?;

        let upload = self.ensure_multipart_upload().await?;
        upload
            .put_part(PutPayload::from_bytes(part.clone()))
            .await
            .map_err(|e| ObjectError::ObjectStorage(e.to_string()))?;

        self.hasher.update(&part);
        self.downloaded_bytes += part.len() as u64;
        Ok(())
    }

    fn validate_size_before_upload(&self, part: &Bytes) -> ObjectResult<()> {
        if self.downloaded_bytes + part.len() as u64 + self.buffer.len() as u64
            > self.metadata.size()
        {
            return Err(ObjectError::VerificationError(
                "Oversized response".to_string(),
            ));
        }
        Ok(())
    }

    async fn ensure_multipart_upload(&mut self) -> ObjectResult<&mut Box<dyn MultipartUpload>> {
        if self.multipart_upload.is_none() {
            self.multipart_upload = Some(
                self.storage
                    .put_multipart(&self.storage_path.path())
                    .await
                    .map_err(|e| ObjectError::ObjectStorage(e.to_string()))?,
            );
        }
        Ok(self.multipart_upload.as_mut().unwrap())
    }

    async fn verify_and_complete(mut self) -> ObjectResult<()> {
        if self.downloaded_bytes != self.metadata.size() {
            return Err(ObjectError::VerificationError(format!(
                "Size mismatch: expected {}, got {}",
                self.metadata.size(),
                self.downloaded_bytes
            )));
        }

        let checksum = Checksum::new_from_hash(self.hasher.finalize().into());
        if checksum != self.metadata.checksum() {
            return Err(ObjectError::VerificationError(format!(
                "Checksum mismatch: expected {:?}, got {:?}",
                self.metadata.checksum(),
                checksum
            )));
        }

        let mut upload = self.multipart_upload.take().ok_or_else(|| {
            ObjectError::ObjectStorage("No multipart upload to complete".to_string())
        })?;
        upload
            .complete()
            .await
            .map_err(|e| ObjectError::ObjectStorage(e.to_string()))?;

        Ok(())
    }

    async fn cleanup_after_error(&mut self) {
        self.buffer.clear();
    }
    fn sleep_duration(&self, attempt: usize) -> Duration {
        self.backoff.saturating_mul(attempt as u32)
    }

    async fn abort(mut self) -> ObjectResult<()> {
        if self.multipart_upload.is_some() {
            let upload = self.multipart_upload.as_mut().unwrap();
            upload
                .abort()
                .await
                .map_err(|e| ObjectError::ObjectStorage(e.to_string()))?;
        }
        Ok(())
    }
}
