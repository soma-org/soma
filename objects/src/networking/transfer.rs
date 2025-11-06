use std::sync::Arc;

use bytes::BytesMut;
use futures::StreamExt;
use object_store::{ObjectStore, PutPayload};
use types::{
    error::{ObjectError, ObjectResult},
    metadata::ObjectPath,
};

use crate::networking::MIN_PART_SIZE;

pub struct Transfer {}

impl Transfer {
    pub async fn transfer(
        src: Arc<dyn ObjectStore>,
        dst: Arc<dyn ObjectStore>,
        object_path: &ObjectPath,
    ) -> ObjectResult<()> {
        // Check existence and get metadata (HEAD is cheap).
        let meta = src
            .head(&object_path.path())
            .await
            .map_err(ObjectError::ObjectStoreError)?;

        // check if dst already has the object
        if dst.head(&object_path.path()).await.is_ok() {
            return Ok(());
        }

        if meta.size == 0 {
            // Empty object: put empty payload.
            dst.put(&object_path.path(), PutPayload::default())
                .await
                .map_err(ObjectError::ObjectStoreError)?;
            return Ok(());
        }

        if meta.size <= MIN_PART_SIZE as u64 {
            // Small: Collect full bytes and direct put (memory-efficient for small).
            let bytes = src
                .get(&object_path.path())
                .await
                .map_err(ObjectError::ObjectStoreError)?
                .bytes()
                .await
                .map_err(ObjectError::ObjectStoreError)?;
            dst.put(&object_path.path(), bytes.into())
                .await
                .map_err(ObjectError::ObjectStoreError)?;
        } else {
            // Large: Multipart stream transfer.
            let mut multipart = dst
                .put_multipart(&object_path.path())
                .await
                .map_err(ObjectError::ObjectStoreError)?;

            let mut stream = src
                .get(&object_path.path())
                .await
                .map_err(ObjectError::ObjectStoreError)?
                .into_stream();

            let mut buffer = BytesMut::new();

            loop {
                // Fill buffer up to PART_SIZE from stream.
                while (buffer.len() as u64) < MIN_PART_SIZE {
                    if let Some(chunk_res) = stream.next().await {
                        let chunk = chunk_res.map_err(ObjectError::ObjectStoreError)?;
                        buffer.extend_from_slice(&chunk);
                    } else {
                        // End of stream.
                        break;
                    }
                }

                if buffer.is_empty() {
                    break;
                }

                // Upload current buffer as part (may be < PART_SIZE for last).
                let part_bytes = buffer
                    .split_to(std::cmp::min(MIN_PART_SIZE as usize, buffer.len()))
                    .freeze();
                let _upload_part = multipart
                    .put_part(PutPayload::from(part_bytes))
                    .await
                    .map_err(ObjectError::ObjectStoreError)?;
            }

            // Complete the multipart upload (impl handles part ordering/etags internally).
            let _put_result = multipart
                .complete()
                .await
                .map_err(ObjectError::ObjectStoreError)?;
        }

        Ok(())
    }
}
