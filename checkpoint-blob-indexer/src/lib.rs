// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod handlers;

pub use config::CommitterLayer;
pub use config::ConcurrentLayer;
pub use config::IndexerConfig;
pub use config::IngestionConfig;
pub use config::PipelineLayer;
pub use handlers::CheckpointBlob;
pub use handlers::CheckpointBlobPipeline;
pub use handlers::EpochCheckpoint;
pub use handlers::EpochsPipeline;

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use indexer_alt_object_store::ObjectStore;
    use indexer_framework::pipeline::concurrent::Handler;
    use indexer_store_traits::Store;
    use object_store::ObjectStore as _;
    use object_store::memory::InMemory;
    use object_store::path::Path as ObjectPath;
    use std::sync::Arc;

    /// An uncompressed blob written by `commit` must read back byte-for-byte. `object_store`'s
    /// `put` is a single atomic upload — the object only becomes visible once the whole upload
    /// completes — so there is no window where a reader can observe a partially-written blob.
    #[tokio::test]
    async fn test_checkpoint_blob_handler_uncompressed() {
        let store = ObjectStore::new(Arc::new(InMemory::new()));
        let mut conn = store.connect().await.unwrap();

        let blob =
            CheckpointBlob { sequence_number: 100, proto_bytes: Bytes::from(vec![1, 2, 3, 4, 5]) };

        let pipeline = CheckpointBlobPipeline { compression_level: None };
        let count = pipeline.commit(&Some(blob), &mut conn).await.unwrap();
        assert_eq!(count, 1);

        let result = conn.object_store().get(&ObjectPath::from("100.binpb")).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), &[1, 2, 3, 4, 5]);
    }

    /// A compressed blob must read back and decompress to the exact original. zstd
    /// decompression fails on a truncated stream, so a successful `decode_all` that equals the
    /// input proves the stored `.binpb.zst` object is complete — i.e. the blob is never
    /// "uploaded before finished writing".
    #[tokio::test]
    async fn test_checkpoint_blob_handler_compressed() {
        let store = ObjectStore::new(Arc::new(InMemory::new()));
        let mut conn = store.connect().await.unwrap();

        let test_data = vec![7u8; 4096];
        let blob =
            CheckpointBlob { sequence_number: 200, proto_bytes: Bytes::from(test_data.clone()) };

        let pipeline = CheckpointBlobPipeline { compression_level: Some(3) };
        let count = pipeline.commit(&Some(blob), &mut conn).await.unwrap();
        assert_eq!(count, 1);

        let result = conn.object_store().get(&ObjectPath::from("200.binpb.zst")).await.unwrap();
        let compressed = result.bytes().await.unwrap();

        let decompressed = zstd::decode_all(&compressed[..]).unwrap();
        assert_eq!(decompressed, test_data, "stored blob must decompress to the original");
        assert!(compressed.len() < test_data.len(), "blob should actually be compressed");
    }

    /// The epochs.json index appends in sorted order and is idempotent — a duplicate commit
    /// neither corrupts the file nor double-inserts.
    #[tokio::test]
    async fn test_epochs_handler() {
        let store = ObjectStore::new(Arc::new(InMemory::new()));

        let commit = |checkpoint_number: u64| {
            let store = store.clone();
            async move {
                let mut conn = store.connect().await.unwrap();
                EpochsPipeline
                    .commit(&Some(EpochCheckpoint { checkpoint_number }), &mut conn)
                    .await
                    .unwrap()
            }
        };
        let read = || {
            let store = store.clone();
            async move {
                let conn = store.connect().await.unwrap();
                let bytes = conn
                    .object_store()
                    .get(&ObjectPath::from("epochs.json"))
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap();
                serde_json::from_slice::<Vec<u64>>(&bytes).unwrap()
            }
        };

        assert_eq!(commit(100).await, 1);
        assert_eq!(read().await, vec![100]);

        assert_eq!(commit(200).await, 1);
        assert_eq!(commit(150).await, 1);
        assert_eq!(read().await, vec![100, 150, 200], "epochs stay sorted");

        // Duplicate commit is a no-op.
        assert_eq!(commit(100).await, 0);
        assert_eq!(read().await, vec![100, 150, 200]);
    }
}
