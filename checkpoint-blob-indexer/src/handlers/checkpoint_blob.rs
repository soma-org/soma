// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::LazyLock;

use bytes::Bytes;
use indexer_alt_object_store::ObjectStore;
use indexer_framework::pipeline::Processor;
use indexer_framework::pipeline::concurrent::BatchStatus;
use indexer_framework::pipeline::concurrent::Handler;
use indexer_store_traits::Store;
use object_store::ObjectStore as _;
use object_store::path::Path as ObjectPath;
use prost::Message;
use rpc::proto::soma::Checkpoint as ProtoCheckpoint;
use rpc::utils::field::FieldMaskTree;
use rpc::utils::merge::Merge;
use types::full_checkpoint_content::Checkpoint;

pub struct CheckpointBlob {
    pub sequence_number: u64,
    pub proto_bytes: Bytes,
}

pub struct CheckpointBlobPipeline {
    pub compression_level: Option<i32>,
}

#[async_trait::async_trait]
impl Processor for CheckpointBlobPipeline {
    const NAME: &'static str = "checkpoint_blob";
    type Value = CheckpointBlob;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        // Soma's proto layer doesn't yet expose Sui's path_builder field-mask API, so use the
        // wildcard mask: serialize every available field on the Checkpoint. The resulting blob is
        // larger than Sui's selective-mask output but the .binpb.zst format remains identical.
        static MASK: LazyLock<FieldMaskTree> = LazyLock::new(FieldMaskTree::new_wildcard);

        let sequence_number = checkpoint.summary.sequence_number;
        let proto_checkpoint = ProtoCheckpoint::merge_from(checkpoint.as_ref(), &MASK);
        let proto_bytes = Bytes::from(proto_checkpoint.encode_to_vec());

        Ok(vec![CheckpointBlob { sequence_number, proto_bytes }])
    }
}

#[async_trait::async_trait]
impl Handler for CheckpointBlobPipeline {
    type Store = ObjectStore;
    type Batch = Option<Self::Value>;

    fn batch(
        &self,
        batch: &mut Self::Batch,
        values: &mut std::vec::IntoIter<Self::Value>,
    ) -> BatchStatus {
        if batch.is_none() && values.len() > 0 {
            *batch = values.next();
            BatchStatus::Ready
        } else {
            BatchStatus::Pending
        }
    }

    async fn commit<'a>(
        &self,
        batch: &Self::Batch,
        conn: &mut <Self::Store as Store>::Connection<'a>,
    ) -> anyhow::Result<usize> {
        let Some(blob) = batch else {
            return Ok(0);
        };

        let mut path = format!("{}.binpb", blob.sequence_number);
        let data: Bytes = if let Some(level) = self.compression_level {
            path = format!("{}.zst", path);
            let bytes = blob.proto_bytes.clone();
            tokio::task::spawn_blocking(move || {
                Ok::<Bytes, std::io::Error>(Bytes::from(zstd::encode_all(&bytes[..], level)?))
            })
            .await??
        } else {
            blob.proto_bytes.clone()
        };

        conn.object_store().put(&ObjectPath::from(path), data.into()).await?;
        Ok(1)
    }
}
