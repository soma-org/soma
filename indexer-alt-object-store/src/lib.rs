// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use async_trait::async_trait;
use bytes::Bytes;
use indexer_store_traits::CommitterWatermark;
use indexer_store_traits::ConcurrentConnection;
use indexer_store_traits::ConcurrentStore;
use indexer_store_traits::Connection;
use indexer_store_traits::InitWatermark;
use indexer_store_traits::PrunerWatermark;
use indexer_store_traits::ReaderWatermark;
use indexer_store_traits::Store;
use object_store::Error as ObjectStoreError;
use object_store::ObjectStore as _;
use object_store::PutMode;
use object_store::PutPayload;
use object_store::path::Path as ObjectPath;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone)]
pub struct ObjectStore {
    object_store: Arc<dyn object_store::ObjectStore>,
}

pub struct ObjectStoreConnection {
    object_store: Arc<dyn object_store::ObjectStore>,
}

/// Used to potentially migrate from the legacy watermark format that does not include `reader_lo`,
/// `pruner_hi`, and `pruner_timestamp_ms`.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LegacyObjectStoreWatermark {
    epoch_hi_inclusive: u64,
    checkpoint_hi_inclusive: Option<u64>,
    tx_hi: u64,
    timestamp_ms_hi_inclusive: u64,
    #[serde(default)]
    reader_lo: Option<u64>,
    #[serde(default)]
    pruner_hi: Option<u64>,
    #[serde(default)]
    pruner_timestamp_ms: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ObjectStoreWatermark {
    epoch_hi_inclusive: u64,
    checkpoint_hi_inclusive: Option<u64>,
    tx_hi: u64,
    timestamp_ms_hi_inclusive: u64,
    reader_lo: u64,
    pruner_hi: u64,
    pruner_timestamp_ms: u64,
}

impl ObjectStore {
    pub fn new(object_store: Arc<dyn object_store::ObjectStore>) -> Self {
        Self { object_store }
    }
}

impl ObjectStoreConnection {
    pub fn object_store(&self) -> Arc<dyn object_store::ObjectStore> {
        self.object_store.clone()
    }

    async fn get_watermark_for_read(
        &self,
        pipeline: &str,
    ) -> anyhow::Result<Option<(ObjectStoreWatermark, u64)>> {
        let object_path = watermark_path(pipeline);
        let result = match self.object_store.get(&object_path).await {
            Ok(result) => result,
            Err(ObjectStoreError::NotFound { .. }) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let bytes = result.bytes().await?;
        let watermark =
            serde_json::from_slice::<ObjectStoreWatermark>(&bytes).with_context(|| {
                format!("Failed to parse watermark from object store pipeline={pipeline}")
            })?;
        // Hide watermarks where `checkpoint_hi_inclusive < reader_lo`.
        let Some(checkpoint_hi_inclusive) =
            watermark.checkpoint_hi_inclusive.filter(|&cp| watermark.reader_lo <= cp)
        else {
            return Ok(None);
        };

        Ok(Some((watermark, checkpoint_hi_inclusive)))
    }

    async fn get_watermark_for_write(
        &self,
        pipeline: &str,
    ) -> anyhow::Result<(ObjectStoreWatermark, Option<String>, Option<String>)> {
        let object_path = watermark_path(pipeline);
        let result = match self.object_store.get(&object_path).await {
            Ok(result) => result,
            Err(e) => return Err(e.into()),
        };

        let e_tag = result.meta.e_tag.clone();
        let version = result.meta.version.clone();
        let bytes = result.bytes().await?;
        let watermark = serde_json::from_slice::<ObjectStoreWatermark>(&bytes)
            .context("Failed to parse watermark from object store")?;

        Ok((watermark, e_tag, version))
    }

    async fn set_watermark(
        &self,
        pipeline: &str,
        watermark: ObjectStoreWatermark,
        e_tag: Option<String>,
        version: Option<String>,
    ) -> anyhow::Result<()> {
        let object_path = watermark_path(pipeline);
        let json_bytes = serde_json::to_vec(&watermark)?;
        let payload: PutPayload = Bytes::from(json_bytes).into();
        self.object_store
            .put_opts(
                &object_path,
                payload,
                PutMode::Update(object_store::UpdateVersion { e_tag, version }).into(),
            )
            .await?;
        Ok(())
    }
}

#[async_trait]
impl ConcurrentStore for ObjectStore {
    type ConcurrentConnection<'c> = ObjectStoreConnection;
}

#[async_trait]
impl Store for ObjectStore {
    type Connection<'c> = ObjectStoreConnection;

    async fn connect<'c>(&'c self) -> anyhow::Result<Self::Connection<'c>> {
        Ok(ObjectStoreConnection { object_store: self.object_store.clone() })
    }
}

#[async_trait]
impl Connection for ObjectStoreConnection {
    async fn init_watermark(
        &mut self,
        pipeline_task: &str,
        checkpoint_hi_inclusive: Option<u64>,
    ) -> anyhow::Result<Option<InitWatermark>> {
        let object_path = watermark_path(pipeline_task);
        let reader_lo = checkpoint_hi_inclusive.map_or(0, |cp| cp + 1);
        let watermark = ObjectStoreWatermark {
            epoch_hi_inclusive: 0,
            checkpoint_hi_inclusive,
            tx_hi: 0,
            timestamp_ms_hi_inclusive: 0,
            reader_lo,
            pruner_hi: reader_lo,
            pruner_timestamp_ms: 0,
        };
        let json_bytes = serde_json::to_vec(&watermark)?;
        let payload: PutPayload = Bytes::from(json_bytes).into();
        // Try create-if-not-exists write first.
        let (checkpoint_hi_inclusive, reader_lo) = match self
            .object_store
            .put_opts(&object_path, payload, PutMode::Create.into())
            .await
        {
            Ok(_) => (checkpoint_hi_inclusive, Some(reader_lo)),
            Err(object_store::Error::AlreadyExists { .. }) => {
                // Fall back to reading existing watermark.
                let result = match self.object_store.get(&object_path).await {
                    Ok(result) => result,
                    Err(e) => return Err(e.into()),
                };
                let e_tag = result.meta.e_tag.clone();
                let version = result.meta.version.clone();
                let bytes = result.bytes().await?;
                let legacy_watermark: LegacyObjectStoreWatermark = serde_json::from_slice(&bytes)
                    .with_context(|| {
                        format!(
                            "Failed to parse legacy watermark from object store pipeline={pipeline_task}"
                        )
                    })?;

                // Write data from the legacy watermark using the new format if it is missing newly added fields.
                if legacy_watermark.reader_lo.is_none()
                    || legacy_watermark.pruner_hi.is_none()
                    || legacy_watermark.pruner_timestamp_ms.is_none()
                {
                    let watermark = ObjectStoreWatermark {
                        epoch_hi_inclusive: legacy_watermark.epoch_hi_inclusive,
                        checkpoint_hi_inclusive: legacy_watermark.checkpoint_hi_inclusive,
                        tx_hi: legacy_watermark.tx_hi,
                        timestamp_ms_hi_inclusive: legacy_watermark.timestamp_ms_hi_inclusive,
                        ..watermark
                    };
                    self.set_watermark(pipeline_task, watermark, e_tag, version).await?;
                }

                (legacy_watermark.checkpoint_hi_inclusive, legacy_watermark.reader_lo)
            }
            Err(e) => return Err(e.into()),
        };
        Ok(Some(InitWatermark { checkpoint_hi_inclusive, reader_lo }))
    }

    async fn accepts_chain_id(
        &mut self,
        pipeline_task: &str,
        chain_id: [u8; 32],
    ) -> anyhow::Result<bool> {
        crate::accepts_chain_id(self.object_store.as_ref(), pipeline_task, chain_id).await
    }

    async fn committer_watermark(
        &mut self,
        pipeline_task: &str,
    ) -> anyhow::Result<Option<CommitterWatermark>> {
        Ok(self.get_watermark_for_read(pipeline_task).await?.map(|(w, checkpoint_hi_inclusive)| {
            CommitterWatermark {
                epoch_hi_inclusive: w.epoch_hi_inclusive,
                checkpoint_hi_inclusive,
                tx_hi: w.tx_hi,
                timestamp_ms_hi_inclusive: w.timestamp_ms_hi_inclusive,
            }
        }))
    }

    async fn set_committer_watermark(
        &mut self,
        pipeline_task: &str,
        watermark: CommitterWatermark,
    ) -> anyhow::Result<bool> {
        let (current_watermark, e_tag, version) =
            self.get_watermark_for_write(pipeline_task).await?;

        if current_watermark
            .checkpoint_hi_inclusive
            .is_some_and(|cp| cp >= watermark.checkpoint_hi_inclusive)
        {
            return Ok(false);
        }

        let new_watermark = ObjectStoreWatermark {
            epoch_hi_inclusive: watermark.epoch_hi_inclusive,
            checkpoint_hi_inclusive: Some(watermark.checkpoint_hi_inclusive),
            tx_hi: watermark.tx_hi,
            timestamp_ms_hi_inclusive: watermark.timestamp_ms_hi_inclusive,
            ..current_watermark
        };
        self.set_watermark(pipeline_task, new_watermark, e_tag, version).await?;
        Ok(true)
    }
}

#[async_trait]
impl ConcurrentConnection for ObjectStoreConnection {
    async fn reader_watermark(
        &mut self,
        pipeline: &str,
    ) -> anyhow::Result<Option<ReaderWatermark>> {
        Ok(self.get_watermark_for_read(pipeline).await?.map(|(w, checkpoint_hi_inclusive)| {
            ReaderWatermark { checkpoint_hi_inclusive, reader_lo: w.reader_lo }
        }))
    }

    async fn pruner_watermark(
        &mut self,
        pipeline: &'static str,
        delay: Duration,
    ) -> anyhow::Result<Option<PrunerWatermark>> {
        let Some((watermark, _)) = self.get_watermark_for_read(pipeline).await? else {
            return Ok(None);
        };
        let pruner_ready_ms = (watermark.pruner_timestamp_ms as u128) + delay.as_millis();
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let wait_for_ms = i64::try_from(pruner_ready_ms.saturating_sub(now_ms))?;
        Ok(Some(PrunerWatermark {
            wait_for_ms,
            reader_lo: watermark.reader_lo,
            pruner_hi: watermark.pruner_hi,
        }))
    }

    async fn set_reader_watermark(
        &mut self,
        pipeline: &'static str,
        reader_lo: u64,
    ) -> anyhow::Result<bool> {
        let (current_watermark, e_tag, version) = self.get_watermark_for_write(pipeline).await?;

        if reader_lo <= current_watermark.reader_lo {
            return Ok(false);
        }

        let new_watermark = ObjectStoreWatermark { reader_lo, ..current_watermark };
        self.set_watermark(pipeline, new_watermark, e_tag, version).await?;
        Ok(true)
    }

    async fn set_pruner_watermark(
        &mut self,
        pipeline: &'static str,
        pruner_hi: u64,
    ) -> anyhow::Result<bool> {
        let (current_watermark, e_tag, version) = self.get_watermark_for_write(pipeline).await?;

        if pruner_hi <= current_watermark.pruner_hi {
            return Ok(false);
        }

        let new_watermark = ObjectStoreWatermark { pruner_hi, ..current_watermark };
        self.set_watermark(pipeline, new_watermark, e_tag, version).await?;
        Ok(true)
    }
}

fn watermark_path(pipeline: &str) -> ObjectPath {
    ObjectPath::from(format!("_metadata/watermarks/{}.json", pipeline))
}

fn chain_id_path(pipeline_task: &str) -> ObjectPath {
    ObjectPath::from(format!("_metadata/chain_id/{pipeline_task}"))
}

pub async fn accepts_chain_id(
    object_store: &dyn object_store::ObjectStore,
    pipeline_task: &str,
    chain_id: [u8; 32],
) -> anyhow::Result<bool> {
    let path = chain_id_path(pipeline_task);
    match object_store
        .put_opts(
            &path,
            chain_id.to_vec().into(),
            object_store::PutOptions { mode: PutMode::Create, ..Default::default() },
        )
        .await
    {
        Ok(_) => Ok(true),
        Err(ObjectStoreError::AlreadyExists { .. }) => {
            let bytes = object_store.get(&path).await?.bytes().await?;
            let stored: [u8; 32] = bytes.as_ref().try_into().ok().with_context(|| {
                format!("stored chain_id at {} has wrong length: {}", path, bytes.len())
            })?;
            Ok(stored == chain_id)
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use indexer_store_traits::concurrent_connection_tests;
    use indexer_store_traits::connection_tests;
    use indexer_store_traits::testing::Harness;
    use object_store::memory::InMemory;

    use super::*;

    struct ObjectStoreHarness {
        store: ObjectStore,
    }

    #[async_trait::async_trait(?Send)]
    impl Harness for ObjectStoreHarness {
        type Store = ObjectStore;

        async fn new() -> Self {
            Self { store: ObjectStore::new(Arc::new(InMemory::new())) }
        }

        fn store(&self) -> &Self::Store {
            &self.store
        }
    }

    connection_tests!(ObjectStoreHarness);
    concurrent_connection_tests!(ObjectStoreHarness);
}
