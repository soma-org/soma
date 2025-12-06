use super::container::Container;
use anyhow::Result;
use encoder::core::encoder_node::EncoderNodeHandle;
use object_store::memory::InMemory;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};
use tracing::info;
use types::config::encoder_config::EncoderConfig;
use types::shard_crypto::keys::EncoderPublicKey;

/// A handle to an in-memory Soma Encoder.
///
/// Each Encoder is attempted to run in isolation from each other by running them in their own tokio
/// runtime in a separate thread. By doing this we can ensure that all asynchronous tasks
/// associated with a Encoder are able to be stopped when desired (either when a Encoder is dropped or
/// explicitly stopped by calling encoder stop) by simply dropping that Encoder's runtime.
#[derive(Debug)]
pub struct Node {
    container: Mutex<Option<Container>>,
    config: Mutex<EncoderConfig>,
    working_dir: PathBuf,
    shared_object_store: Option<Arc<InMemory>>,
}

impl Node {
    pub fn new(
        config: EncoderConfig,
        working_dir: PathBuf,
        shared_object_store: Option<Arc<InMemory>>,
    ) -> Self {
        Self {
            container: Default::default(),
            config: config.into(),
            working_dir,
            shared_object_store,
        }
    }

    /// Return the `name` of this Node
    pub fn name(&self) -> EncoderPublicKey {
        self.config().protocol_public_key()
    }

    pub fn config(&self) -> MutexGuard<'_, EncoderConfig> {
        self.config.lock().unwrap()
    }

    /// Start this Node
    pub async fn spawn(&self) -> Result<()> {
        info!("starting in-memory node {:?}", self.name());
        let config = self.config().clone();
        *self.container.lock().unwrap() = Some(
            Container::spawn(
                config,
                self.working_dir.clone(),
                self.shared_object_store.clone(),
            )
            .await,
        );
        Ok(())
    }

    /// Start this Node, waiting until its completely started up.
    pub async fn start(&self) -> Result<()> {
        self.spawn().await
    }

    /// Stop this Node
    pub fn stop(&self) {
        info!("stopping in-memory node {:?}", self.name());
        *self.container.lock().unwrap() = None;
        info!("node stopped {:?}", self.name());
    }

    /// If this Node is currently running
    pub fn is_running(&self) -> bool {
        self.container
            .lock()
            .unwrap()
            .as_ref()
            .map_or(false, |c| c.is_alive())
    }

    pub fn get_node_handle(&self) -> Option<EncoderNodeHandle> {
        self.container
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|c| c.get_node_handle())
    }
}
