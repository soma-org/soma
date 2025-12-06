use crate::swarm_node::Node;
use anyhow::Result;
use encoder::core::encoder_node::EncoderNodeHandle;
use futures::future::try_join_all;
use object_store::memory::InMemory;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;
use tracing::info;
use types::shard_crypto::keys::EncoderPublicKey;
use types::{config::encoder_config::EncoderConfig, crypto::NetworkPublicKey};

#[derive(Debug)]
enum SwarmDirectory {
    Persistent(PathBuf),
    Temporary(TempDir),
}

impl SwarmDirectory {
    fn new_temporary() -> Self {
        SwarmDirectory::Temporary(TempDir::new().unwrap())
    }

    fn encoder_dir(&self, encoder_index: usize) -> PathBuf {
        self.join(format!("encoder_{}", encoder_index))
    }
}

impl std::ops::Deref for SwarmDirectory {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        match self {
            SwarmDirectory::Persistent(dir) => dir.deref(),
            SwarmDirectory::Temporary(dir) => dir.path(),
        }
    }
}

impl AsRef<Path> for SwarmDirectory {
    fn as_ref(&self) -> &Path {
        match self {
            SwarmDirectory::Persistent(dir) => dir.as_ref(),
            SwarmDirectory::Temporary(dir) => dir.as_ref(),
        }
    }
}

#[derive(Debug)]
pub struct EncoderSwarm {
    dir: SwarmDirectory,
    nodes: HashMap<EncoderPublicKey, Node>,
    shared_object_store: Option<Arc<InMemory>>,
}

impl Drop for EncoderSwarm {
    fn drop(&mut self) {
        self.nodes_iter_mut().for_each(|node| node.stop());
    }
}

impl EncoderSwarm {
    fn nodes_iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.nodes.values_mut()
    }

    /// Return a new Builder
    pub fn builder() -> EncoderSwarmBuilder {
        EncoderSwarmBuilder::new()
    }

    pub fn dir(&self) -> &Path {
        self.dir.as_ref()
    }

    /// Start all nodes associated with this Swarm
    pub async fn launch(&mut self) -> Result<()> {
        try_join_all(self.nodes_iter_mut().map(|node| node.start())).await?;
        tracing::info!("Successfully launched Encoder Swarm");
        Ok(())
    }

    // Doesn't mean all the encoders are active
    pub fn encoder_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node(&self, name: &EncoderPublicKey) -> Option<&Node> {
        self.nodes.get(name)
    }

    pub fn encoder_node_handles(&self) -> Vec<EncoderNodeHandle> {
        self.encoder_nodes()
            .map(|node| node.get_node_handle().unwrap())
            .collect()
    }

    /// Returns an iterator over all currently active encoders.
    pub fn active_encoders(&self) -> impl Iterator<Item = &Node> {
        self.encoder_nodes().filter(|node| {
            node.get_node_handle().map_or(false, |handle| {
                // TODO: define what an active encoder is by letting EncoderNode's state be inspectable
                true
                // let state = handle.state();
                // state.is_active()
            })
        })
    }

    pub async fn spawn_new_encoder(&mut self, config: EncoderConfig) -> EncoderNodeHandle {
        let name = config.protocol_public_key();
        let working_dir = self.dir.encoder_dir(self.nodes.len());
        let node = Node::new(config, working_dir, self.shared_object_store.clone());
        node.start().await.unwrap();
        let handle = node.get_node_handle().unwrap();
        self.nodes.insert(name, node);
        handle
    }
}

pub struct EncoderSwarmBuilder {
    encoders: Option<Vec<EncoderConfig>>,
    client_key: Option<NetworkPublicKey>,
    dir: Option<PathBuf>, // Add this field
    shared_object_store: Option<Arc<InMemory>>,
}

impl EncoderSwarmBuilder {
    pub fn new() -> Self {
        Self {
            encoders: None,
            client_key: None,
            dir: None,
            shared_object_store: None,
        }
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.encoders = Some(encoders);
        self
    }

    pub fn with_client_key(mut self, client_key: NetworkPublicKey) -> Self {
        self.client_key = Some(client_key);
        self
    }

    pub fn with_shared_object_store(mut self, store: Arc<InMemory>) -> Self {
        self.shared_object_store = Some(store);
        self
    }

    pub fn dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.dir = Some(dir.into());
        self
    }

    pub fn build(self) -> EncoderSwarm {
        let dir = if let Some(dir) = self.dir {
            SwarmDirectory::Persistent(dir)
        } else {
            SwarmDirectory::new_temporary()
        };

        // We require configs to be provided
        let encoder_configs = self.encoders.expect("Encoder configs must be provided");
        let shared_store = self.shared_object_store;

        // Create nodes from the configs
        let nodes = encoder_configs
            .iter()
            .enumerate()
            .map(|(idx, config)| {
                let node_working_dir = dir.encoder_dir(idx);
                info!(
                    "Configuring encoder with name {:?} and dir {:?}",
                    config.protocol_public_key(),
                    node_working_dir
                );

                (
                    config.protocol_public_key(),
                    Node::new(config.clone(), node_working_dir, shared_store.clone()),
                )
            })
            .collect();

        EncoderSwarm {
            dir,
            nodes,
            shared_object_store: shared_store,
        }
    }
}
