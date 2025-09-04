use crate::swarm_node::Node;
use anyhow::Result;
use encoder::core::encoder_node::EncoderNodeHandle;
use futures::future::try_join_all;
use std::collections::HashMap;
use tracing::info;
use types::config::encoder_config::EncoderConfig;
use types::shard_crypto::keys::{EncoderPublicKey, PeerPublicKey};

#[derive(Debug)]
pub struct EncoderSwarm {
    nodes: HashMap<EncoderPublicKey, Node>,
    client_key: Option<PeerPublicKey>,
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

        let node = Node::new(config, self.client_key.clone());
        node.start().await.unwrap();
        let handle = node.get_node_handle().unwrap();
        self.nodes.insert(name, node);
        handle
    }
}

pub struct EncoderSwarmBuilder {
    encoders: Option<Vec<EncoderConfig>>,
    client_key: Option<PeerPublicKey>,
}

impl EncoderSwarmBuilder {
    pub fn new() -> Self {
        Self {
            encoders: None,
            client_key: None,
        }
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.encoders = Some(encoders);
        self
    }

    pub fn with_client_key(mut self, client_key: PeerPublicKey) -> Self {
        self.client_key = Some(client_key);
        self
    }

    pub fn build(self) -> EncoderSwarm {
        // We require configs to be provided
        let encoder_configs = self.encoders.expect("Encoder configs must be provided");

        // Create nodes from the configs
        let nodes = encoder_configs
            .iter()
            .map(|config| {
                info!(
                    "Configuring encoder with name {:?}",
                    config.protocol_public_key()
                );
                (
                    config.protocol_public_key(),
                    Node::new(config.clone(), self.client_key.clone()),
                )
            })
            .collect();

        EncoderSwarm {
            nodes,
            client_key: self.client_key,
        }
    }
}
