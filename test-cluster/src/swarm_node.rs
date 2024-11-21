use anyhow::Result;
use node::handle::SomaNodeHandle;
use std::sync::{Mutex, MutexGuard};
use tracing::info;
use types::{
    base::{AuthorityName, ConciseableName},
    config::node_config::NodeConfig,
};

use super::container::Container;

/// A handle to an in-memory Soma Node.
///
/// Each Node is attempted to run in isolation from each other by running them in their own tokio
/// runtime in a separate thread. By doing this we can ensure that all asynchronous tasks
/// associated with a Node are able to be stopped when desired (either when a Node is dropped or
/// explicitly stopped by calling [`Node::stop`]) by simply dropping that Node's runtime.
#[derive(Debug)]
pub struct Node {
    container: Mutex<Option<Container>>,
    config: Mutex<NodeConfig>,
}

impl Node {
    /// Create a new Node from the provided `NodeConfig`.
    ///
    /// The Node is returned without being started. See [`Node::spawn`] or [`Node::start`] for how to
    /// start the node.
    pub fn new(config: NodeConfig) -> Self {
        Self {
            container: Default::default(),
            config: config.into(),
        }
    }

    /// Return the `name` of this Node
    pub fn name(&self) -> AuthorityName {
        self.config().protocol_public_key()
    }

    pub fn config(&self) -> MutexGuard<'_, NodeConfig> {
        self.config.lock().unwrap()
    }

    /// Start this Node
    pub async fn spawn(&self) -> Result<()> {
        info!(name =% self.name().concise(), "starting in-memory node");
        let config = self.config().clone();
        *self.container.lock().unwrap() = Some(Container::spawn(config).await);
        Ok(())
    }

    /// Start this Node, waiting until its completely started up.
    pub async fn start(&self) -> Result<()> {
        self.spawn().await
    }

    /// Stop this Node
    pub fn stop(&self) {
        info!(name =% self.name().concise(), "stopping in-memory node");
        *self.container.lock().unwrap() = None;
        info!(name =% self.name().concise(), "node stopped");
    }

    /// If this Node is currently running
    pub fn is_running(&self) -> bool {
        self.container
            .lock()
            .unwrap()
            .as_ref()
            .map_or(false, |c| c.is_alive())
    }

    pub fn get_node_handle(&self) -> Option<SomaNodeHandle> {
        self.container
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|c| c.get_node_handle())
    }
}
