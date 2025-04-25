use fastcrypto::traits::KeyPair;
use futures::FutureExt;
use node::handle::SomaNodeHandle;
use node::SomaNode;
use std::sync::{Arc, Weak};
use std::thread;
use tracing::{info, trace};
use types::base::ConciseableName;
use types::config::node_config::NodeConfig;
use types::crypto::AuthorityPublicKeyBytes;

#[derive(Debug)]
pub(crate) struct Container {
    join_handle: Option<thread::JoinHandle<()>>,
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,
    node: Weak<SomaNode>,
}

/// When dropped, stop and wait for the node running in this Container to completely shutdown.
impl Drop for Container {
    fn drop(&mut self) {
        trace!("dropping Container");

        let thread = self.join_handle.take().unwrap();

        let cancel_handle = self.cancel_sender.take().unwrap();

        // Notify the thread to shutdown
        let _ = cancel_handle.send(());

        // Wait for the thread to join
        thread.join().unwrap();

        trace!("finished dropping Container");
    }
}

impl Container {
    /// Spawn a new Node.
    pub async fn spawn(config: NodeConfig) -> Self {
        let (startup_sender, startup_receiver) = tokio::sync::oneshot::channel();
        let (cancel_sender, cancel_receiver) = tokio::sync::oneshot::channel();
        let name = AuthorityPublicKeyBytes::from(config.protocol_key_pair().public())
            .concise()
            .to_string();

        let thread = thread::Builder::new().name(name).spawn(move || {
            let span =  Some(tracing::span!(
                tracing::Level::INFO,
                "node",
                name =% AuthorityPublicKeyBytes::from(config.protocol_key_pair().public()).concise(),
            ));

            let _guard = span.as_ref().map(|span| span.enter());

            let mut builder = tokio::runtime::Builder::new_current_thread(); // TODO: multi threaded runtime
            let runtime = builder.enable_all().build().unwrap();

            runtime.block_on(async move {
               
                let server = SomaNode::start(config).await.unwrap();
                // Notify that we've successfully started the node
                let _ = startup_sender.send(Arc::downgrade(&server));
                // run until canceled
                cancel_receiver.map(|_| ()).await;

                trace!("cancellation received; shutting down thread");
            });
        }).unwrap();

        let node = startup_receiver.await.unwrap();

        Self {
            join_handle: Some(thread),
            cancel_sender: Some(cancel_sender),
            node,
        }
    }

    /// Get a SomaNodeHandle to the node owned by the container.
    pub fn get_node_handle(&self) -> Option<SomaNodeHandle> {
        Some(SomaNodeHandle::new(self.node.upgrade()?))
    }

    /// Check to see that the Node is still alive by checking if the receiving side of the
    /// `cancel_sender` has been dropped.
    ///
    //TODO When we move to rust 1.61 we should also use
    // https://doc.rust-lang.org/stable/std/thread/struct.JoinHandle.html#method.is_finished
    // in order to check if the thread has finished.
    pub fn is_alive(&self) -> bool {
        if let Some(cancel_sender) = &self.cancel_sender {
            !cancel_sender.is_closed()
        } else {
            false
        }
    }
}
