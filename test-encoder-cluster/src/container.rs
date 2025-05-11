use crate::config::EncoderConfig;
use crate::swarm::multiaddr_compat::to_network_multiaddr;
use encoder::core::encoder_node::{EncoderNode, EncoderNodeHandle};
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::traits::KeyPair;
use futures::FutureExt;
use shared::crypto::keys::PeerKeyPair;
use std::sync::{Arc, Weak};
use std::thread;
use tracing::{info, trace};

#[derive(Debug)]
pub(crate) struct Container {
    join_handle: Option<thread::JoinHandle<()>>,
    cancel_sender: Option<tokio::sync::oneshot::Sender<()>>,
    node: Weak<EncoderNode>,
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
    pub async fn spawn(config: EncoderConfig) -> Self {
        let (startup_sender, startup_receiver) = tokio::sync::oneshot::channel();
        let (cancel_sender, cancel_receiver) = tokio::sync::oneshot::channel();
        let name = format!(
            "{:?}",
            Hex::encode(
                config
                    .protocol_public_key()
                    .inner()
                    .to_string()
                    .get(0..4)
                    .unwrap()
            )
        );

        let thread = thread::Builder::new()
            .name(name)
            .spawn(move || {
                let span = Some(tracing::span!(
                    tracing::Level::INFO,
                    "node",
                    name =% format!(
                        "{:?}",
                        Hex::encode(
                            config
                                .protocol_public_key()
                                .inner()
                                .to_string()
                                .get(0..4)
                                .unwrap()
                        )
                    ),
                ));

                let _guard = span.as_ref().map(|span| span.enter());

                let mut builder = tokio::runtime::Builder::new_current_thread(); // TODO: multi threaded runtime
                let runtime = builder.enable_all().build().unwrap();

                runtime.block_on(async move {
                    let server = Arc::new(
                        EncoderNode::start(
                            config.encoder_keypair.encoder_keypair().clone(),
                            config.parameters,
                            config.object_parameters,
                            config.probe_parameters,
                            PeerKeyPair::new(config.peer_keypair.keypair().inner().copy()),
                            to_network_multiaddr(&config.internal_network_address),
                            to_network_multiaddr(&config.external_network_address),
                            to_network_multiaddr(&config.object_address),
                            to_network_multiaddr(&config.probe_address),
                            &config.project_root,
                            &config.entry_point,
                            config.validator_rpc_address,
                            config.genesis_committee,
                            config.epoch_duration_ms,
                        )
                        .await,
                    );
                    // Notify that we've successfully started the node
                    let _ = startup_sender.send(Arc::downgrade(&server));
                    // run until canceled
                    cancel_receiver.map(|_| ()).await;

                    trace!("cancellation received; shutting down thread");
                });
            })
            .unwrap();

        let node = startup_receiver.await.unwrap();

        Self {
            join_handle: Some(thread),
            cancel_sender: Some(cancel_sender),
            node,
        }
    }

    /// Get a SomaNodeHandle to the node owned by the container.
    pub fn get_node_handle(&self) -> Option<EncoderNodeHandle> {
        Some(EncoderNodeHandle::new(self.node.upgrade()?))
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
