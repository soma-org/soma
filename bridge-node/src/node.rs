//! Bridge node orchestrator.
//!
//! Spawns and coordinates all bridge node subsystems:
//! - EthSyncer (Ethereum event watching)
//! - gRPC server (signature exchange)
//! - Checkpoint watcher (Soma-side observation)
//! - Deposit handler (processes Ethereum deposits → signs → submits BridgeDeposit)
//! - Withdrawal handler (processes Soma withdrawals → signs → submits Ethereum tx)

use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tracing::{info, warn};

use fastcrypto::hash::HashFunction;
use fastcrypto::secp256k1::Secp256k1KeyPair;
use types::bridge::{BridgeCommittee, sign_bridge_message};

use crate::checkpoint_watcher::{CheckpointEvent, CheckpointWatcher};
use crate::config::BridgeNodeConfig;
use crate::error::BridgeResult;
use crate::eth_client::EthClient;
use crate::eth_syncer::EthSyncer;
use crate::proto;
use crate::server::BridgeServer;

/// The bridge node orchestrator.
pub struct BridgeNode {
    config: BridgeNodeConfig,
    bridge_keypair: Arc<Secp256k1KeyPair>,
    signer_index: u32,
    committee: BridgeCommittee,
}

impl BridgeNode {
    pub fn new(
        config: BridgeNodeConfig,
        bridge_keypair: Secp256k1KeyPair,
        signer_index: u32,
        committee: BridgeCommittee,
    ) -> Self {
        Self {
            config,
            bridge_keypair: Arc::new(bridge_keypair),
            signer_index,
            committee,
        }
    }

    /// Start all bridge node subsystems. Returns task handles.
    pub async fn run(self) -> BridgeResult<Vec<JoinHandle<()>>> {
        let mut handles = Vec::new();

        // 1. Create Ethereum client
        let eth_client = Arc::new(
            EthClient::new(
                self.config.eth_rpc_urls.clone(),
                &self.config.bridge_contract_address,
            )
            .await?,
        );

        // 2. Create signature exchange server
        let server = Arc::new(BridgeServer::new(self.committee.clone()));

        // 3. gRPC server deferred — peer exchange requires protoc-generated service wiring.
        let grpc_addr = self.config.grpc_listen_address;
        info!(%grpc_addr, "gRPC server address configured (peer exchange deferred)");

        // 4. Start Ethereum syncer
        let poll_interval =
            Duration::from_millis(self.config.eth_poll_interval_ms);
        let syncer = EthSyncer::new(
            Arc::clone(&eth_client),
            poll_interval,
            self.config.max_log_query_range,
        );
        // TODO: load last processed block from persistent state
        let start_block = 0;
        let syncer_handle = syncer.start(start_block);
        handles.extend(syncer_handle.task_handles);

        // 5. Start checkpoint watcher
        let (_checkpoint_watcher, mut checkpoint_rx) = CheckpointWatcher::new(256);
        // TODO: wire into data-ingestion checkpoint subscription

        // 6. Deposit handler — processes Ethereum deposit events
        let keypair = Arc::clone(&self.bridge_keypair);
        let signer_idx = self.signer_index;
        let deposit_server = Arc::clone(&server);
        let mut event_rx = syncer_handle.event_rx;
        let deposit_handle = tokio::spawn(async move {
            info!("Deposit handler started");
            while let Some((_block, events)) = event_rx.recv().await {
                for event in events {
                    let action = event.to_bridge_action();
                    let msg_bytes = action.to_message_bytes();
                    let sig = sign_bridge_message(&keypair, &msg_bytes);
                    let digest = fastcrypto::hash::Keccak256::digest(&msg_bytes);

                    let req = tonic::Request::new(proto::SignatureRequest {
                        action_digest: digest.as_ref().to_vec(),
                        signature: sig.as_ref().to_vec(),
                        signer_index: signer_idx,
                    });

                    use crate::proto::bridge_node_server::BridgeNode as BridgeNodeTrait;
                    if let Err(e) = deposit_server.submit_signature(req).await {
                        warn!("Failed to store deposit signature: {e}");
                    }

                    // TODO: broadcast to peers via gRPC client
                    // TODO: check quorum and submit BridgeDeposit system tx to Soma
                }
            }
        });
        handles.push(deposit_handle);

        // 7. Withdrawal handler — processes Soma PendingWithdrawal events
        let keypair2 = Arc::clone(&self.bridge_keypair);
        let signer_idx2 = self.signer_index;
        let withdrawal_server = Arc::clone(&server);
        let withdrawal_handle = tokio::spawn(async move {
            info!("Withdrawal handler started");
            while let Some(event) = checkpoint_rx.recv().await {
                match event {
                    CheckpointEvent::NewWithdrawal(w) => {
                        let action = w.to_bridge_action();
                        let msg_bytes = action.to_message_bytes();
                        let sig = sign_bridge_message(&keypair2, &msg_bytes);
                        let digest =
                            fastcrypto::hash::Keccak256::digest(&msg_bytes);

                        let req = tonic::Request::new(proto::SignatureRequest {
                            action_digest: digest.as_ref().to_vec(),
                            signature: sig.as_ref().to_vec(),
                            signer_index: signer_idx2,
                        });

                        use crate::proto::bridge_node_server::BridgeNode as BridgeNodeTrait;
                        if let Err(e) = withdrawal_server.submit_signature(req).await
                        {
                            warn!("Failed to store withdrawal signature: {e}");
                        }

                        // TODO: broadcast to peers via gRPC client
                        // TODO: check quorum and submit Ethereum withdraw tx
                    }
                    CheckpointEvent::EpochBoundary { epoch } => {
                        info!(
                            epoch,
                            "Epoch boundary — committee update deferred"
                        );
                        // TODO: fetch new validator set, sign committee update,
                        // submit to Ethereum contract
                    }
                }
            }
        });
        handles.push(withdrawal_handle);

        info!("Bridge node started with {} subsystems", handles.len());
        Ok(handles)
    }
}
