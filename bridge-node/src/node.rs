//! Bridge node orchestrator.
//!
//! Spawns and coordinates all bridge node subsystems:
//! - EthSyncer (Ethereum event watching)
//! - gRPC server (signature exchange)
//! - Checkpoint watcher (Soma-side observation)
//! - Deposit handler (processes Ethereum deposits → signs → submits BridgeDeposit)
//! - Withdrawal handler (processes Soma withdrawals → signs → submits Ethereum tx)

use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use tracing::{info, warn};

use fastcrypto::secp256k1::Secp256k1KeyPair;
use fastcrypto::traits::KeyPair;
use types::base::SomaAddress;
use types::bridge::{BridgeChainId, BridgeCommittee};
use types::crypto::SomaKeyPair;

use crate::action_executor::{
    BridgeActionExecutionWrapper, BridgeActionExecutor, submit_to_executor,
};
use crate::aggregator::BridgeAuthorityAggregator;
use crate::checkpoint_watcher::{CheckpointEvent, CheckpointWatcher};
use crate::config::BridgeNodeConfig;
use crate::error::BridgeResult;
use crate::eth_client::EthClient;
use crate::eth_syncer::EthSyncer;
use crate::soma_client::SomaBridgeClient;
use crate::storage::BridgeOrchestratorTables;
use crate::types::BridgeAction;

/// The bridge node orchestrator.
pub struct BridgeNode {
    config: BridgeNodeConfig,
    bridge_keypair: Arc<Secp256k1KeyPair>,
    committee: BridgeCommittee,
    /// Optional relayer config for the on-chain executor. Without
    /// this, the bridge node runs in **sig-cache-only mode** —
    /// validators still sign observed events and serve sigs to peers
    /// over gRPC, but no one is responsible for assembling a quorum
    /// cert and submitting it on chain. Use [`Self::with_relayer`] to
    /// enable end-to-end action submission.
    relayer: Option<RelayerConfig>,
}

#[derive(Clone)]
struct RelayerConfig {
    address: SomaAddress,
    keypair: Arc<SomaKeyPair>,
    soma_chain_id: BridgeChainId,
}

impl BridgeNode {
    pub fn new(
        config: BridgeNodeConfig,
        bridge_keypair: Secp256k1KeyPair,
        committee: BridgeCommittee,
    ) -> Self {
        Self { config, bridge_keypair: Arc::new(bridge_keypair), committee, relayer: None }
    }

    /// Enable on-chain action submission. Without calling this, the
    /// node runs in sig-cache-only mode. `relayer_address` is the
    /// account that pays for and signs the wrapper user-tx; it must
    /// hold enough USDC to cover validator fees.
    pub fn with_relayer(
        mut self,
        relayer_address: SomaAddress,
        relayer_keypair: SomaKeyPair,
        soma_chain_id: BridgeChainId,
    ) -> Self {
        self.relayer = Some(RelayerConfig {
            address: relayer_address,
            keypair: Arc::new(relayer_keypair),
            soma_chain_id,
        });
        self
    }

    /// Start all bridge node subsystems. Returns task handles.
    pub async fn run(self) -> BridgeResult<Vec<JoinHandle<()>>> {
        let mut handles = Vec::new();

        // 0. Open the durable WAL — pending actions + cursors. Production
        // restarts must pick up exactly where the previous run left off.
        let wal = BridgeOrchestratorTables::open(&self.config.wal_path)?;
        info!(
            wal_path = %self.config.wal_path.display(),
            pending = wal.get_all_pending_actions()?.len(),
            "Bridge WAL opened"
        );

        // 1. Create Ethereum client
        let eth_client = Arc::new(
            EthClient::new(self.config.eth_rpc_urls.clone(), &self.config.bridge_contract_address)
                .await?,
        );

        // 2. Sig exchange happens over HTTP (spawned below once
        // soma_client is reachable). The old gRPC BridgeServer is
        // retired — fetch-and-sign via HTTP is both safer (each sig
        // request re-verifies against chain state) and simpler (no
        // proto codegen, no separate cache to keep coherent).

        // 3.5. Action executor (optional, opt-in via `with_relayer`).
        // When configured, the executor drains the WAL on startup and
        // then consumes from `executor_signing_tx` for every newly
        // observed action. When NOT configured, the node runs in
        // sig-cache-only mode (Stage 6a/6b behavior).
        //
        // `committee_snapshot` propagates the monitor's `ArcSwap` out
        // of the relayer branch so post-executor handlers (epoch
        // boundary → CommitteeUpdate) can read the live committee
        // without a fresh RPC call.
        let mut committee_snapshot: Option<Arc<arc_swap::ArcSwap<BridgeCommittee>>> = None;
        let executor_signing_tx: Option<mpsc::Sender<BridgeActionExecutionWrapper>> = match self
            .relayer
            .clone()
        {
            Some(relayer) => {
                match SomaBridgeClient::new_rpc(&self.config.soma_rpc_url, relayer.soma_chain_id)
                    .await
                {
                    Ok(client) => {
                        // Spawn the bridge state monitor — it polls
                        // pause state + committee on a timer and
                        // publishes changes to in-memory channels.
                        // Subsystems read from the channels rather
                        // than re-polling the RPC themselves.
                        let (monitor, monitor_channels) = crate::monitor::BridgeMonitor::new(
                            Arc::clone(&client),
                            false, // assume unpaused at startup; first
                            // poll corrects this immediately.
                            self.committee.clone(),
                            crate::monitor::DEFAULT_POLL_INTERVAL,
                        );
                        handles.push(monitor.run());

                        // Spawn the HTTP REST sig-exchange server
                        // (Sui parity). Peers fetch signatures by
                        // GETing the action-specific route; the
                        // handler re-verifies each action against
                        // chain state (via eth_client / soma client)
                        // before signing. The gRPC server below
                        // remains for now as a transitional dual
                        // surface — Phase 4b will switch the
                        // executor's aggregator to peer-broadcast
                        // and the gRPC path can be retired.
                        //
                        // Construction can only fail if the operator
                        // put a token-transfer action in the
                        // governance whitelist (config mistake); if
                        // so we log and run without the HTTP
                        // surface rather than crashing the node.
                        match crate::handler::BridgeRequestHandler::new(
                            (*self.bridge_keypair).copy(),
                            Arc::clone(&eth_client),
                            Arc::clone(&client),
                            self.config.approved_governance_actions.clone(),
                        ) {
                            Ok(handler) => {
                                let metadata =
                                    Arc::new(crate::http_server::BridgeNodePublicMetadata::new(
                                        env!("CARGO_PKG_VERSION"),
                                    ));
                                let http_addr = self.config.http_listen_address;
                                info!(%http_addr, "bridge HTTP server starting");
                                handles.push(crate::http_server::run_server(
                                    &http_addr, handler, metadata,
                                ));
                            }
                            Err(e) => {
                                warn!(
                                    error = %e,
                                    "approved_governance_actions invalid; HTTP server NOT spawned"
                                );
                            }
                        }

                        // Peer-broadcast aggregator. Fans out HTTP
                        // sig requests to each non-blocklisted
                        // committee member's registered `http_url`
                        // and assembles a cert as responses arrive.
                        // Reads committee state directly from the
                        // monitor's `ArcSwap` so rotations propagate
                        // without a restart and without a glue task.
                        let aggregator: Arc<dyn BridgeAuthorityAggregator> =
                            Arc::new(crate::peer_aggregator::PeerBroadcastAggregator::new(
                                Arc::clone(&monitor_channels.committee),
                            ));

                        // Executor reads pause state from the
                        // monitor's watch channel and the live
                        // committee from its `ArcSwap` snapshot —
                        // no per-attempt RPC polling, no glue task.
                        let executor = BridgeActionExecutor::new(
                            Arc::clone(&client),
                            aggregator,
                            Arc::clone(&wal),
                            relayer.address,
                            relayer.keypair,
                            Arc::clone(&monitor_channels.committee),
                            monitor_channels.bridge_paused_rx.clone(),
                        );
                        let (executor_handles, signing_tx) = executor.run();
                        handles.extend(executor_handles);

                        // Drain any pending actions left in the WAL
                        // by a previous run (crash recovery). Each
                        // gets re-attempted with a fresh attempt
                        // counter from zero.
                        let pending = wal.get_all_pending_actions()?;
                        info!(
                            count = pending.len(),
                            "replaying pending bridge actions to executor",
                        );
                        for action in pending {
                            if let Err(e) = submit_to_executor(&signing_tx, action).await {
                                warn!("WAL replay submit failed: {e}");
                            }
                        }

                        info!("Bridge action executor wired up");

                        // Hand the live committee snapshot out so
                        // the epoch-boundary handler can read it
                        // when emitting CommitteeUpdate actions.
                        committee_snapshot = Some(Arc::clone(&monitor_channels.committee));

                        // Spawn the conservation-invariant watchdog
                        // if configured. It polls Eth vault balance
                        // + Soma USDC supply on a timer and emits
                        // an auto-pause action via the executor's
                        // signing queue on sustained violation.
                        //
                        // Eligible only when both the relayer and a
                        // watchdog config are present — the
                        // watchdog needs both the Eth client (to
                        // read vault balance) and the executor's
                        // signing_tx (to fire pause actions).
                        if let Some(wd_cfg) = self.config.watchdog.clone() {
                            let poll_interval = wd_cfg.poll_interval();
                            let usdc_addr = wd_cfg.usdc_contract_address.clone();
                            let bridge_addr = wd_cfg.eth_bridge_contract_address.clone();

                            // Real Soma supply reader — closures over
                            // the production SomaBridgeClient. Replaces
                            // the stub-at-0 from before PR A landed
                            // BridgeState.total_usdc_supply.
                            let soma_supply: crate::watchdog::SomaSupplyReader = {
                                let c = Arc::clone(&client);
                                Arc::new(move || {
                                    let c = Arc::clone(&c);
                                    Box::pin(async move { c.get_total_usdc_supply().await })
                                })
                            };
                            let soma_paused: crate::watchdog::SomaPausedReader = {
                                let c = Arc::clone(&client);
                                Arc::new(move || {
                                    let c = Arc::clone(&c);
                                    Box::pin(async move { c.is_bridge_paused().await })
                                })
                            };

                            // Real reader: read the on-chain
                            // BridgeState.system_message_seq_nums[EmergencyOp]
                            // every time the watchdog fires an
                            // auto-pause. Re-read each time so a
                            // pause cert that landed since the
                            // last watchdog tick (manual op or
                            // peer-fired) doesn't get the same
                            // nonce as our about-to-fire cert.
                            let expected_pause_nonce: crate::watchdog::ExpectedPauseNonceReader = {
                                let c = Arc::clone(&client);
                                Arc::new(move || {
                                    let c = Arc::clone(&c);
                                    Box::pin(async move {
                                        c.get_next_system_message_seq(
                                            types::bridge::BridgeMessageType::EmergencyOp,
                                        )
                                        .await
                                    })
                                })
                            };

                            let watchdog = crate::watchdog::BridgeWatchdog::new()
                                .with(Box::new(crate::watchdog::EthVaultBalanceObservable::new(
                                    Arc::clone(&eth_client),
                                    usdc_addr.clone(),
                                    bridge_addr.clone(),
                                    poll_interval,
                                )))
                                .with(Box::new(crate::watchdog::SomaUsdcSupplyObservable::new(
                                    Arc::clone(&soma_supply),
                                    poll_interval,
                                )))
                                .with(Box::new(crate::watchdog::EthBridgeStatusObservable::new(
                                    Arc::clone(&eth_client),
                                    bridge_addr.clone(),
                                    poll_interval,
                                )))
                                .with(Box::new(crate::watchdog::SomaBridgeStatusObservable::new(
                                    soma_paused,
                                    poll_interval,
                                )))
                                .with(Box::new(
                                    crate::watchdog::ConservationInvariantObservable::new(
                                        Arc::clone(&eth_client),
                                        soma_supply,
                                        usdc_addr,
                                        bridge_addr.clone(),
                                        poll_interval,
                                        wd_cfg.failure_threshold,
                                        wd_cfg.in_flight_tolerance_micro,
                                        signing_tx.clone(),
                                        expected_pause_nonce,
                                    ),
                                ));
                            handles.extend(watchdog.start());
                            info!(
                                eth_contract = %bridge_addr,
                                "BridgeWatchdog spawned (5 observables)"
                            );
                        } else {
                            info!("no watchdog configured; auto-pause disabled");
                        }

                        // Spawn the Eth-side outbound relayer if
                        // configured. Polls Soma for cert-attached
                        // PendingWithdrawals and submits release
                        // txs to Ethereum via the operator wallet.
                        if let Some(or_cfg) = self.config.outbound_relayer.clone() {
                            match build_outbound_relayer(
                                &or_cfg,
                                &self.config.eth_rpc_urls,
                                Arc::clone(&client),
                                Arc::clone(&wal),
                            ) {
                                Ok(relayer) => {
                                    handles.push(relayer.start());
                                    info!(
                                        bridge_contract = %or_cfg.bridge_contract_address,
                                        poll_ms = or_cfg.poll_interval_ms,
                                        scan_window = or_cfg.scan_window,
                                        "OutboundRelayer spawned — Eth submission live"
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        "OutboundRelayer config invalid; Eth-side release disabled"
                                    );
                                }
                            }
                        } else {
                            info!("no outbound relayer configured; Eth-side release disabled");
                        }

                        Some(signing_tx)
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            "could not connect to Soma RPC; running in sig-cache-only mode"
                        );
                        None
                    }
                }
            }
            None => {
                info!("no relayer configured; running in sig-cache-only mode");
                None
            }
        };

        // 4. Start Ethereum syncer. Resume from the persisted cursor if
        // present (Sui parity: stored value is "last processed", so we
        // start from cursor + 1). Otherwise fall back to the configured
        // `eth_start_block_fallback`.
        let poll_interval = Duration::from_millis(self.config.eth_poll_interval_ms);
        let syncer =
            EthSyncer::new(Arc::clone(&eth_client), poll_interval, self.config.max_log_query_range);
        let bridge_contract_bytes: [u8; 20] =
            parse_eth_addr(&self.config.bridge_contract_address).unwrap_or([0u8; 20]);
        let start_block = match wal.get_eth_cursor(&bridge_contract_bytes)? {
            Some(last_processed) => last_processed.saturating_add(1),
            None => self.config.eth_start_block_fallback,
        };
        info!(start_block, "Eth syncer starting");
        let syncer_handle = syncer.start(start_block);
        handles.extend(syncer_handle.task_handles);

        // 5. Start checkpoint watcher + Soma syncer that drives it.
        // The watcher emits CheckpointEvent::NewWithdrawal /
        // EpochBoundary into `checkpoint_rx`; the withdrawal handler
        // below consumes from it. The syncer polls Soma's RPC for
        // sequential checkpoints, resuming from the persisted WAL
        // cursor on restart.
        let (checkpoint_watcher, mut checkpoint_rx) = CheckpointWatcher::new(256);
        let soma_start_seq = wal.get_soma_cursor()?.map(|c| c + 1).unwrap_or(0);
        match rpc::api::client::Client::new(self.config.soma_rpc_url.clone()) {
            Ok(soma_rpc_client) => {
                info!(start_seq = soma_start_seq, "Soma syncer starting");
                handles.push(crate::soma_syncer::run_soma_syncer(
                    soma_rpc_client,
                    checkpoint_watcher,
                    Arc::clone(&wal),
                    soma_start_seq,
                ));
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "could not construct Soma RPC client; outbound observation disabled"
                );
                // Drop watcher; rx will close. Withdrawal handler still
                // spawns but never receives events — equivalent to
                // running with no Soma side observer.
                drop(checkpoint_watcher);
            }
        }

        // 6. Deposit handler — processes Ethereum deposit events. Persists
        // the action to the WAL BEFORE handing to the executor, so a crash
        // before submission is recoverable on restart (Sui parity:
        // insert_pending_actions before cursor advance).
        //
        // No local sig-cache write here anymore — peers fetch sigs from
        // each other via the HTTP server's fetch-and-sign endpoints,
        // which re-verify each request against chain state before
        // producing a signature. The gRPC sig-cache it replaced was an
        // attack surface (peers could pollute caches) and a redundancy.
        let deposit_wal = Arc::clone(&wal);
        let deposit_executor_tx = executor_signing_tx.clone();
        let mut event_rx = syncer_handle.event_rx;
        let deposit_handle = tokio::spawn(async move {
            info!("Deposit handler started");
            while let Some((batch_end_block, events)) = event_rx.recv().await {
                for event in events {
                    let action = event.to_bridge_action();
                    let digest_hex = hex::encode(action.digest());
                    let nonce = action.nonce();
                    if let Err(e) = deposit_wal.insert_pending_action(&action) {
                        warn!(
                            action_digest = %digest_hex,
                            nonce,
                            error = %e,
                            "WAL insert failed for deposit"
                        );
                        continue;
                    }
                    if let Some(tx) = deposit_executor_tx.as_ref() {
                        if let Err(e) = submit_to_executor(tx, action).await {
                            warn!(
                                action_digest = %digest_hex,
                                nonce,
                                error = %e,
                                "submit_to_executor failed for deposit"
                            );
                        }
                    }
                }

                if let Err(e) =
                    deposit_wal.update_eth_cursor(bridge_contract_bytes, batch_end_block)
                {
                    warn!(
                        block = batch_end_block,
                        error = %e,
                        "WAL cursor update failed"
                    );
                }
            }
        });
        handles.push(deposit_handle);

        // 7. Withdrawal handler — processes Soma PendingWithdrawal events.
        // Persists action to WAL before handing to the executor (same
        // crash-safety pattern as the deposit handler). No local sig
        // cache write — peers fetch sigs via fetch-and-sign HTTP.
        let withdrawal_wal = Arc::clone(&wal);
        let withdrawal_executor_tx = executor_signing_tx.clone();
        let epoch_committee = committee_snapshot.clone();
        let withdrawal_handle = tokio::spawn(async move {
            info!("Withdrawal handler started");
            while let Some(event) = checkpoint_rx.recv().await {
                match event {
                    CheckpointEvent::NewWithdrawal(w) => {
                        let action = w.to_bridge_action();
                        let digest_hex = hex::encode(action.digest());
                        let nonce = action.nonce();
                        if let Err(e) = withdrawal_wal.insert_pending_action(&action) {
                            warn!(
                                action_digest = %digest_hex,
                                nonce,
                                error = %e,
                                "WAL insert failed for withdrawal"
                            );
                            continue;
                        }
                        if let Some(tx) = withdrawal_executor_tx.as_ref() {
                            if let Err(e) = submit_to_executor(tx, action).await {
                                warn!(
                                    action_digest = %digest_hex,
                                    nonce,
                                    error = %e,
                                    "submit_to_executor failed for withdrawal"
                                );
                            }
                        }
                    }
                    CheckpointEvent::EpochBoundary { epoch } => {
                        // Sui parity: there is no automatic on-chain
                        // CommitteeUpdate at epoch boundary. Sui's
                        // Eth-side BridgeCommittee membership is set
                        // at deploy and only changes via UUPS upgrade
                        // (the existing committee signs a quorum
                        // EvmContractUpgrade message that swaps the
                        // BridgeCommittee impl to one with new members
                        // baked into its initializer). The blocklist
                        // gives operators a fast lever to neutralize
                        // an individual member without rotation. So
                        // epoch boundaries are a no-op here.
                        info!(
                            epoch,
                            "epoch boundary observed (no-op: Eth committee rotation is via UUPS upgrade)"
                        );
                    }
                }
            }
        });
        handles.push(withdrawal_handle);

        info!("Bridge node started with {} subsystems", handles.len());
        Ok(handles)
    }
}

/// Parse a hex-encoded Eth address (with or without "0x" prefix) into
/// 20 raw bytes. Used to key the WAL Eth cursor by contract address.
fn parse_eth_addr(s: &str) -> Option<[u8; 20]> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).ok()?;
    bytes.try_into().ok()
}

/// Build a fully-wired outbound relayer from config: parses the
/// operator wallet, constructs the [`crate::eth_submitter::EthSubmitter`]
/// against the chosen Eth RPC endpoint, and hands it to a fresh
/// [`crate::outbound_relayer::OutboundRelayer`].
fn build_outbound_relayer<C: crate::soma_client::SomaBridgeClientInner + 'static>(
    cfg: &crate::config::OutboundRelayerConfigBlock,
    eth_rpc_urls: &[String],
    soma_client: Arc<crate::soma_client::SomaBridgeClient<C>>,
    wal: Arc<crate::storage::BridgeOrchestratorTables>,
) -> crate::error::BridgeResult<crate::outbound_relayer::OutboundRelayer<C>> {
    let wallet = crate::eth_wallet::EthWallet::from_hex(&cfg.operator_private_key_hex)?;
    let rpc_url = cfg
        .eth_submit_rpc_url
        .clone()
        .or_else(|| eth_rpc_urls.first().cloned())
        .ok_or_else(|| {
            crate::error::BridgeError::ConfigError(
                "OutboundRelayer: no Eth RPC URL available (set eth_submit_rpc_url or eth_rpc_urls)".to_string(),
            )
        })?;
    let bridge_addr_bytes = parse_eth_addr(&cfg.bridge_contract_address).ok_or_else(|| {
        crate::error::BridgeError::ConfigError(format!(
            "OutboundRelayer: bridge_contract_address `{}` is not a valid 20-byte Eth address",
            cfg.bridge_contract_address
        ))
    })?;
    let bridge_addr = alloy::primitives::Address::from(bridge_addr_bytes);

    let submitter =
        Arc::new(crate::eth_submitter::EthSubmitter::new(&rpc_url, bridge_addr, wallet)?);
    // WAL-backed tracker: a restart sees the same relayed set the
    // previous run finished with, so already-landed withdrawals don't
    // re-submit and burn operator gas on Eth-side reverts.
    let tracker = Arc::new(crate::outbound_relayer::WalRelayedTracker::new(wal));

    Ok(crate::outbound_relayer::OutboundRelayer::new(
        soma_client,
        submitter,
        tracker,
        Duration::from_millis(cfg.poll_interval_ms),
        cfg.scan_window,
    ))
}
