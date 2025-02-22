use std::{path::PathBuf, sync::Arc, time::Duration};

use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use fastcrypto::traits::KeyPair;
use mysticeti_client::LazyMysticetiClient;
use mysticeti_manager::MysticetiManager;
use parking_lot::RwLock;
use tokio::time::{sleep, timeout};
use tracing::info;
use types::{
    accumulator::AccumulatorStore,
    committee::EpochId,
    config::node_config::{ConsensusConfig, NodeConfig},
    consensus::ConsensusTransaction,
    error::SomaResult,
    storage::{consensus::ConsensusStore, read_store::ReadCommitteeStore},
};

use crate::{
    adapter::{ConsensusAdapter, SubmitToConsensus},
    epoch_store::AuthorityPerEpochStore,
    handler::ConsensusHandlerInitializer,
    tx_validator::TxValidator,
};

pub mod mysticeti_client;
pub mod mysticeti_manager;

#[derive(PartialEq)]
pub(crate) enum Running {
    True(EpochId),
    False,
}

#[async_trait]
pub trait ConsensusManagerTrait {
    async fn start(
        &self,
        node_config: &NodeConfig,
        epoch_store: Arc<AuthorityPerEpochStore>,
        consensus_handler_initializer: ConsensusHandlerInitializer,
        tx_validator: TxValidator,
    );

    async fn shutdown(&self);

    async fn is_running(&self) -> bool;
}

/// Used by validator to start consensus protocol for each epoch.
pub struct ConsensusManager {
    consensus_config: ConsensusConfig,
    mysticeti_manager: MysticetiManager,
    mysticeti_client: Arc<LazyMysticetiClient>,
    active: parking_lot::Mutex<bool>,
    consensus_client: Arc<ConsensusClient>,
}

impl ConsensusManager {
    pub fn new(
        node_config: &NodeConfig,
        consensus_config: &ConsensusConfig,
        consensus_client: Arc<ConsensusClient>,
        accumulator_store: Arc<dyn AccumulatorStore>,
        consensus_adapter: Arc<ConsensusAdapter>,
        consensus_store: Arc<dyn ConsensusStore>,
        committee_store: Arc<dyn ReadCommitteeStore>,
    ) -> Self {
        let mysticeti_client = Arc::new(LazyMysticetiClient::new());
        let mysticeti_manager = MysticetiManager::new(
            node_config.worker_key_pair().into_inner(),
            node_config.network_key_pair().into_inner(),
            node_config.protocol_key_pair().copy(),
            consensus_config.db_path().to_path_buf(),
            mysticeti_client.clone(),
            accumulator_store,
            consensus_adapter,
            consensus_store,
            committee_store,
        );
        Self {
            consensus_config: consensus_config.clone(),
            mysticeti_manager,
            mysticeti_client,
            active: parking_lot::Mutex::new(false),
            consensus_client,
        }
    }

    pub fn get_storage_base_path(&self) -> PathBuf {
        self.consensus_config.db_path().to_path_buf()
    }
}

#[async_trait]
impl ConsensusManagerTrait for ConsensusManager {
    async fn start(
        &self,
        node_config: &NodeConfig,
        epoch_store: Arc<AuthorityPerEpochStore>,
        consensus_handler_initializer: ConsensusHandlerInitializer,
        tx_validator: TxValidator,
    ) {
        let protocol_manager = {
            let mut active = self.active.lock();
            assert!(
                !*active,
                "Cannot start consensus. ConsensusManager is already running"
            );
            info!("Starting Mysticeti consensus protocol ...");
            *active = true;
            self.consensus_client.set(self.mysticeti_client.clone());
            &self.mysticeti_manager
        };

        protocol_manager
            .start(
                node_config,
                epoch_store,
                consensus_handler_initializer,
                tx_validator,
            )
            .await
    }

    async fn shutdown(&self) {
        info!("Shutting down consensus ...");
        let prev_active = {
            let mut active = self.active.lock();
            std::mem::replace(&mut *active, false)
        };
        if prev_active {
            self.mysticeti_manager.shutdown().await;
        }
        self.consensus_client.clear();
    }

    async fn is_running(&self) -> bool {
        let active = self.active.lock();
        *active
    }
}

#[derive(Default)]
pub struct ConsensusClient {
    // An extra layer of Arc<> is needed as required by ArcSwapAny.
    client: ArcSwapOption<Arc<dyn SubmitToConsensus>>,
}

impl ConsensusClient {
    pub fn new() -> Self {
        Self {
            client: ArcSwapOption::empty(),
        }
    }

    async fn get(&self) -> Arc<Arc<dyn SubmitToConsensus>> {
        const START_TIMEOUT: Duration = Duration::from_secs(30);
        const RETRY_INTERVAL: Duration = Duration::from_millis(100);
        if let Ok(client) = timeout(START_TIMEOUT, async {
            loop {
                let Some(client) = self.client.load_full() else {
                    sleep(RETRY_INTERVAL).await;
                    continue;
                };
                return client;
            }
        })
        .await
        {
            return client;
        }

        panic!(
            "Timed out after {:?} waiting for Consensus to start!",
            START_TIMEOUT,
        );
    }

    pub fn set(&self, client: Arc<dyn SubmitToConsensus>) {
        self.client.store(Some(Arc::new(client)));
    }

    pub fn clear(&self) {
        self.client.store(None);
    }
}

#[async_trait]
impl SubmitToConsensus for ConsensusClient {
    async fn submit_to_consensus(
        &self,
        transactions: &[ConsensusTransaction],
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        let client = self.get().await;
        client.submit_to_consensus(transactions, epoch_store).await
    }
}
