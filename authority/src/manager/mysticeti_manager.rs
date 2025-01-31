use crate::{
    adapter::ConsensusAdapter,
    epoch_store::AuthorityPerEpochStore,
    handler::{ConsensusHandlerInitializer, MysticetiConsensusHandler},
    tx_validator::TxValidator,
};
use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use consensus::{CommitConsumer, ConsensusAuthority};
use fastcrypto::ed25519::Ed25519KeyPair;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::Mutex;
use tracing::info;
use types::consensus::{block::Round, commit::CommitIndex};
use types::{accumulator, parameters::Parameters};
use types::{
    accumulator::AccumulatorStore,
    crypto::{NetworkKeyPair, ProtocolKeyPair},
    system_state::EpochStartSystemStateTrait,
};
use types::{committee::EpochId, config::node_config::NodeConfig};

use super::{mysticeti_client::LazyMysticetiClient, ConsensusManagerTrait, Running};

pub struct MysticetiManager {
    protocol_keypair: ProtocolKeyPair,
    network_keypair: NetworkKeyPair,
    storage_base_path: PathBuf,
    running: Mutex<Running>,
    authority: ArcSwapOption<ConsensusAuthority>,
    // Use a shared lazy mysticeti client so we can update the internal mysticeti
    // client that gets created for every new epoch.
    client: Arc<LazyMysticetiClient>,
    consensus_handler: Mutex<Option<MysticetiConsensusHandler>>,
    accumulator_store: Arc<dyn AccumulatorStore>,
    consensus_adapter: Arc<ConsensusAdapter>,
}

impl MysticetiManager {
    /// NOTE: Mysticeti protocol key uses Ed25519 instead of BLS.
    /// But for security, the protocol keypair must be different from the network keypair.
    pub fn new(
        protocol_keypair: Ed25519KeyPair,
        network_keypair: Ed25519KeyPair,
        storage_base_path: PathBuf,
        client: Arc<LazyMysticetiClient>,
        accumulator_store: Arc<dyn AccumulatorStore>,
        consensus_adapter: Arc<ConsensusAdapter>,
    ) -> Self {
        Self {
            protocol_keypair: ProtocolKeyPair::new(protocol_keypair),
            network_keypair: NetworkKeyPair::new(network_keypair),
            storage_base_path,
            running: Mutex::new(Running::False),
            authority: ArcSwapOption::empty(),
            client,
            consensus_handler: Mutex::new(None),
            accumulator_store,
            consensus_adapter,
        }
    }

    fn get_store_path(&self, epoch: EpochId) -> PathBuf {
        let mut store_path = self.storage_base_path.clone();
        store_path.push(format!("{}", epoch));
        store_path
    }
}

#[async_trait]
impl ConsensusManagerTrait for MysticetiManager {
    async fn start(
        &self,
        config: &NodeConfig,
        epoch_store: Arc<AuthorityPerEpochStore>,
        consensus_handler_initializer: ConsensusHandlerInitializer,
        tx_validator: TxValidator,
    ) {
        let system_state = epoch_store.epoch_start_state();
        let committee = system_state.get_committee();
        let epoch = epoch_store.epoch();
        let protocol_config = epoch_store.protocol_config();

        let consensus_config = config
            .consensus_config()
            .expect("consensus_config should exist");
        let parameters = Parameters {
            db_path: self.get_store_path(epoch),
            ..consensus_config.parameters.clone().unwrap_or_default()
        };

        let own_protocol_key = self.protocol_keypair.public();
        let (own_index, _) = committee
            .authorities()
            .find(|(_, a)| a.protocol_key == own_protocol_key)
            .expect("Own authority should be among the consensus authorities!");

        let (commit_sender, commit_receiver) = unbounded_channel();

        let consensus_handler = consensus_handler_initializer.new_consensus_handler();
        let consumer = CommitConsumer::new(
            commit_sender,
            consensus_handler.last_executed_sub_dag_round() as Round,
            consensus_handler.last_executed_sub_dag_index() as CommitIndex,
        );

        let authority = ConsensusAuthority::start(
            own_index,
            committee.clone(),
            parameters.clone(),
            self.protocol_keypair.clone(),
            self.network_keypair.clone(),
            Arc::new(tx_validator.clone()),
            consumer,
            self.accumulator_store.clone(),
        )
        .await;
        let client = authority.transaction_client();

        self.authority.swap(Some(Arc::new(authority)));

        // Initialize the client to send transactions to this Mysticeti instance.
        self.client.set(client);

        // spin up the new mysticeti consensus handler to listen for committed sub dags
        let handler = MysticetiConsensusHandler::new(
            consensus_handler,
            commit_receiver,
            self.consensus_adapter.clone(),
        );
        let mut consensus_handler = self.consensus_handler.lock().await;
        *consensus_handler = Some(handler);
    }

    async fn shutdown(&self) {
        // Stop consensus submissions.
        self.client.clear();

        // swap with empty to ensure there is no other reference to authority and we can safely do Arc unwrap
        let r = self.authority.swap(None).unwrap();
        let Ok(authority) = Arc::try_unwrap(r) else {
            panic!("Failed to retrieve the mysticeti authority");
        };

        // shutdown the authority and wait for it
        authority.stop().await;

        // drop the old consensus handler to force stop any underlying task running.
        let mut consensus_handler = self.consensus_handler.lock().await;
        if let Some(mut handler) = consensus_handler.take() {
            handler.abort().await;
        }
    }

    async fn is_running(&self) -> bool {
        Running::False != *self.running.lock().await
    }
}
