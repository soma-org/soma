use std::{sync::Arc, time::Instant};

use crate::{
    block_manager::BlockManager,
    broadcaster::Broadcaster,
    commit_observer::{CommitConsumer, CommitObserver},
    commit_syncer::{CommitSyncer, CommitVoteMonitor},
    core::{Core, CoreSignals},
    core_thread::{ChannelCoreThreadDispatcher, CoreThreadDispatcher, CoreThreadHandle},
    dag_state::DagState,
    leader_schedule::LeaderSchedule,
    leader_timeout::{LeaderTimeoutTask, LeaderTimeoutTaskHandle},
    network::{
        tonic_network::{TonicClient, TonicManager},
        NetworkManager, NetworkService,
    },
    service::AuthorityService,
    synchronizer::{Synchronizer, SynchronizerHandle},
};
use parking_lot::RwLock;
use tracing::info;
use types::{
    accumulator,
    committee::{AuthorityIndex, Committee},
    crypto::AuthorityKeyPair,
    storage::consensus::ConsensusStore,
};
use types::{
    accumulator::AccumulatorStore,
    crypto::{NetworkKeyPair, ProtocolKeyPair},
};
use types::{consensus::EndOfEpochAPI, parameters::Parameters};
use types::{
    consensus::{
        block_verifier::SignedBlockVerifier,
        context::{Clock, Context},
        transaction::{TransactionClient, TransactionConsumer, TransactionVerifier},
    },
    storage::consensus::mem_store::MemStore,
};

pub struct ConsensusAuthority {
    context: Arc<Context>,
    start_time: Instant,
    transaction_client: Arc<TransactionClient>,
    synchronizer: Arc<SynchronizerHandle>,
    commit_syncer: CommitSyncer<TonicClient>,
    core_thread_handle: CoreThreadHandle,
    leader_timeout_handle: LeaderTimeoutTaskHandle,
    broadcaster: Broadcaster,
    network_manager: TonicManager,
}

impl ConsensusAuthority {
    pub async fn start(
        own_index: AuthorityIndex,
        committee: Committee,
        parameters: Parameters,
        // To avoid accidentally leaking the private key, the protocol key pair should only be
        // kept in Core.
        protocol_keypair: ProtocolKeyPair,
        network_keypair: NetworkKeyPair,
        authority_keypair: AuthorityKeyPair,
        transaction_verifier: Arc<dyn TransactionVerifier>,
        commit_consumer: CommitConsumer,
        accumulator_store: Arc<dyn AccumulatorStore>,
        epoch_store: Arc<dyn EndOfEpochAPI>,
        consensus_store: Arc<dyn ConsensusStore>,
    ) -> Self {
        info!(
            "Starting consensus authority {}\n{:#?}\n{:#?}",
            own_index, committee, parameters
        );
        assert!(committee.is_valid_index(own_index));
        let context = Arc::new(Context::new(
            Some(own_index),
            committee,
            parameters,
            Arc::new(Clock::new()),
        ));

        let start_time = Instant::now();

        let (tx_client, tx_receiver) = TransactionClient::new(context.clone());
        let tx_consumer = TransactionConsumer::new(tx_receiver, context.clone(), None);

        let (core_signals, signals_receivers) = CoreSignals::new(context.clone());

        let mut network_manager = TonicManager::new(context.clone(), network_keypair);
        let network_client: Arc<TonicClient> = <TonicManager as NetworkManager<
            AuthorityService<ChannelCoreThreadDispatcher>,
        >>::client(&network_manager);

        // REQUIRED: Broadcaster must be created before Core, to start listening on the
        // broadcast channel in order to not miss blocks and cause test failures.
        let broadcaster =
            Broadcaster::new(context.clone(), network_client.clone(), &signals_receivers);

        // let store = Arc::new(MemStore::new());
        // let store = Arc::new(RocksDBStore::new(store_path));
        let dag_state = Arc::new(RwLock::new(DagState::new(
            context.clone(),
            consensus_store.clone(),
        )));

        let block_verifier = Arc::new(SignedBlockVerifier::new(
            context.clone(),
            transaction_verifier,
            accumulator_store.clone(),
            None,
        ));

        let block_manager = BlockManager::new(dag_state.clone(), block_verifier.clone());

        let leader_schedule = Arc::new(LeaderSchedule::new(context.clone()));

        let commit_observer = CommitObserver::new(
            context.clone(),
            commit_consumer,
            dag_state.clone(),
            consensus_store.clone(),
            leader_schedule.clone(),
        );

        let core = Core::new(
            context.clone(),
            leader_schedule,
            tx_consumer,
            block_manager,
            // For streaming RPC, Core will be notified when consumer is available.
            // For non-streaming RPC, there is no way to know so default to true.
            commit_observer,
            core_signals,
            protocol_keypair,
            authority_keypair,
            dag_state.clone(),
            epoch_store,
        );

        let (core_dispatcher, core_thread_handle) =
            ChannelCoreThreadDispatcher::start(core, context.clone());
        let core_dispatcher = Arc::new(core_dispatcher);
        let leader_timeout_handle =
            LeaderTimeoutTask::start(core_dispatcher.clone(), &signals_receivers, context.clone());

        let commit_vote_monitor = Arc::new(CommitVoteMonitor::new(context.clone()));
        let synchronizer = Synchronizer::start(
            network_client.clone(),
            context.clone(),
            core_dispatcher.clone(),
            commit_vote_monitor.clone(),
            block_verifier.clone(),
            dag_state.clone(),
        );

        let commit_syncer = CommitSyncer::new(
            context.clone(),
            core_dispatcher.clone(),
            commit_vote_monitor.clone(),
            network_client.clone(),
            block_verifier.clone(),
            dag_state.clone(),
        );

        let network_service = Arc::new(AuthorityService::new(
            context.clone(),
            core_dispatcher,
            block_verifier,
            commit_vote_monitor,
            synchronizer.clone(),
            dag_state.clone(),
            consensus_store,
        ));

        network_manager.install_service(network_service).await;

        info!(
            "Consensus authority started, took {:?}",
            start_time.elapsed()
        );

        Self {
            context,
            start_time,
            transaction_client: Arc::new(tx_client),
            synchronizer,
            commit_syncer,
            leader_timeout_handle,
            core_thread_handle,
            broadcaster,
            network_manager,
        }
    }

    pub async fn stop(mut self) {
        info!(
            "Stopping authority. Total run time: {:?}",
            self.start_time.elapsed()
        );

        // First shutdown components calling into Core.
        self.synchronizer.stop().await.ok();
        self.commit_syncer.stop().await;
        self.leader_timeout_handle.stop().await;
        // Shutdown Core to stop block productions and broadcast.
        // When using streaming, all subscribers to broadcasted blocks stop after this.
        self.core_thread_handle.stop().await;
        self.broadcaster.stop();

        <TonicManager as NetworkManager<AuthorityService<ChannelCoreThreadDispatcher>>>::stop(
            &mut self.network_manager,
        )
        .await;
    }

    pub fn transaction_client(&self) -> Arc<TransactionClient> {
        self.transaction_client.clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::sync::Mutex;
    use std::{collections::BTreeSet, sync::Arc, time::Duration};

    use accumulator::TestAccumulatorStore;
    use fastcrypto::traits::KeyPair;
    use tempfile::TempDir;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
    use tokio::time::sleep;
    use types::consensus::TestEpochStore;
    use types::parameters::Parameters;

    use crate::authority;

    use super::*;
    use types::consensus::{
        block::BlockAPI as _, commit::CommittedSubDag, committee::local_committee_and_keys,
        transaction::NoopTransactionVerifier,
    };

    #[tokio::test]
    async fn test_authority_start_and_stop() {
        let (committee, keypairs, authority_keypairs) = local_committee_and_keys(0, vec![1]);

        let temp_dir = TempDir::new().unwrap();
        let parameters = Parameters {
            db_path: temp_dir.into_path(),
            ..Default::default()
        };
        let txn_verifier = NoopTransactionVerifier {};

        let own_index = committee.to_authority_index(0).unwrap();
        let protocol_keypair = keypairs[own_index].1.clone();
        let network_keypair = keypairs[own_index].0.clone();
        let authority_keypair = authority_keypairs[own_index].copy();

        let (sender, _receiver) = unbounded_channel();
        let commit_consumer = CommitConsumer::new(sender, 0, 0);
        let store = Arc::new(MemStore::new());

        let authority = ConsensusAuthority::start(
            own_index,
            committee,
            parameters,
            protocol_keypair,
            network_keypair,
            authority_keypair,
            Arc::new(txn_verifier),
            commit_consumer,
            Arc::new(TestAccumulatorStore::default()),
            Arc::new(TestEpochStore::new()),
            store.clone(),
        )
        .await;

        assert_eq!(authority.context.own_index, Some(own_index));
        assert_eq!(authority.context.committee.epoch(), 0);
        assert_eq!(authority.context.committee.size(), 1);

        authority.stop().await;
    }

    // TODO: build AuthorityFixture.
    #[tokio::test(flavor = "current_thread")]
    async fn test_authority_committee() {
        let (committee, keypairs, authority_keypairs) =
            local_committee_and_keys(0, vec![1, 1, 1, 1]);
        let temp_dirs = (0..4).map(|_| TempDir::new().unwrap()).collect::<Vec<_>>();

        let mut output_receivers = Vec::with_capacity(committee.size());
        let mut authorities = Vec::with_capacity(committee.size());

        for (index, _authority_info) in committee.authorities() {
            let (authority, receiver) = make_authority(
                index,
                &temp_dirs[index.value()],
                committee.clone(),
                keypairs.clone(),
                authority_keypairs[index.value()].copy(),
            )
            .await;
            output_receivers.push(receiver);
            authorities.push(authority);
        }

        const NUM_TRANSACTIONS: u8 = 15;
        let mut submitted_transactions = BTreeSet::<Vec<u8>>::new();
        for i in 0..NUM_TRANSACTIONS {
            let txn = vec![i; 16];
            submitted_transactions.insert(txn.clone());
            authorities[i as usize % authorities.len()]
                .transaction_client()
                .submit(vec![txn])
                .await
                .unwrap();
        }

        for receiver in &mut output_receivers {
            let mut expected_transactions = submitted_transactions.clone();
            loop {
                let committed_subdag =
                    tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                        .await
                        .unwrap()
                        .unwrap();
                for b in committed_subdag.blocks {
                    for txn in b.transactions().iter().map(|t| t.data().to_vec()) {
                        assert!(
                            expected_transactions.remove(&txn),
                            "Transaction not submitted or already seen: {:?}",
                            txn
                        );
                    }
                }

                if expected_transactions.is_empty() {
                    break;
                }
            }
        }

        // Stop authority 1.
        let index = committee.to_authority_index(1).unwrap();
        authorities.remove(index.value()).stop().await;
        sleep(Duration::from_secs(15)).await;

        // Restart authority 1 and let it run.
        let (authority, receiver) = make_authority(
            index,
            &temp_dirs[index.value()],
            committee.clone(),
            keypairs,
            authority_keypairs[index.value()].copy(),
        )
        .await;
        output_receivers[index] = receiver;
        authorities.insert(index.value(), authority);
        sleep(Duration::from_secs(15)).await;

        // Stop all authorities and exit.
        for authority in authorities {
            authority.stop().await;
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_amnesia_success() {
        let _ = tracing_subscriber::fmt::try_init();

        let (committee, keypairs, authority_keypairs) =
            local_committee_and_keys(0, vec![1, 1, 1, 1]);
        let mut output_receivers = vec![];
        let mut authorities = vec![];

        for (index, _authority_info) in committee.authorities() {
            let (authority, receiver) = make_authority(
                index,
                &TempDir::new().unwrap(),
                committee.clone(),
                keypairs.clone(),
                authority_keypairs[index.value()].copy(),
            )
            .await;
            output_receivers.push(receiver);
            authorities.push(authority);
        }

        const NUM_TRANSACTIONS: u8 = 15;
        let mut submitted_transactions = BTreeSet::<Vec<u8>>::new();
        for i in 0..NUM_TRANSACTIONS {
            let txn = vec![i; 16];
            submitted_transactions.insert(txn.clone());
            authorities[i as usize % authorities.len()]
                .transaction_client()
                .submit(vec![txn])
                .await
                .unwrap();
        }

        for receiver in &mut output_receivers {
            let mut expected_transactions = submitted_transactions.clone();
            loop {
                let committed_subdag =
                    tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                        .await
                        .unwrap()
                        .unwrap();
                info!("Received committed subdag: {:?}", committed_subdag);
                for b in committed_subdag.blocks {
                    for txn in b.transactions().iter().map(|t| t.data().to_vec()) {
                        assert!(
                            expected_transactions.remove(&txn),
                            "Transaction not submitted or already seen: {:?}",
                            txn
                        );
                    }
                }

                if expected_transactions.is_empty() {
                    break;
                }
            }
        }

        // Stop authority 1.
        let index = committee.to_authority_index(1).unwrap();
        authorities.remove(index.value()).stop().await;
        sleep(Duration::from_secs(5)).await;

        // now create a new directory to simulate amnesia. The node will start having participated previously
        // to consensus but now will attempt to synchronize the last own block and recover from there.
        let (authority, mut receiver) = make_authority(
            index,
            &TempDir::new().unwrap(),
            committee.clone(),
            keypairs,
            authority_keypairs[index.value()].copy(),
        )
        .await;
        authorities.insert(index.value(), authority);
        sleep(Duration::from_secs(5)).await;

        // We wait until we see at least one committed block authored from this authority
        'outer: while let Some(result) = receiver.recv().await {
            for block in result.blocks {
                if block.author() == index {
                    break 'outer;
                }
            }
        }

        // Stop all authorities and exit.
        for authority in authorities {
            authority.stop().await;
        }
    }

    #[tokio::test]
    async fn test_amnesia_failure() {
        let _ = tracing_subscriber::fmt::try_init();

        let occurred_panic = Arc::new(Mutex::new(None));
        let occurred_panic_cloned = occurred_panic.clone();

        let default_panic_handler = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let mut l = occurred_panic_cloned.lock().unwrap();
            *l = Some(panic.to_string());
            default_panic_handler(panic);
        }));

        let (committee, keypairs, authority_keypairs) =
            local_committee_and_keys(0, vec![1, 1, 1, 1]);
        let mut output_receivers = vec![];
        let mut authorities = vec![];

        for (index, _authority_info) in committee.authorities() {
            let (authority, receiver) = make_authority(
                index,
                &TempDir::new().unwrap(),
                committee.clone(),
                keypairs.clone(),
                authority_keypairs[index.value()].copy(),
            )
            .await;
            output_receivers.push(receiver);
            authorities.push(authority);
        }

        // Let the network run for a few seconds
        sleep(Duration::from_secs(5)).await;

        // Stop all authorities
        while let Some(authority) = authorities.pop() {
            authority.stop().await;
        }

        sleep(Duration::from_secs(2)).await;

        let index = AuthorityIndex::new_for_test(0);
        let (_authority, _receiver) = make_authority(
            index,
            &TempDir::new().unwrap(),
            committee,
            keypairs,
            authority_keypairs[index.value()].copy(),
        )
        .await;
        sleep(Duration::from_secs(5)).await;

        // Now reset the panic hook
        let _default_panic_handler = std::panic::take_hook();

        // We expect this test to panic as all the other peers are down and the node that tries to
        // recover its last produced block fails.
        let panic_info = occurred_panic.lock().unwrap().take().unwrap();
        assert!(panic_info.contains(
            "No peer has returned any acceptable result, can not safely update min round"
        ));
    }

    // TODO: create a fixture
    async fn make_authority(
        index: AuthorityIndex,
        db_dir: &TempDir,
        committee: Committee,
        keypairs: Vec<(NetworkKeyPair, ProtocolKeyPair)>,
        authority_keypair: AuthorityKeyPair,
    ) -> (ConsensusAuthority, UnboundedReceiver<CommittedSubDag>) {
        // Cache less blocks to exercise commit sync.
        let parameters = Parameters {
            db_path: db_dir.path().to_path_buf(),
            dag_state_cached_rounds: 5,
            commit_sync_parallel_fetches: 3,
            commit_sync_batch_size: 3,
            sync_last_proposed_block_timeout: Duration::from_millis(2_000),
            ..Default::default()
        };
        let txn_verifier = NoopTransactionVerifier {};

        let protocol_keypair = keypairs[index].1.clone();
        let network_keypair = keypairs[index].0.clone();

        let (sender, receiver) = unbounded_channel();
        let commit_consumer = CommitConsumer::new(sender, 0, 0);
        let store = Arc::new(MemStore::new());

        let authority = ConsensusAuthority::start(
            index,
            committee,
            parameters,
            protocol_keypair,
            network_keypair,
            authority_keypair,
            Arc::new(txn_verifier),
            commit_consumer,
            Arc::new(TestAccumulatorStore::default()),
            Arc::new(TestEpochStore::new()),
            store.clone(),
        )
        .await;
        (authority, receiver)
    }
}
