// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/simtests/tests/consensus_tests.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the Soma project.

//! Integration tests for the consensus authority using the msim deterministic simulator.
//!
//! These tests spin up multi-node consensus committees with simulated networking,
//! submit transactions, and verify they are committed across all authorities.
//! Unlike the unit tests in authority_node.rs (which use real tonic networking and
//! are limited to 1-2 node committees), these tests use msim's simulated network
//! to reliably test 4-10 node committees.
//!
//! Ported from Sui's `consensus/simtests/tests/consensus_tests.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use protocol_config::{Chain, ProtocolConfig, ProtocolVersion};
use tempfile::TempDir;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::watch;
use tokio::time::{sleep, timeout};
use tracing::info;
use types::committee::{AuthorityIndex, Committee};
use types::consensus::block::{BlockAPI, BlockTimestampMs, GENESIS_ROUND};
use types::consensus::commit::CommittedSubDag;
use types::consensus::context::Clock;
use types::crypto::{NetworkKeyPair, ProtocolKeyPair};
use types::parameters::Parameters;

use crate::commit_consumer::CommitConsumerArgs;
use crate::network::tonic_network::to_socket_addr;
use crate::transaction::NoopTransactionVerifier;
use crate::{CommitConsumerMonitor, ConsensusAuthority, NetworkType, TransactionClient};

// ---------------------------------------------------------------------------
// AuthorityNode wrapper — manages a ConsensusAuthority inside an msim node
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SimtestConfig {
    authority_index: AuthorityIndex,
    db_dir: Arc<TempDir>,
    committee: Committee,
    keypairs: Vec<(NetworkKeyPair, ProtocolKeyPair)>,
    boot_counter: u64,
    clock_drift: BlockTimestampMs,
    protocol_config: ProtocolConfig,
}

struct AuthorityNode {
    inner: Mutex<Option<AuthorityNodeInner>>,
    config: SimtestConfig,
    commit_receiver: Mutex<Option<UnboundedReceiver<CommittedSubDag>>>,
}

struct AuthorityNodeInner {
    handle: Option<NodeHandle>,
    cancel_sender: Option<watch::Sender<bool>>,
    consensus_authority: ConsensusAuthority,
    commit_receiver: Option<UnboundedReceiver<CommittedSubDag>>,
    commit_consumer_monitor: Arc<CommitConsumerMonitor>,
}

#[derive(Debug)]
struct NodeHandle {
    node_id: msim::task::NodeId,
}

impl Drop for AuthorityNodeInner {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            tracing::info!("shutting down msim node {}", handle.node_id);
            msim::runtime::Handle::try_current().map(|h| h.delete_node(handle.node_id));
        }
    }
}

impl AuthorityNode {
    fn new(config: SimtestConfig) -> Self {
        Self { inner: Mutex::new(None), config, commit_receiver: Mutex::new(None) }
    }

    fn index(&self) -> AuthorityIndex {
        self.config.authority_index
    }

    async fn start(&self) {
        info!(index = %self.config.authority_index, "starting simtest consensus node");
        let config = self.config.clone();
        *self.inner.lock() = Some(AuthorityNodeInner::spawn(config).await);
    }

    fn spawn_committed_subdag_consumer(&self) {
        let authority_index = self.config.authority_index;
        let mut inner = self.inner.lock();
        if let Some(inner) = inner.as_mut() {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            *self.commit_receiver.lock() = Some(rx);

            let mut commit_receiver =
                inner.commit_receiver.take().expect("commit receiver already taken");
            let commit_consumer_monitor = inner.commit_consumer_monitor.clone();

            tokio::spawn(async move {
                while let Some(subdag) = commit_receiver.recv().await {
                    info!(
                        authority = %authority_index,
                        commit_index = %subdag.commit_ref.index,
                        "Received committed subdag"
                    );
                    commit_consumer_monitor.set_highest_handled_commit(subdag.commit_ref.index);
                    let _ = tx.send(subdag);
                }
            });
        }
    }

    fn commit_consumer_monitor(&self) -> Arc<CommitConsumerMonitor> {
        let inner = self.inner.lock();
        inner.as_ref().expect("Node not initialised").commit_consumer_monitor.clone()
    }

    fn commit_consumer_receiver(&self) -> UnboundedReceiver<CommittedSubDag> {
        self.commit_receiver.lock().take().expect("No commit consumer receiver found")
    }

    fn transaction_client(&self) -> Arc<TransactionClient> {
        let inner = self.inner.lock();
        inner.as_ref().expect("Node not initialised").consensus_authority.transaction_client()
    }

    fn stop(&self) {
        info!(index = %self.config.authority_index, "stopping simtest consensus node");
        *self.inner.lock() = None;
        info!(index = %self.config.authority_index, "node stopped");
    }
}

impl AuthorityNodeInner {
    async fn spawn(config: SimtestConfig) -> Self {
        let (startup_sender, mut startup_receiver) = watch::channel(false);
        let (cancel_sender, cancel_receiver) = watch::channel(false);

        let handle = msim::runtime::Handle::current();
        let builder = handle.create_node();

        let authority = config
            .committee
            .authority_by_authority_index(config.authority_index)
            .expect("Authority index not in committee");
        let socket_addr = to_socket_addr(&authority.address).unwrap();
        let ip = match socket_addr {
            SocketAddr::V4(v4) => IpAddr::V4(*v4.ip()),
            _ => panic!("unsupported protocol"),
        };

        // Transfer authority + receiver out of the msim node closure via shared slot.
        let result_slot: Arc<
            Mutex<
                Option<(
                    ConsensusAuthority,
                    UnboundedReceiver<CommittedSubDag>,
                    Arc<CommitConsumerMonitor>,
                )>,
            >,
        > = Arc::new(Mutex::new(None));
        let result_slot_clone = result_slot.clone();

        let node = builder
            .ip(ip)
            .name(format!("{}", config.authority_index))
            .init(move || {
                info!("Consensus simtest node init");
                let config = config.clone();
                let mut cancel_receiver = cancel_receiver.clone();
                let result_slot = result_slot_clone.clone();
                let startup_sender = startup_sender.clone();

                async move {
                    let (authority, commit_receiver, monitor) = make_authority(config).await;

                    *result_slot.lock() = Some((authority, commit_receiver, monitor));
                    startup_sender.send(true).ok();

                    // Run until canceled.
                    loop {
                        if cancel_receiver.changed().await.is_err() || *cancel_receiver.borrow() {
                            break;
                        }
                    }
                    tracing::trace!("cancellation received; shutting down simtest node");
                }
            })
            .build();

        startup_receiver.changed().await.unwrap();

        let (consensus_authority, commit_receiver, commit_consumer_monitor) =
            result_slot.lock().take().expect("Components should be initialised by now");

        Self {
            handle: Some(NodeHandle { node_id: node.id() }),
            cancel_sender: Some(cancel_sender),
            consensus_authority,
            commit_receiver: Some(commit_receiver),
            commit_consumer_monitor,
        }
    }
}

async fn make_authority(
    config: SimtestConfig,
) -> (ConsensusAuthority, UnboundedReceiver<CommittedSubDag>, Arc<CommitConsumerMonitor>) {
    let SimtestConfig {
        authority_index,
        db_dir,
        committee,
        keypairs,
        boot_counter,
        protocol_config,
        clock_drift,
    } = config;

    let parameters = Parameters {
        db_path: db_dir.path().to_path_buf(),
        dag_state_cached_rounds: 5,
        commit_sync_parallel_fetches: 2,
        commit_sync_batch_size: 3,
        sync_last_known_own_block_timeout: Duration::from_millis(2_000),
        ..Default::default()
    };

    let protocol_keypair = keypairs[authority_index.value()].1.clone();
    let network_keypair = keypairs[authority_index.value()].0.clone();

    let (commit_consumer, commit_receiver, _blocks_receiver) = CommitConsumerArgs::new(0, 0);
    let commit_consumer_monitor = commit_consumer.monitor();

    let authority = ConsensusAuthority::start(
        NetworkType::Tonic,
        0,
        authority_index,
        committee,
        parameters,
        protocol_config,
        protocol_keypair,
        network_keypair,
        Arc::new(Clock::new_for_test(clock_drift)),
        Arc::new(NoopTransactionVerifier {}),
        commit_consumer,
        boot_counter,
    )
    .await;

    (authority, commit_receiver, commit_consumer_monitor)
}

// ---------------------------------------------------------------------------
// Committee creation with unique IPs + unique ports for msim
// ---------------------------------------------------------------------------

/// Creates a committee with unique IPs (10.10.0.{i+1}) per authority for msim.
/// Each authority also gets a unique ephemeral port to avoid bind conflicts
/// on the real OS (std::net::TcpListener::bind operates on real sockets).
fn simtest_committee_and_keys(
    num_authorities: usize,
) -> (Committee, Vec<(NetworkKeyPair, ProtocolKeyPair)>) {
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::BTreeMap;
    use types::base::AuthorityName;
    use types::committee::{Authority, Stake, get_available_local_address};
    use types::crypto::{AuthorityKeyPair, KeypairTraits};
    use types::multiaddr::Multiaddr;

    let mut authorities = BTreeMap::new();
    let mut voting_weights = BTreeMap::new();
    // Store keypairs by AuthorityName so they stay aligned after BTreeMap sorting.
    let mut key_pairs_by_name: BTreeMap<AuthorityName, (NetworkKeyPair, ProtocolKeyPair)> =
        BTreeMap::new();
    let mut rng = StdRng::from_seed([0; 32]);

    for i in 0..num_authorities {
        let authority_keypair = AuthorityKeyPair::generate(&mut rng);
        let protocol_keypair = ProtocolKeyPair::generate(&mut rng);
        let network_keypair = NetworkKeyPair::generate(&mut rng);
        let name = AuthorityName::from(authority_keypair.public());

        // Get an available port from the OS to avoid bind conflicts.
        let local_addr = get_available_local_address();
        let port = types::multiaddr::Multiaddr::to_socket_addr(&local_addr).unwrap().port();
        // Unique IP per authority for msim node identity.
        let addr: Multiaddr = format!("/ip4/10.10.0.{}/tcp/{}", i + 1, port).parse().unwrap();

        authorities.insert(
            name,
            Authority {
                stake: 1,
                address: addr,
                hostname: format!("test_host_{i}"),
                authority_key: authority_keypair.public().clone(),
                protocol_key: protocol_keypair.public(),
                network_key: network_keypair.public(),
            },
        );
        voting_weights.insert(name, 1 as Stake);
        key_pairs_by_name.insert(name, (network_keypair, protocol_keypair));
    }

    let committee =
        Committee::new_for_testing_with_normalized_voting_power(0, voting_weights, authorities);

    // Build key_pairs Vec in committee authority-index order so that
    // keypairs[authority_index] is always the correct keypair.
    let key_pairs: Vec<_> = committee
        .authorities()
        .map(|(_index, authority)| {
            let name = AuthorityName::from(&authority.authority_key);
            key_pairs_by_name.remove(&name).expect("keypair must exist for every authority")
        })
        .collect();

    (committee, key_pairs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: Start a 10-node committee, submit transactions, wait for commits,
/// then start a late-joining node and verify it catches up.
#[msim::sim_test]
async fn test_committee_start_simple() {
    utils::logging::init_tracing();

    const NUM_OF_AUTHORITIES: usize = 10;
    let (committee, keypairs) = simtest_committee_and_keys(NUM_OF_AUTHORITIES);
    let protocol_config = ProtocolConfig::get_for_version(ProtocolVersion::max(), Chain::Unknown);

    let mut authorities = Vec::with_capacity(committee.size());
    let mut transaction_clients = Vec::with_capacity(committee.size());
    let mut boot_counters = vec![0u64; NUM_OF_AUTHORITIES];
    let mut clock_drifts = vec![0u64; NUM_OF_AUTHORITIES];
    // Introduce non-trivial clock drift for a few nodes to exercise timestamp checks.
    clock_drifts[0] = 50;
    clock_drifts[1] = 100;
    clock_drifts[2] = 120;

    for (authority_index, _authority_info) in committee.authorities() {
        let config = SimtestConfig {
            authority_index,
            db_dir: Arc::new(TempDir::new().unwrap()),
            committee: committee.clone(),
            keypairs: keypairs.clone(),
            boot_counter: boot_counters[authority_index.value()],
            protocol_config: protocol_config.clone(),
            clock_drift: clock_drifts[authority_index.value()],
        };
        let node = AuthorityNode::new(config);

        // Start all except the last node (it will join late).
        if authority_index != AuthorityIndex::new_for_test(NUM_OF_AUTHORITIES as u32 - 1) {
            node.start().await;
            node.spawn_committed_subdag_consumer();

            let client = node.transaction_client();
            transaction_clients.push(client);
        }

        boot_counters[authority_index.value()] += 1;
        authorities.push(node);
    }

    // Submit transactions from a background task.
    let transaction_clients_clone = transaction_clients.clone();
    let _handle = tokio::spawn(async move {
        const NUM_TRANSACTIONS: u16 = 1000;
        for i in 0..NUM_TRANSACTIONS {
            let txn = vec![i as u8; 16];
            transaction_clients_clone[i as usize % transaction_clients_clone.len()]
                .submit(vec![txn])
                .await
                .unwrap();
        }
    });

    // Wait for authorities to make progress.
    sleep(Duration::from_secs(60)).await;

    // Now start the last authority.
    tracing::info!(
        authority = %NUM_OF_AUTHORITIES - 1,
        "Starting authority and waiting for it to catch up"
    );
    authorities[NUM_OF_AUTHORITIES - 1].start().await;
    authorities[NUM_OF_AUTHORITIES - 1].spawn_committed_subdag_consumer();

    // Wait for it to catch up via commit sync.
    sleep(Duration::from_secs(230)).await;
    let commit_consumer_monitor = authorities[NUM_OF_AUTHORITIES - 1].commit_consumer_monitor();
    let highest_committed_index = commit_consumer_monitor.highest_handled_commit();
    assert!(highest_committed_index >= 80, "Highest handled commit {highest_committed_index} < 80");
}

/// Test: Start a 4-node committee, submit transactions, verify all committed,
/// then stop and restart an authority to verify recovery.
#[msim::sim_test]
async fn test_authority_committee_simtest() {
    utils::logging::init_tracing();

    const NUM_OF_AUTHORITIES: usize = 4;
    let (committee, keypairs) = simtest_committee_and_keys(NUM_OF_AUTHORITIES);
    let protocol_config = ProtocolConfig::get_for_version(ProtocolVersion::max(), Chain::Unknown);

    let temp_dirs: Vec<_> =
        (0..NUM_OF_AUTHORITIES).map(|_| Arc::new(TempDir::new().unwrap())).collect();

    let mut commit_receivers = Vec::with_capacity(committee.size());
    let mut authorities = Vec::with_capacity(committee.size());
    let mut boot_counters = vec![0u64; NUM_OF_AUTHORITIES];

    for (index, _authority_info) in committee.authorities() {
        let config = SimtestConfig {
            authority_index: index,
            db_dir: temp_dirs[index.value()].clone(),
            committee: committee.clone(),
            keypairs: keypairs.clone(),
            boot_counter: boot_counters[index.value()],
            protocol_config: protocol_config.clone(),
            clock_drift: 0,
        };
        let node = AuthorityNode::new(config);
        node.start().await;
        node.spawn_committed_subdag_consumer();
        commit_receivers.push(node.commit_consumer_receiver());
        boot_counters[index.value()] += 1;
        authorities.push(node);
    }

    // Submit transactions.
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

    // Verify all transactions are committed on every authority.
    for receiver in &mut commit_receivers {
        let mut expected_transactions = submitted_transactions.clone();
        loop {
            let committed_subdag = timeout(Duration::from_secs(30), receiver.recv())
                .await
                .expect("Timed out waiting for committed subdag")
                .unwrap();
            for b in committed_subdag.blocks {
                for txn in b.transactions().iter().map(|t| t.data().to_vec()) {
                    expected_transactions.remove(&txn);
                }
            }
            if expected_transactions.is_empty() {
                break;
            }
        }
    }

    // Stop authority 1.
    let index = AuthorityIndex::new_for_test(1);
    authorities[index.value()].stop();
    sleep(Duration::from_secs(10)).await;

    // Restart authority 1 with the same DB directory.
    let config = SimtestConfig {
        authority_index: index,
        db_dir: temp_dirs[index.value()].clone(),
        committee: committee.clone(),
        keypairs: keypairs.clone(),
        boot_counter: boot_counters[index.value()],
        protocol_config: protocol_config.clone(),
        clock_drift: 0,
    };
    let new_node = AuthorityNode::new(config);
    new_node.start().await;
    new_node.spawn_committed_subdag_consumer();
    boot_counters[index.value()] += 1;

    // Let the restarted node run for a bit.
    sleep(Duration::from_secs(10)).await;

    // Stop all.
    for authority in &authorities {
        authority.stop();
    }
    new_node.stop();
}

/// Test: Amnesia recovery — wipe a node's DB, restart, and verify it recovers
/// by syncing from peers.
#[msim::sim_test]
async fn test_amnesia_recovery_simtest() {
    utils::logging::init_tracing();

    const NUM_OF_AUTHORITIES: usize = 4;
    let (committee, keypairs) = simtest_committee_and_keys(NUM_OF_AUTHORITIES);
    let protocol_config = ProtocolConfig::get_for_version(ProtocolVersion::max(), Chain::Unknown);

    let mut commit_receivers = vec![];
    let mut authorities = BTreeMap::new();
    let mut temp_dirs = BTreeMap::new();
    let mut boot_counters = vec![0u64; NUM_OF_AUTHORITIES];

    for (index, _authority_info) in committee.authorities() {
        let dir = Arc::new(TempDir::new().unwrap());
        let config = SimtestConfig {
            authority_index: index,
            db_dir: dir.clone(),
            committee: committee.clone(),
            keypairs: keypairs.clone(),
            boot_counter: boot_counters[index.value()],
            protocol_config: protocol_config.clone(),
            clock_drift: 0,
        };
        let node = AuthorityNode::new(config);
        node.start().await;
        node.spawn_committed_subdag_consumer();
        commit_receivers.push(node.commit_consumer_receiver());
        boot_counters[index.value()] += 1;
        authorities.insert(index, node);
        temp_dirs.insert(index, dir);
    }

    // Wait until we see at least one committed block authored by authority 1.
    let index_1 = AuthorityIndex::new_for_test(1);
    'outer: while let Some(result) =
        timeout(Duration::from_secs(30), commit_receivers[index_1.value()].recv())
            .await
            .expect("Timed out waiting for at least one committed block from authority 1")
    {
        for block in result.blocks {
            if block.round() > GENESIS_ROUND && block.author() == index_1 {
                break 'outer;
            }
        }
    }

    // Stop authority 1 & 2.
    // * Authority 1 will have its DB wiped to force amnesia recovery.
    // * Authority 2 is stopped to simulate less than f+1 availability,
    //   making authority 1 retry during amnesia recovery.
    authorities.get(&index_1).unwrap().stop();
    let index_2 = AuthorityIndex::new_for_test(2);
    authorities.get(&index_2).unwrap().stop();
    sleep(Duration::from_secs(5)).await;

    // Authority 1: new temp directory = simulates amnesia. Reset boot counter.
    let new_dir = Arc::new(TempDir::new().unwrap());
    boot_counters[index_1.value()] = 0;
    let config = SimtestConfig {
        authority_index: index_1,
        db_dir: new_dir.clone(),
        committee: committee.clone(),
        keypairs: keypairs.clone(),
        boot_counter: boot_counters[index_1.value()],
        protocol_config: protocol_config.clone(),
        clock_drift: 0,
    };
    let node_1 = AuthorityNode::new(config);
    node_1.start().await;
    node_1.spawn_committed_subdag_consumer();
    let mut commit_receiver_1 = node_1.commit_consumer_receiver();
    boot_counters[index_1.value()] += 1;
    temp_dirs.insert(index_1, new_dir);
    sleep(Duration::from_secs(5)).await;

    // Now spin up authority 2 using its earlier directory — no amnesia recovery here.
    // Authority 1 should be able to recover from amnesia successfully.
    let config = SimtestConfig {
        authority_index: index_2,
        db_dir: temp_dirs[&index_2].clone(),
        committee: committee.clone(),
        keypairs: keypairs.clone(),
        boot_counter: boot_counters[index_2.value()],
        protocol_config: protocol_config.clone(),
        clock_drift: 0,
    };
    let node_2 = AuthorityNode::new(config);
    node_2.start().await;
    node_2.spawn_committed_subdag_consumer();
    boot_counters[index_2.value()] += 1;
    sleep(Duration::from_secs(5)).await;

    // Wait until we see at least one committed block authored by authority 1 (recovery).
    'outer2: while let Some(result) = timeout(Duration::from_secs(60), commit_receiver_1.recv())
        .await
        .expect("Timed out waiting for amnesia recovery of authority 1")
    {
        for block in result.blocks {
            if block.round() > GENESIS_ROUND && block.author() == index_1 {
                break 'outer2;
            }
        }
    }

    // Stop all.
    for (_, authority) in &authorities {
        authority.stop();
    }
    node_1.stop();
    node_2.stop();
}
