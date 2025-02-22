use std::{collections::HashSet, fs::File, sync::Once, time::Duration};

use futures::future::join_all;
use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use test_cluster::{
    config::genesis_config::{ValidatorGenesisConfig, ValidatorGenesisConfigBuilder},
    TestCluster, TestClusterBuilder,
};
use tokio::time::sleep;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::fmt;
use types::{
    base::SomaAddress,
    crypto::{KeypairTraits, PublicKey},
    system_state::SystemStateTrait,
    transaction::{
        AddValidatorArgs, RemoveValidatorArgs, StateTransaction, StateTransactionKind, Transaction,
        TransactionData, TransactionKind,
    },
};

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        // Open the file in write mode, which will truncate it if it already exists
        let file = File::create("test.log").expect("Failed to create log file");

        let subscriber = fmt::Subscriber::builder()
            .with_max_level(LevelFilter::DEBUG)
            // .with_env_filter(EnvFilter::from_default_env())
            .with_writer(file)
            .with_ansi(false)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");
    });
}

#[msim::sim_test]
async fn advance_epoch_tx_test() {
    let _ = tracing_subscriber::fmt::try_init();
    let test_cluster = TestClusterBuilder::new().build().await;
    let states = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|handle| handle.with(|node| node.state()))
        .collect::<Vec<_>>();
    let tasks: Vec<_> = states
        .iter()
        .map(|state| async {
            let system_state = state
                .create_and_execute_advance_epoch_tx(&state.epoch_store_for_testing(), 1000)
                .await
                .unwrap();
            system_state
        })
        .collect();
    let results: HashSet<_> = join_all(tasks)
        .await
        .into_iter()
        .map(|(state, _)| state.epoch())
        .collect();
    // Check that all validators have the same result.
    assert_eq!(results.len(), 1);
}

#[msim::sim_test]
async fn basic_reconfig_end_to_end_test() {
    let _ = tracing_subscriber::fmt::try_init();
    // TODO remove this sleep when this test passes consistently
    sleep(Duration::from_secs(1)).await;
    let test_cluster = TestClusterBuilder::new().build().await;
    test_cluster.trigger_reconfiguration().await;
}

#[msim::sim_test]
async fn test_state_sync() {
    init_tracing();

    let mut test_cluster = TestClusterBuilder::new().build().await;

    // Make sure the validators are quiescent before bringing up the node.
    sleep(Duration::from_millis(10000)).await;

    // Start a new fullnode that is not on the write path
    let fullnode = test_cluster.spawn_new_fullnode().await.soma_node;

    sleep(Duration::from_millis(30000)).await;
}

#[msim::sim_test]
async fn test_reconfig_with_committee_change_basic() {
    init_tracing();
    // This test exercise the full flow of a validator joining the network, catch up and then leave.

    let new_validator = ValidatorGenesisConfigBuilder::new().build(&mut OsRng);
    let address = (&new_validator.account_key_pair.public()).into();
    let mut test_cluster = TestClusterBuilder::new()
        .with_validator_candidates([address])
        .build()
        .await;

    execute_add_validator_transactions(&test_cluster, &new_validator).await;

    test_cluster.trigger_reconfiguration().await;

    // Check that a new validator has joined the committee.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            5
        );
    });
    let new_validator_handle = test_cluster.spawn_new_validator(new_validator).await;
    test_cluster.wait_for_epoch_all_nodes(1).await;

    new_validator_handle.with(|node| {
        assert!(node
            .state()
            .is_validator(&node.state().epoch_store_for_testing()));
    });

    execute_remove_validator_tx(&test_cluster, &new_validator_handle).await;
    test_cluster.trigger_reconfiguration().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            4
        );
    });
}

// This test just starts up a cluster that reconfigures itself under 0 load.
#[msim::sim_test]
async fn test_passive_reconfig_normal() {
    do_test_passive_reconfig().await;
}

#[msim::sim_test(check_determinism)]
async fn test_passive_reconfig_determinism() {
    do_test_passive_reconfig().await;
}

async fn do_test_passive_reconfig() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(1000)
        .build()
        .await;

    let target_epoch: u64 = std::env::var("RECONFIG_TARGET_EPOCH")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(4);

    test_cluster.wait_for_epoch(Some(target_epoch)).await;
}

#[msim::sim_test]
async fn test_reconfig_with_committee_change_stress() {
    do_test_reconfig_with_committee_change_stress().await;
}

#[msim::sim_test(check_determinism)]
async fn test_reconfig_with_committee_change_stress_determinism() {
    do_test_reconfig_with_committee_change_stress().await;
}

async fn do_test_reconfig_with_committee_change_stress() {
    let mut candidates = (0..6)
        .map(|_| ValidatorGenesisConfigBuilder::new().build(&mut OsRng))
        .collect::<Vec<_>>();
    let addresses = candidates
        .iter()
        .map(|c| (&c.account_key_pair.public()).into())
        .collect::<Vec<SomaAddress>>();
    let mut test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_validator_candidates(addresses)
        // .with_num_unpruned_validators(2)
        .build()
        .await;

    let mut cur_epoch = 0;

    while let Some(v1) = candidates.pop() {
        let v2 = candidates.pop().unwrap();
        execute_add_validator_transactions(&test_cluster, &v1).await;
        execute_add_validator_transactions(&test_cluster, &v2).await;
        let mut removed_validators = vec![];
        for v in test_cluster.swarm.active_validators().take(2) {
            let h = v.get_node_handle().unwrap();
            removed_validators.push(h.state().name);
            execute_remove_validator_tx(&test_cluster, &h).await;
        }
        let handle1 = test_cluster.spawn_new_validator(v1).await;
        let handle2 = test_cluster.spawn_new_validator(v2).await;

        tokio::join!(
            test_cluster.wait_for_epoch_on_node(&handle1, Some(cur_epoch), Duration::from_secs(60)),
            test_cluster.wait_for_epoch_on_node(&handle2, Some(cur_epoch), Duration::from_secs(60))
        );

        test_cluster.trigger_reconfiguration().await;
        let committee = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().committee().clone());
        cur_epoch = committee.epoch();
        assert_eq!(committee.num_members(), 7);
        assert!(committee.authority_exists(&handle1.state().name));
        assert!(committee.authority_exists(&handle2.state().name));
        removed_validators
            .iter()
            .all(|v| !committee.authority_exists(v));
    }
}

async fn execute_remove_validator_tx(test_cluster: &TestCluster, handle: &SomaNodeHandle) {
    let address = handle.with(|node| node.get_config().soma_address());

    let tx = handle.with(|node| {
        Transaction::from_data_and_signer(
            TransactionData::new(
                TransactionKind::StateTransaction(StateTransaction {
                    kind: StateTransactionKind::RemoveValidator(RemoveValidatorArgs {
                        pubkey_bytes: bcs::to_bytes(
                            &node.get_config().account_key_pair.keypair().public(),
                        )
                        .unwrap(),
                    }),
                    sender: (&node.get_config().account_key_pair.keypair().public()).into(),
                }),
                (&node.get_config().account_key_pair.keypair().public()).into(),
            ),
            vec![node.get_config().account_key_pair.keypair()],
        )
    });

    info!(?tx, "Executing remove validator tx");

    test_cluster.execute_transaction(tx).await;
}

/// Execute a sequence of transactions to add a validator, including adding candidate, adding stake
/// and activate the validator.
/// It does not however trigger reconfiguration yet.
async fn execute_add_validator_transactions(
    test_cluster: &TestCluster,
    new_validator: &ValidatorGenesisConfig,
) {
    let pending_active_count = test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node.state().get_system_state_object_for_testing();
        system_state.validators.pending_active_validators.len()
    });

    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::StateTransaction(StateTransaction {
                kind: StateTransactionKind::AddValidator(AddValidatorArgs {
                    pubkey_bytes: bcs::to_bytes(&new_validator.key_pair.public()).unwrap(),
                    network_pubkey_bytes: bcs::to_bytes(&new_validator.network_key_pair.public())
                        .unwrap(),
                    worker_pubkey_bytes: bcs::to_bytes(&new_validator.worker_key_pair.public())
                        .unwrap(),
                    net_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                    p2p_address: bcs::to_bytes(&new_validator.consensus_address).unwrap(),
                    primary_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                }),
                sender: (&new_validator.account_key_pair.public()).into(),
            }),
            (&new_validator.account_key_pair.public()).into(),
        ),
        vec![&new_validator.account_key_pair],
    );
    test_cluster.execute_transaction(tx).await;

    // Check that we can get the pending validator from 0x5.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node.state().get_system_state_object_for_testing();
        let pending_active_validators = system_state.validators.pending_active_validators;
        assert_eq!(pending_active_validators.len(), pending_active_count + 1);
        assert_eq!(
            pending_active_validators[pending_active_validators.len() - 1]
                .metadata
                .soma_address,
            (&new_validator.account_key_pair.public()).into()
        );
    });
}

// async fn test_inactive_validator_pool_read()
// async fn test_validator_candidate_pool_read()
// async fn test_reconfig_with_failing_validator(
// async fn test_create_advance_epoch_tx_race()
// async fn test_expired_locks()
// async fn do_test_passive_reconfig()
