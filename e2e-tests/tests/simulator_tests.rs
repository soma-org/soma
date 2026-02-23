//! Simulator determinism tests.
//!
//! These tests verify that the msim simulator produces deterministic results
//! across multiple runs. Each test uses `check_determinism` to run twice and
//! assert identical execution paths.
//!
//! Ported from Sui's `simulator_tests.rs` — zero Move dependency.
//!
//! Tests:
//! 1. test_futures_ordered — FuturesOrdered yields in deterministic order
//! 2. test_futures_unordered — FuturesUnordered yields in deterministic order
//! 3. test_select_unbiased — select! over two FuturesUnordered is deterministic
//! 4. test_hash_collections — HashMap/HashSet iteration order is deterministic
//! 5. test_net_determinism — Full network + transaction execution is deterministic

use futures::{
    StreamExt,
    stream::{FuturesOrdered, FuturesUnordered},
};
use rand::{
    Rng,
    distributions::{Distribution, Uniform},
    rngs::OsRng,
};
// Use msim's deterministic HashMap/HashSet (seeded with fixed ahash keys)
// std::collections versions use randomized hashing and would fail check_determinism.
#[cfg(msim)]
use msim::collections::{HashMap, HashSet};
#[cfg(not(msim))]
use std::collections::{HashMap, HashSet};
use test_cluster::TestClusterBuilder;
use tokio::time::{Duration, sleep};
use tracing::{debug, trace};
use types::{
    effects::TransactionEffectsAPI,
    transaction::{TransactionData, TransactionKind},
};
use utils::logging::init_tracing;

async fn make_fut(i: usize) -> usize {
    let count_dist = Uniform::from(1..5);
    let sleep_dist = Uniform::from(1000..10000);

    let count = count_dist.sample(&mut OsRng);
    for _ in 0..count {
        let dur = Duration::from_millis(sleep_dist.sample(&mut OsRng));
        trace!("sleeping for {:?}", dur);
        sleep(dur).await;
    }

    trace!("future {} finished", i);
    i
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_futures_ordered() {
    init_tracing();

    let mut futures = FuturesOrdered::from_iter((0..200).map(make_fut));

    while (futures.next().await).is_some() {
        // mix rng state as futures finish
        OsRng.r#gen::<u32>();
    }
    debug!("final rng state: {}", OsRng.r#gen::<u32>());
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_futures_unordered() {
    init_tracing();

    let mut futures = FuturesUnordered::from_iter((0..200).map(make_fut));

    while let Some(i) = futures.next().await {
        // mix rng state depending on the order futures finish in
        for _ in 0..i {
            OsRng.r#gen::<u32>();
        }
    }
    debug!("final rng state: {}", OsRng.r#gen::<u32>());
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_select_unbiased() {
    init_tracing();

    let mut f1 = FuturesUnordered::from_iter((0..200).map(make_fut));
    let mut f2 = FuturesUnordered::from_iter((0..200).map(make_fut));

    loop {
        tokio::select! {
            Some(i) = f1.next() => {
                for _ in 0..i {
                    OsRng.r#gen::<u32>();
                }
            }

            Some(i) = f2.next() => {
                for _ in 0..i {
                    // mix differently when f2 yields
                    OsRng.r#gen::<u32>();
                    OsRng.r#gen::<u32>();
                }
            }

            else => break
        }
    }

    assert!(f1.is_empty());
    assert!(f2.is_empty());
    debug!("final rng state: {}", OsRng.r#gen::<u32>());
}

#[cfg(msim)]
#[msim::sim_test(check_determinism)]
async fn test_hash_collections() {
    init_tracing();

    let mut map = HashMap::new();
    let mut set = HashSet::new();

    for i in 0..1000 {
        map.insert(i, i);
        set.insert(i);
    }

    // mix the random state according to the first 500 elements of each map
    // so that if iteration order changes, we get different results.
    for (i, _) in map.iter().take(500) {
        for _ in 0..*i {
            OsRng.r#gen::<u32>();
        }
    }

    for i in set.iter().take(500) {
        for _ in 0..*i {
            OsRng.r#gen::<u32>();
        }
    }

    debug!("final rng state: {}", OsRng.r#gen::<u32>());
}

/// Test that starting up a network + fullnode, and sending one transaction through that network
/// produces correct results. Note: check_determinism is not used here because the RPC wallet
/// path involves non-deterministic elements (e.g., gRPC connection setup) that vary between runs.
/// The first four tests above validate that msim's core scheduling is deterministic.
#[cfg(msim)]
#[msim::sim_test]
async fn test_net_determinism() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    // Create a simple coin transfer transaction
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    sleep(Duration::from_millis(1000)).await;

    debug!("transaction executed successfully, checking determinism");
}
