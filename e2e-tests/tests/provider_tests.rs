// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the on-chain provider registry.
//!
//! Exercises the full submission path: a user-signed `Transaction`
//! flows through the wallet/RPC layer, hits validators via consensus,
//! runs through the executor, and produces durable state changes that
//! the fullnode reflects. This is the validators-actually-run-it
//! version of the executor tests in `authority/src/execution/provider.rs`.

use test_cluster::{TestCluster, TestClusterBuilder};
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI as _;
use types::object::{Object, ObjectID};
use types::provider::Provider;
use types::transaction::{RegisterProviderArgs, TransactionKind, UpdateProviderArgs};
use utils::logging::init_tracing;

fn read_provider(test_cluster: &TestCluster, id: ObjectID) -> Option<Provider> {
    test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_object_store()
            .get_object(&id)
            .as_ref()
            .and_then(Object::as_provider)
    })
}

async fn submit_register(
    test_cluster: &TestCluster,
    signer: SomaAddress,
    endpoint: &str,
) -> bool {
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        signer,
        TransactionKind::RegisterProvider(RegisterProviderArgs {
            endpoint: endpoint.to_string(),
        }),
    );
    match test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
    {
        Ok(r) => {
            let ok = r.effects.status().is_ok();
            if !ok {
                eprintln!("RegisterProvider rejected: {:?}", r.effects.status());
            }
            ok
        }
        Err(e) => {
            eprintln!("RegisterProvider submission errored: {:?}", e);
            false
        }
    }
}

async fn submit_update(
    test_cluster: &TestCluster,
    signer: SomaAddress,
    endpoint: &str,
) -> bool {
    let provider_id = Provider::derive_id(signer);
    let tx_data = e2e_tests::stateless_tx_data(
        test_cluster,
        signer,
        TransactionKind::UpdateProvider(UpdateProviderArgs {
            provider_id,
            endpoint: endpoint.to_string(),
        }),
    );
    match test_cluster
        .wallet
        .execute_transaction_may_fail(test_cluster.wallet.sign_transaction(&tx_data).await)
        .await
    {
        Ok(r) => {
            let ok = r.effects.status().is_ok();
            if !ok {
                tracing::warn!(status = ?r.effects.status(), "UpdateProvider rejected");
            }
            ok
        }
        Err(e) => {
            tracing::warn!(err = ?e, "UpdateProvider submission errored");
            false
        }
    }
}

/// Happy path: register lands a Provider object at the derived id
/// with the supplied endpoint. Update changes the endpoint while
/// preserving address.
#[cfg(msim)]
#[msim::sim_test]
async fn provider_register_then_update_e2e() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let addrs = test_cluster.wallet.get_addresses();
    let signer = addrs[0];

    // Register.
    let endpoint_1 = "https://provider-one.example:8080";
    assert!(
        submit_register(&test_cluster, signer, endpoint_1).await,
        "RegisterProvider must succeed"
    );

    let id = Provider::derive_id(signer);
    let p1 = read_provider(&test_cluster, id).expect("Provider object created");
    assert_eq!(p1.address(), signer);
    assert_eq!(p1.endpoint(), endpoint_1);

    // Update — endpoint change.
    let endpoint_2 = "https://provider-one-v2.example:8080";
    assert!(
        submit_update(&test_cluster, signer, endpoint_2).await,
        "UpdateProvider must succeed"
    );

    let p2 = read_provider(&test_cluster, id).expect("Provider still present");
    assert_eq!(p2.address(), signer, "address never changes");
    assert_eq!(p2.endpoint(), endpoint_2, "endpoint updated");
}

// Note: duplicate-detection and wrong-signer paths are covered by
// the unit tests in `authority/src/execution/provider.rs`. They
// exercise the `temporary_store.read_object` path with the Provider
// already loaded as a shared input — which the e2e scheduler cannot
// arrange when the object's existence is the property under test.
// SDKs avoid the duplicate-register pitfall by going through
// `sdk::provider::register_or_update`.

/// UpdateProvider on a never-registered address fails (no Provider
/// object to load).
#[cfg(msim)]
#[msim::sim_test]
async fn provider_update_rejects_unregistered_e2e() {
    init_tracing();
    let test_cluster = TestClusterBuilder::new().build().await;
    let signer = test_cluster.wallet.get_addresses()[0];
    let ok = submit_update(&test_cluster, signer, "https://prov.example").await;
    assert!(!ok, "UpdateProvider without RegisterProvider must fail");

    // No Provider object got created.
    let p = read_provider(&test_cluster, Provider::derive_id(signer));
    assert!(p.is_none(), "no Provider object should exist");
}
