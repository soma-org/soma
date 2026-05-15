// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for e2e tests.
//!
//! Stage 13c: balance-mode txs are "stateless" — they have no owned-object
//! inputs to anchor replay protection — and so must declare an explicit
//! `TransactionExpiration::ValidDuring` with both `min_epoch` and
//! `max_epoch` set. The validator rejects stateless txs with
//! `TransactionExpiration::None` as a "use of disabled feature". Most
//! e2e tests just want a simple "deliver this BalanceTransfer in the
//! current/next epoch" — this helper abstracts the boilerplate.

use test_cluster::TestCluster;
use types::base::SomaAddress;
use types::object::CoinType;
use types::transaction::{
    BalanceTransferArgs, TransactionData, TransactionExpiration, TransactionKind,
};

/// Build a stateless balance-mode `TransactionData` with a default
/// expiration window of `[current_epoch, current_epoch + 1]`, and a
/// fresh nonce. The protocol restricts `ValidDuring` to either
/// `max_epoch == min_epoch` or `max_epoch == min_epoch + 1`; this
/// helper uses the latter so the tx survives a single epoch
/// boundary that may roll while it's in flight. Chain ID and
/// current epoch are read from the fullnode at call time.
pub fn balance_transfer_data(
    test_cluster: &TestCluster,
    coin_type: CoinType,
    sender: SomaAddress,
    transfers: Vec<(SomaAddress, u64)>,
) -> TransactionData {
    let current_epoch = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());
    balance_transfer_data_with_window(
        test_cluster,
        coin_type,
        sender,
        transfers,
        current_epoch,
        current_epoch + 1,
    )
}

/// Build a stateless balance-mode `TransactionData` with an explicit
/// epoch window. Use this when the test crosses epoch boundaries.
pub fn balance_transfer_data_with_window(
    test_cluster: &TestCluster,
    coin_type: CoinType,
    sender: SomaAddress,
    transfers: Vec<(SomaAddress, u64)>,
    min_epoch: u64,
    max_epoch: u64,
) -> TransactionData {
    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TransactionData::new_with_expiration(
        TransactionKind::BalanceTransfer(BalanceTransferArgs { coin_type, transfers }),
        sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(min_epoch),
            max_epoch: Some(max_epoch),
            chain,
            nonce,
        },
    )
}

/// Build a balance-mode `TransactionData` for an arbitrary
/// `TransactionKind`. Stage 13c: gas comes from the sender's USDC
/// accumulator, so `gas_payment` is empty. Wraps the kind in a
/// fresh-nonce `ValidDuring` expiration spanning the current epoch
/// and the next one.
pub fn stateless_tx_data(
    test_cluster: &TestCluster,
    sender: SomaAddress,
    kind: TransactionKind,
) -> TransactionData {
    let chain = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_chain_identifier());
    let current_epoch = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().epoch_store_for_testing().epoch());
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    TransactionData::new_with_expiration(
        kind,
        sender,
        Vec::new(),
        TransactionExpiration::ValidDuring {
            min_epoch: Some(current_epoch),
            max_epoch: Some(current_epoch + 1),
            chain,
            nonce,
        },
    )
}

/// Tx-local nonce — keeps each `balance_transfer_data` /
/// `stateless_tx_data` call producing a distinct digest even when
/// the inputs are identical.
static NONCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
