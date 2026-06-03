// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security regression test for finding C1: an externally-submitted
//! FORGED SYSTEM TRANSACTION.
//!
//! A remote attacker (no validator key, no stake) crafts a raw
//! `Transaction` whose kind is `Settlement` and whose `sender` is
//! `SomaAddress::ZERO`, signs it with a dummy all-zero signature, and
//! submits it through the normal client ingress. The `Settlement` kind
//! lets the attacker name arbitrary `BalanceEvent::Deposit` changes —
//! i.e. mint USDC out of thin air to an address they control.
//!
//! Sui rejects this in `SenderSignedData::validity_check` ("CRITICAL!!
//! Users cannot send system transactions"). The question this test
//! answers empirically: does Soma reject it, or accept + execute it?
//!
//! EXPECTED (secure) outcome: submission is rejected at ingress, and the
//! attacker's USDC balance is unchanged (stays 0).
//! FAILURE (vulnerable) outcome: the tx is accepted and/or the attacker's
//! balance increases — arbitrary mint.
//!
//! Faithfulness note: we build a *raw, unverified* `Transaction` (never a
//! `VerifiedTransaction`) and submit it via `execute_transaction_may_fail`,
//! which routes through the same client -> validator verification path a
//! remote attacker would hit. We do NOT use any privileged/verified
//! constructor.

use fastcrypto::traits::ToFromBytes;
use test_cluster::TestClusterBuilder;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::crypto::{Ed25519SomaSignature, SomaSignatureInner};
use types::effects::TransactionEffectsAPI;
use types::object::CoinType;
use types::transaction::SettlementTransaction;
use types::transaction::{
    SenderSignedData, Transaction, TransactionData, TransactionExpiration, TransactionKind,
};
use utils::logging::init_tracing;

fn read_usdc(test_cluster: &test_cluster::TestCluster, address: SomaAddress) -> u64 {
    test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().database_for_testing().get_balance(address, CoinType::Usdc))
        .unwrap_or(0)
}

/// C1: a forged `sender == ZERO` `Settlement` submitted by an external
/// client must NOT be accepted, and must NOT mint balance.
#[cfg(msim)]
#[msim::sim_test]
async fn forged_settlement_system_tx_is_rejected() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().with_num_validators(4).build().await;

    // Attacker-controlled address that should never legitimately receive USDC.
    let attacker = SomaAddress::random();
    let before = read_usdc(&test_cluster, attacker);
    assert_eq!(before, 0, "attacker address must start with no USDC");

    const MINT_AMOUNT: u64 = 1_000_000_000; // 1,000 USDC (6 decimals)

    // === Forge the transaction exactly as a remote attacker would ===
    // A `Settlement` whose `changes` credit the attacker out of nothing.
    let kind = TransactionKind::Settlement(SettlementTransaction {
        epoch: 0,
        round: 0,
        sub_dag_index: None,
        changes: vec![BalanceEvent::deposit(attacker, CoinType::Usdc, MINT_AMOUNT)],
        delegation_changes: vec![],
    });

    // sender = ZERO, no gas, no replay protection — the system-tx shape.
    let data = TransactionData::new_with_expiration(
        kind,
        SomaAddress::ZERO,
        Vec::new(),
        TransactionExpiration::None,
    );

    // Dummy all-zero Ed25519 signature — no real key involved.
    let dummy_sig =
        Ed25519SomaSignature::from_bytes(&vec![0u8; Ed25519SomaSignature::LENGTH]).unwrap();
    let sender_signed = SenderSignedData::new_from_sender_signature(data, dummy_sig.into());

    // Raw, UNVERIFIED transaction envelope (what goes over the wire).
    let forged_tx = Transaction::new(sender_signed);

    // === Submit through the real client ingress ===
    let result = test_cluster.wallet.execute_transaction_may_fail(forged_tx).await;

    match &result {
        Err(e) => {
            info!("forged settlement REJECTED at ingress (secure): {e}");
        }
        Ok(resp) => {
            info!("forged settlement ACCEPTED — status: {:?}", resp.effects.status());
        }
    }

    // The decisive security assertions, independent of how far the tx got:
    let after = read_usdc(&test_cluster, attacker);
    assert_eq!(
        after,
        before,
        "VULNERABLE: forged Settlement minted {} USDC to the attacker (before={}, after={})",
        after.saturating_sub(before),
        before,
        after,
    );

    // Belt-and-suspenders: a forged system tx must not be accepted+executed
    // with success status. (If `result` is Err, the forgery was rejected —
    // the secure outcome.)
    if let Ok(resp) = result {
        assert!(
            !resp.effects.status().is_ok(),
            "VULNERABLE: forged system Settlement transaction executed successfully"
        );
    }
}

// Pull `info!` into scope only under msim (the test is cfg(msim)).
#[cfg(msim)]
use tracing::info;
