// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Helpers for building, signing, and verifying payment-channel
//! vouchers, plus client-side prediction of the on-chain channel ID
//! for a freshly-opened channel.
//!
//! Vouchers are off-chain Ed25519/MultiSig signatures over an
//! `IntentMessage<Voucher>` (BCS-encoded) bound to
//! `IntentScope::PaymentVoucher`. The payee redeems the highest
//! voucher they hold by submitting it in a `Settle` tx; the executor
//! verifies the signature against the channel's `authorized_signer`.
//!
//! Most integrators only need [`sign_voucher`] (payer side, per
//! request) and [`predicted_channel_id`] (right after the OpenChannel
//! tx fires, to know what to put in subsequent vouchers without
//! waiting on the indexer to pick up the channel).

use soma_keys::keystore::AccountKeystore;
use types::base::SomaAddress;
use types::channel::{Channel, HttpVoucher, Voucher};
use types::crypto::{GenericSignature, Signature};
use types::digests::TransactionDigest;
use types::intent::{Intent, IntentMessage, IntentScope};
use types::object::{CoinType, ObjectID};
use types::transaction::{
    OpenChannelArgs, RequestCloseArgs, SettleArgs, TopUpArgs, Transaction,
    TransactionKind, WithdrawAfterTimeoutArgs,
};

use crate::transaction_builder::TransactionBuilder;
use crate::wallet_context::WalletContext;

/// Sign a voucher authorizing `cumulative_amount` on `channel_id`,
/// returning the canonical wire form (`GenericSignature`) ready to
/// embed in a `SettleArgs`.
///
/// `signer` must be the channel's `authorized_signer` (typically the
/// payer's address, or a hot-key delegate).
///
/// The keystore is consulted for the signing key — this is
/// deliberately the same path used by every other tx-signing flow on
/// the chain, so MultiSig signers work transparently.
pub async fn sign_voucher<K: AccountKeystore>(
    keystore: &K,
    signer: &SomaAddress,
    channel_id: ObjectID,
    cumulative_amount: u64,
) -> anyhow::Result<GenericSignature> {
    let voucher = Voucher::new(channel_id, cumulative_amount);
    let sig: Signature = keystore
        .sign_secure::<Voucher>(signer, &voucher, Intent::soma_app(IntentScope::PaymentVoucher))
        .await
        .map_err(|e| anyhow::anyhow!("voucher signing failed: {}", e))?;
    Ok(sig.into())
}

/// Verify a voucher's signature against the channel's
/// `authorized_signer`. Cheap, in-process — useful for payees who
/// want to validate a voucher before redeeming it on-chain (a
/// rejected on-chain `Settle` still burns gas).
pub fn verify_voucher(
    channel: &Channel,
    voucher: Voucher,
    signature: &GenericSignature,
) -> anyhow::Result<()> {
    let intent_msg = IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
    signature
        .verify_authenticator(&intent_msg, channel.authorized_signer)
        .map_err(|e| anyhow::anyhow!("voucher signature verification failed: {}", e))
}

/// Predict the `ObjectID` that an `OpenChannel` tx will produce.
///
/// The on-chain executor derives the channel's id deterministically
/// from the open-tx digest plus the in-tx creation counter (which is
/// 0 for `OpenChannel`, since the channel is the only object the op
/// creates). Clients can therefore know the id before the tx is
/// even confirmed — useful for caching, indexing, and constructing
/// the next tx pointing at this channel without waiting on a fullnode
/// round-trip.
pub fn predicted_channel_id(open_tx_digest: TransactionDigest) -> ObjectID {
    // Mirrors `ChannelExecutor::execute_open` exactly — `OpenChannel`
    // creates the channel as the first (and only) object.
    ObjectID::derive_id(open_tx_digest, 0)
}

/// Sign an HTTP-bound voucher for the inference marketplace HTTP path.
///
/// Uses the same primitive as [`sign_voucher`] (Ed25519/MultiSig over
/// `IntentMessage<HttpVoucher>` under
/// [`IntentScope::PaymentVoucherHttp`]) — same keystore call, same
/// MultiSig support, same `GenericSignature` wire type. The only
/// difference is the per-request HTTP binding committed in
/// `HttpVoucher`.
pub async fn sign_http_voucher<K: AccountKeystore>(
    keystore: &K,
    signer: &SomaAddress,
    http_voucher: &HttpVoucher,
) -> anyhow::Result<GenericSignature> {
    let sig: Signature = keystore
        .sign_secure::<HttpVoucher>(
            signer,
            http_voucher,
            Intent::soma_app(IntentScope::PaymentVoucherHttp),
        )
        .await
        .map_err(|e| anyhow::anyhow!("HTTP voucher signing failed: {}", e))?;
    Ok(sig.into())
}

/// Verify an HTTP voucher signature against the channel's
/// `authorized_signer`. Mirror of [`verify_voucher`] but for the
/// HTTP-bound layer.
pub fn verify_http_voucher(
    channel: &Channel,
    http_voucher: &HttpVoucher,
    signature: &GenericSignature,
) -> anyhow::Result<()> {
    let intent_msg = IntentMessage::new(
        Intent::soma_app(IntentScope::PaymentVoucherHttp),
        *http_voucher,
    );
    signature
        .verify_authenticator(&intent_msg, channel.authorized_signer)
        .map_err(|e| anyhow::anyhow!("HTTP voucher signature verification failed: {}", e))
}

// ---------------------------------------------------------------------
// On-chain SDK ops — wrappers around `WalletContext` + `TransactionBuilder`
// that any caller (proxy, provider, CLI) can use to drive a channel
// without re-implementing the build/sign/submit loop.
// ---------------------------------------------------------------------

/// Open a new on-chain channel. Returns the new `Channel`'s
/// `ObjectID` (deterministic — `predicted_channel_id(tx_digest)`)
/// once the tx is finalized.
pub async fn open_channel(
    ctx: &WalletContext,
    sender: SomaAddress,
    payee: SomaAddress,
    authorized_signer: SomaAddress,
    coin_type: CoinType,
    deposit_amount: u64,
) -> anyhow::Result<ObjectID> {
    let kind = TransactionKind::OpenChannel(OpenChannelArgs {
        payee,
        authorized_signer,
        token: coin_type,
        deposit_amount,
    });
    let tx = build_signed(ctx, sender, kind).await?;
    let tx_digest = *tx.digest();
    let _ = ctx.execute_transaction_must_succeed(tx).await;
    Ok(predicted_channel_id(tx_digest))
}

/// Settle on-chain: provider-side. Submits the latest voucher
/// signature against the channel's running cumulative.
pub async fn settle(
    ctx: &WalletContext,
    sender: SomaAddress,
    voucher: Voucher,
    voucher_signature: GenericSignature,
) -> anyhow::Result<()> {
    let kind = TransactionKind::Settle(SettleArgs {
        channel_id: voucher.channel_id,
        cumulative_amount: voucher.cumulative_amount,
        voucher_signature,
    });
    let tx = build_signed(ctx, sender, kind).await?;
    let _ = ctx.execute_transaction_must_succeed(tx).await;
    Ok(())
}

/// Top up an existing channel. Payer-only on-chain.
pub async fn top_up(
    ctx: &WalletContext,
    sender: SomaAddress,
    channel_id: ObjectID,
    coin_type: CoinType,
    amount: u64,
) -> anyhow::Result<()> {
    let kind = TransactionKind::TopUp(TopUpArgs { channel_id, coin_type, amount });
    let tx = build_signed(ctx, sender, kind).await?;
    let _ = ctx.execute_transaction_must_succeed(tx).await;
    Ok(())
}

/// Begin the close timer. Payer-only on-chain.
pub async fn request_close(
    ctx: &WalletContext,
    sender: SomaAddress,
    channel_id: ObjectID,
) -> anyhow::Result<()> {
    let kind = TransactionKind::RequestClose(RequestCloseArgs { channel_id });
    let tx = build_signed(ctx, sender, kind).await?;
    let _ = ctx.execute_transaction_must_succeed(tx).await;
    Ok(())
}

/// Withdraw remainder after the grace period elapses. Payer-only.
pub async fn withdraw_after_timeout(
    ctx: &WalletContext,
    sender: SomaAddress,
    channel_id: ObjectID,
) -> anyhow::Result<()> {
    let kind =
        TransactionKind::WithdrawAfterTimeout(WithdrawAfterTimeoutArgs { channel_id });
    let tx = build_signed(ctx, sender, kind).await?;
    let _ = ctx.execute_transaction_must_succeed(tx).await;
    Ok(())
}

async fn build_signed(
    ctx: &WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
) -> anyhow::Result<Transaction> {
    TransactionBuilder::new(ctx)
        .build_transaction_async(sender, kind)
        .await
}

/// Compute the next voucher's cumulative amount given the previous
/// voucher's cumulative and an additional amount to authorize on
/// this request. Just `prev + delta`, but spelled out so callers
/// don't accidentally re-send the previous cumulative or an absolute
/// amount.
///
/// Returns `None` on overflow.
pub fn next_cumulative(prev_cumulative: u64, additional_amount: u64) -> Option<u64> {
    prev_cumulative.checked_add(additional_amount)
}

#[cfg(test)]
mod tests {
    use fastcrypto::ed25519::Ed25519KeyPair;
    use types::crypto::get_key_pair;
    use types::object::CoinType;

    use super::*;

    /// `predicted_channel_id` is deterministic and matches what the
    /// executor would compute.
    #[test]
    fn predicted_channel_id_matches_executor_derivation() {
        let digest = TransactionDigest::default();
        let predicted = predicted_channel_id(digest);
        let direct = ObjectID::derive_id(digest, 0);
        assert_eq!(predicted, direct);
    }

    /// Round-trip: signed voucher verifies against the channel's
    /// authorized_signer.
    #[test]
    fn voucher_round_trip() {
        let (signer_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        let payer = SomaAddress::random();
        let payee = SomaAddress::random();
        let channel = Channel::new(payer, payee, signer_addr, CoinType::Usdc, 1_000);

        let channel_id = ObjectID::random();
        let voucher = Voucher::new(channel_id, 250);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        let sig: GenericSignature = Signature::new_secure(&intent_msg, &kp).into();

        verify_voucher(&channel, voucher, &sig).expect("verifies");
    }

    /// `next_cumulative` is plain addition; overflow surfaces as
    /// `None` rather than wrapping silently.
    #[test]
    fn next_cumulative_basic() {
        assert_eq!(next_cumulative(100, 50), Some(150));
        assert_eq!(next_cumulative(0, 0), Some(0));
        assert_eq!(next_cumulative(u64::MAX - 1, 1), Some(u64::MAX));
        assert_eq!(next_cumulative(u64::MAX, 1), None);
    }
}
