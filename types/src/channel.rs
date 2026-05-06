// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unidirectional payment channels.
//!
//! A `Channel` is a shared on-chain object holding escrowed funds and the
//! cumulative-amount semantics of a unidirectional payment relationship
//! between a `payer` and a `payee`. The payer escrows a deposit and
//! signs off-chain `Voucher`s authorizing the payee to claim a
//! monotonically increasing cumulative amount; the payee submits those
//! vouchers on-chain to settle.
//!
//! Designed to mirror Tempo's MPP `TempoStreamChannel` semantics adapted
//! to Soma's stack:
//!   - Ed25519 / MultiSig signatures via [`crate::crypto::Signature`] /
//!     [`crate::crypto::GenericSignature`] instead of EIP-712 ecrecover.
//!   - BCS encoding via [`crate::intent::IntentMessage`] for domain
//!     separation (see [`crate::intent::IntentScope::PaymentVoucher`]).
//!   - Channel ID derived from the open-tx digest, so clients can predict
//!     it client-side without a salt parameter.
//!
//! See `authority::execution::channel` for the executor side.

use serde::{Deserialize, Serialize};

use crate::base::{SomaAddress, TimestampMs};
use crate::digests::TransactionDigest;
use crate::object::{CoinType, Object, ObjectData, ObjectID, ObjectType, Owner, Version};

/// On-chain payment channel.
///
/// Created by `OpenChannel`, mutated by `Settle` / `RequestClose`,
/// and **deleted** (not just flagged closed) on `WithdrawAfterTimeout`.
/// The object's existence is the channel's liveness signal — there is
/// no `closed: bool` field. (A payee-initiated `Close` op and a
/// `TopUp` op are planned for Phase 2; today only the four ops listed
/// above are implemented.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Channel {
    /// Address that opened the channel and owns the deposit (gets
    /// remainder back on close). Authorized to call `RequestClose`
    /// and `WithdrawAfterTimeout` (Phase 1); future `TopUp` (Phase 2)
    /// will also require this address.
    pub payer: SomaAddress,

    /// Address that receives settlements. Authorized to call `Settle`.
    /// Restricting voucher-driven ops to the payee prevents the payer
    /// from short-paying with stale vouchers (see Tempo's
    /// access-control rules). Phase 2 will add a payee-initiated
    /// `Close`.
    pub payee: SomaAddress,

    /// Address whose key signs off-chain vouchers. Typically equal to
    /// `payer` but may differ if the payer wants a hot/cold split: the
    /// cold key (`payer`) holds the deposit while a hot key
    /// (`authorized_signer`) signs vouchers. Use a MultiSig-derived
    /// address for k-of-n joint-custody signing.
    pub authorized_signer: SomaAddress,

    /// Coin denomination escrowed in this channel. USDC for the
    /// inference marketplace; the field exists so other denominations
    /// can be added without a Channel layout change.
    pub token: CoinType,

    /// Current escrow balance. Decreases by `delta` on each `Settle`.
    /// (Phase 2 `TopUp` will also increase it.) Funds flow back to
    /// `payer` on `WithdrawAfterTimeout`.
    pub deposit: u64,

    /// Highest `cumulative_amount` paid out so far. Strictly
    /// increasing across `Settle` calls — old vouchers can never
    /// replay. Always `<= original_deposit`.
    pub settled_amount: u64,

    /// `Some(ts)` once `RequestClose` has been called by the payer;
    /// `None` while the channel is in normal operation. Phase 2's
    /// `TopUp` will clear this so a renewing payer can withdraw their
    /// close request. The grace period elapses when
    /// `current_clock_ts - ts >= channel_grace_period_ms`.
    pub close_requested_at_ms: Option<TimestampMs>,
}

/// Off-chain payment voucher signed by the channel's
/// `authorized_signer`. The voucher commits the signer to letting the
/// payee claim up to `cumulative_amount` on-chain via `Settle` or
/// `Close`.
///
/// **Cumulative semantics**: each new voucher supersedes the previous
/// — the payee submits the highest one they hold, and the channel pays
/// `(cumulative_amount - settled_amount)`. Old vouchers cannot replay
/// because the executor rejects any voucher whose
/// `cumulative_amount <= channel.settled_amount`.
///
/// **Domain separation**: signed via `IntentMessage<Voucher>` with
/// [`crate::intent::IntentScope::PaymentVoucher`]. `channel_id` scopes
/// the signature so the same key signing for multiple channels can't
/// have its vouchers cross-replayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Voucher {
    pub channel_id: ObjectID,
    pub cumulative_amount: u64,
}

impl Voucher {
    pub const fn new(channel_id: ObjectID, cumulative_amount: u64) -> Self {
        Self { channel_id, cumulative_amount }
    }
}

/// Off-chain HTTP-bound voucher signed for the inference marketplace
/// (proxy → provider HTTP path). Same primitive as [`Voucher`]
/// (Ed25519/MultiSig over `IntentMessage<HttpVoucher>` under
/// [`crate::intent::IntentScope::PaymentVoucherHttp`]) but additionally
/// commits to the per-request HTTP context so an adversarial provider
/// can't replay the signature against a different request.
///
/// The on-chain executor never sees `HttpVoucher` — it's purely a
/// transport-layer authorization. When the provider settles on-chain,
/// it builds an ordinary `Voucher` with the same `(channel_id,
/// cumulative_amount)` and uses that signature instead.
///
/// **Field rationale**:
///   - `channel_id` + `cumulative_amount`: matches `Voucher`, so a
///     payer signing both layers commits to the same monetary intent.
///   - `expires_ms`: per-request expiry so a provider can't sit on a
///     signature indefinitely.
///   - `body_sha256`: binds to the request body so the request body
///     can't be swapped after the signature is produced.
///   - `request_id_sha256`: scopes to a unique request id; together
///     with `body_sha256` prevents two distinct requests from sharing
///     a signature.
///   - `method_path_sha256`: binds to method+path (precomputed by the
///     signer so the on-the-wire struct stays fixed-size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HttpVoucher {
    pub channel_id: ObjectID,
    pub cumulative_amount: u64,
    pub expires_ms: TimestampMs,
    pub body_sha256: [u8; 32],
    pub request_id_sha256: [u8; 32],
    pub method_path_sha256: [u8; 32],
}

impl HttpVoucher {
    pub const fn new(
        channel_id: ObjectID,
        cumulative_amount: u64,
        expires_ms: TimestampMs,
        body_sha256: [u8; 32],
        request_id_sha256: [u8; 32],
        method_path_sha256: [u8; 32],
    ) -> Self {
        Self {
            channel_id,
            cumulative_amount,
            expires_ms,
            body_sha256,
            request_id_sha256,
            method_path_sha256,
        }
    }

    /// Convenience constructor that hashes the per-request strings so
    /// callers don't need to import sha2 directly.
    pub fn from_request(
        channel_id: ObjectID,
        cumulative_amount: u64,
        expires_ms: TimestampMs,
        body_bytes: &[u8],
        request_id: &str,
        method: &str,
        path: &str,
    ) -> Self {
        use sha2::{Digest, Sha256};
        let body_sha256: [u8; 32] = Sha256::digest(body_bytes).into();
        let request_id_sha256: [u8; 32] = Sha256::digest(request_id.as_bytes()).into();
        let mut h = Sha256::new();
        h.update(method.as_bytes());
        h.update(b"\n");
        h.update(path.as_bytes());
        let method_path_sha256: [u8; 32] = h.finalize().into();
        Self::new(
            channel_id,
            cumulative_amount,
            expires_ms,
            body_sha256,
            request_id_sha256,
            method_path_sha256,
        )
    }

    /// Project this HTTP voucher down to its on-chain `Voucher`
    /// equivalent — the same `(channel_id, cumulative_amount)` pair
    /// the provider would settle with on the chain.
    pub fn to_voucher(&self) -> Voucher {
        Voucher::new(self.channel_id, self.cumulative_amount)
    }
}

impl Channel {
    /// Construct a fresh channel for `OpenChannel` execution.
    /// `settled_amount` starts at 0 and `close_requested_at_ms` at None.
    pub fn new(
        payer: SomaAddress,
        payee: SomaAddress,
        authorized_signer: SomaAddress,
        token: CoinType,
        deposit: u64,
    ) -> Self {
        Self {
            payer,
            payee,
            authorized_signer,
            token,
            deposit,
            settled_amount: 0,
            close_requested_at_ms: None,
        }
    }

    /// `deposit + settled_amount` — the maximum legal `cumulative_amount`
    /// a voucher could carry. Anything beyond this implies overspending
    /// the escrow.
    pub fn max_cumulative_amount(&self) -> u64 {
        self.deposit.saturating_add(self.settled_amount)
    }

    /// Remainder that would flow back to the payer on a close right
    /// now (i.e., the live deposit). Equal to `self.deposit`. Named
    /// for clarity at call sites.
    pub fn remainder_to_payer(&self) -> u64 {
        self.deposit
    }
}

impl Object {
    /// Build a new `Channel` shared object for `OpenChannel`. Uses
    /// `OBJECT_START_VERSION` for `initial_shared_version` so all
    /// channels have a predictable shared-version key, lettin clients
    /// construct subsequent `Settle` / `RequestClose` /
    /// `WithdrawAfterTimeout` transactions without first looking up the
    /// channel's lamport-timestamped version. The execution layer
    /// preserves this value (see
    /// `temporary_store::ExecutionResults::update_version_and_previous_tx`).
    pub fn new_channel(
        id: ObjectID,
        channel: Channel,
        previous_transaction: TransactionDigest,
    ) -> Self {
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Channel,
            Version::MIN,
            bcs::to_bytes(&channel).expect("Channel serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            previous_transaction,
        )
    }

    /// Test/debug constructor: creates a Channel object at
    /// `OBJECT_START_VERSION` so it can be loaded as an input without
    /// going through full open-tx execution.
    pub fn new_channel_for_testing(id: ObjectID, channel: Channel) -> Self {
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Channel,
            crate::object::OBJECT_START_VERSION,
            bcs::to_bytes(&channel).expect("Channel serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            TransactionDigest::default(),
        )
    }

    /// If this object is a Channel, deserialize and return it.
    pub fn as_channel(&self) -> Option<Channel> {
        if *self.data.object_type() == ObjectType::Channel {
            bcs::from_bytes::<Channel>(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Overwrite a Channel object's contents. Caller must ensure this
    /// object actually IS a Channel; debug-asserts the type.
    pub fn set_channel_data(&mut self, channel: &Channel) {
        debug_assert_eq!(
            *self.data.object_type(),
            ObjectType::Channel,
            "set_channel_data called on non-Channel object"
        );
        self.update_contents(channel);
    }
}

#[cfg(test)]
mod tests {
    use fastcrypto::ed25519::Ed25519KeyPair;
    use fastcrypto::traits::KeyPair;

    use super::*;
    use crate::crypto::{Signature, SomaSignature, get_key_pair};
    use crate::intent::{Intent, IntentMessage, IntentScope};

    /// Channel BCS round-trip: serialize → deserialize → equal.
    #[test]
    fn channel_bcs_round_trip() {
        let ch = Channel::new(
            SomaAddress::random(),
            SomaAddress::random(),
            SomaAddress::random(),
            CoinType::Usdc,
            1_000_000,
        );
        let bytes = bcs::to_bytes(&ch).expect("Channel serializes");
        let decoded: Channel = bcs::from_bytes(&bytes).expect("Channel deserializes");
        assert_eq!(decoded, ch);
        assert_eq!(decoded.settled_amount, 0);
        assert_eq!(decoded.close_requested_at_ms, None);
    }

    /// Voucher BCS round-trip — the on-the-wire representation must
    /// be stable.
    #[test]
    fn voucher_bcs_round_trip() {
        let v = Voucher::new(ObjectID::random(), 12_345);
        let bytes = bcs::to_bytes(&v).expect("Voucher serializes");
        let decoded: Voucher = bcs::from_bytes(&bytes).expect("Voucher deserializes");
        assert_eq!(decoded, v);
    }

    /// Object<->Channel helpers: type, ownership, contents survive
    /// the construction → as_channel round-trip.
    #[test]
    fn object_channel_helpers() {
        let id = ObjectID::random();
        let ch = Channel::new(
            SomaAddress::random(),
            SomaAddress::random(),
            SomaAddress::random(),
            CoinType::Usdc,
            500,
        );
        let obj = Object::new_channel_for_testing(id, ch.clone());
        assert_eq!(obj.id(), id);
        assert_eq!(*obj.type_(), ObjectType::Channel);
        assert!(obj.is_shared(), "Channel must be a shared object");
        assert_eq!(obj.as_channel().expect("deserialize"), ch);
    }

    /// `set_channel_data` overwrites contents, preserving id and type.
    #[test]
    fn set_channel_data_round_trips() {
        let id = ObjectID::random();
        let ch1 = Channel::new(
            SomaAddress::ZERO,
            SomaAddress::ZERO,
            SomaAddress::ZERO,
            CoinType::Usdc,
            100,
        );
        let mut obj = Object::new_channel_for_testing(id, ch1);

        let mut ch2 = obj.as_channel().unwrap();
        ch2.deposit = 80;
        ch2.settled_amount = 20;
        obj.set_channel_data(&ch2);

        assert_eq!(obj.id(), id, "id must not change");
        assert_eq!(*obj.type_(), ObjectType::Channel, "type must not change");
        let read_back = obj.as_channel().unwrap();
        assert_eq!(read_back.deposit, 80);
        assert_eq!(read_back.settled_amount, 20);
    }

    /// `as_channel` returns None for non-Channel objects.
    #[test]
    fn as_channel_rejects_other_types() {
        let coin = Object::with_id_owner_for_testing(ObjectID::random(), SomaAddress::random());
        assert!(coin.as_channel().is_none());
    }

    /// **End-to-end voucher signing**: produce a signature with an
    /// Ed25519 keypair, verify with the existing
    /// `Signature::verify_secure` API, confirm tampering is rejected.
    /// This is the path the executor will use.
    #[test]
    fn voucher_signs_and_verifies() {
        let (signer_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        let voucher = Voucher::new(ObjectID::random(), 42);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);

        let sig = Signature::new_secure(&intent_msg, &kp);
        sig.verify_secure(&intent_msg, signer_addr, sig.scheme())
            .expect("signature verifies against the signer's address");
    }

    /// A voucher signed by one key must not verify against a different
    /// claimed author — IncorrectSigner rejection.
    #[test]
    fn voucher_rejected_for_wrong_author() {
        let (_, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        let other = SomaAddress::random();
        let voucher = Voucher::new(ObjectID::random(), 1);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        let sig = Signature::new_secure(&intent_msg, &kp);
        sig.verify_secure(&intent_msg, other, sig.scheme())
            .expect_err("verification must fail when claimed signer != actual signer");
    }

    /// Tampering with the cumulative_amount must invalidate the
    /// signature — verifier hashes the entire IntentMessage, not just
    /// channel_id.
    #[test]
    fn voucher_rejected_after_tampering() {
        let (signer_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        let original = Voucher::new(ObjectID::random(), 100);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), original);
        let sig = Signature::new_secure(&intent_msg, &kp);

        // Forge a higher amount but use the same signature — must reject.
        let tampered = Voucher::new(original.channel_id, 9999);
        let tampered_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), tampered);
        sig.verify_secure(&tampered_msg, signer_addr, sig.scheme())
            .expect_err("tampered cumulative_amount must invalidate the signature");
    }

    /// Cross-channel replay: a voucher signed for channel A must NOT
    /// verify if presented as a voucher for channel B. This is the
    /// channel_id field's whole purpose.
    #[test]
    fn voucher_does_not_replay_across_channels() {
        let (signer_addr, kp): (SomaAddress, Ed25519KeyPair) = get_key_pair();
        let chan_a = ObjectID::random();
        let chan_b = ObjectID::random();
        assert_ne!(chan_a, chan_b);

        let voucher_a = Voucher::new(chan_a, 50);
        let intent_msg_a =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher_a);
        let sig_a = Signature::new_secure(&intent_msg_a, &kp);

        // The same `cumulative_amount=50` for chan_b must not verify
        // with `sig_a` — the channel_id is part of the hashed payload.
        let voucher_b = Voucher::new(chan_b, 50);
        let intent_msg_b =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher_b);
        sig_a
            .verify_secure(&intent_msg_b, signer_addr, sig_a.scheme())
            .expect_err("voucher signed for channel A must not verify against channel B");
    }

    /// Domain separation: a voucher hash differs from the same
    /// (channel_id, cumulative_amount) under a *different* IntentScope.
    /// Sanity check that PaymentVoucher is its own domain.
    #[test]
    fn voucher_domain_separated_from_other_scopes() {
        let voucher = Voucher::new(ObjectID::ZERO, 0);
        let im_voucher =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        // ProofOfPossession over the same struct shape (BCS bytes are
        // the same content but the intent prefix differs).
        let im_pop =
            IntentMessage::new(Intent::soma_app(IntentScope::ProofOfPossession), voucher);
        assert_ne!(
            bcs::to_bytes(&im_voucher).unwrap(),
            bcs::to_bytes(&im_pop).unwrap(),
            "intent prefix must domain-separate scopes"
        );
    }
}
