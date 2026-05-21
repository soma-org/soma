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
//! ## Schema versioning
//!
//! [`Channel`], [`Voucher`], and [`HttpVoucher`] are encoded as Rust
//! enums whose only variant today is `V1(...)`. BCS encodes an enum as
//! a `uleb128` discriminant followed by the variant payload — so the
//! leading byte of every Channel/Voucher/HttpVoucher BCS payload is
//! the variant index, identical to a `version: u8` field but with
//! compile-time exhaustivity. When a v2 layout is introduced, we add a
//! `V2(ChannelV2)` arm; every read site is forced to match exhaustively
//! and stale readers cannot accidentally interpret v2 bytes as v1.
//!
//! Code paths that need to mutate fields destructure to the inner
//! versioned struct (`let Channel::V1(c) = channel`); code paths that
//! only need to read use the accessor methods on the outer enum.
//!
//! See `authority::execution::channel` for the executor side.

use serde::{Deserialize, Serialize};

use crate::base::{SomaAddress, TimestampMs};
use crate::digests::TransactionDigest;
use crate::object::{CoinType, Object, ObjectData, ObjectID, ObjectType, Owner, Version};

/// On-chain payment channel.
///
/// Created by `OpenChannel`, mutated by `Settle` / `RequestClose` /
/// `TopUp`, and **deleted** (not just flagged closed) on
/// `WithdrawAfterTimeout`. The object's existence is the channel's
/// liveness signal — there is no `closed: bool` field.
///
/// Versioned via the enum tag — see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Channel {
    V1(ChannelV1),
}

/// V1 layout of [`Channel`]. See [`Channel`] for the wrapper.
///
/// **One model per channel** — `model_id` plus the price/SLA snapshot
/// columns are frozen at `OpenChannel` time from the matching
/// [`crate::offering::Offering`] object. The provider may freely update
/// the offering afterward; this channel's settlement math is
/// deterministic regardless. Buyer opens a fresh channel to get the
/// new price.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelV1 {
    /// Address that opened the channel and owns the deposit (gets
    /// remainder back on close). Authorized to call `RequestClose`,
    /// `WithdrawAfterTimeout`, and `TopUp`.
    pub payer: SomaAddress,

    /// Address that receives settlements. Authorized to call `Settle`.
    /// Restricting voucher-driven ops to the payee prevents the payer
    /// from short-paying with stale vouchers (see Tempo's
    /// access-control rules).
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

    /// Current escrow balance. Decreases by `delta` on each `Settle`,
    /// increases on `TopUp`. Funds flow back to `payer` on
    /// `WithdrawAfterTimeout`.
    pub deposit: u64,

    /// Highest `cumulative_amount` paid out so far. Strictly
    /// increasing across `Settle` calls — old vouchers can never
    /// replay. Always `<= original_deposit`.
    pub settled_amount: u64,

    /// `Some(ts)` once `RequestClose` has been called by the payer;
    /// `None` while the channel is in normal operation. `TopUp`
    /// clears this so a renewing payer can withdraw their close
    /// request. The grace period elapses when
    /// `current_clock_ts - ts >= channel_grace_period_ms`.
    pub close_requested_at_ms: Option<TimestampMs>,

    /// Canonical model_id from the protocol-config `ModelRegistry`
    /// snapshotted at open time. This channel may only serve requests
    /// for this specific model; pre-flight checks at the inference
    /// server bind on it.
    pub model_id: String,

    /// Snapshot of `Offering.prompt_micros_per_1k` at open time.
    pub prompt_micros_per_1k: u64,
    /// Snapshot of `Offering.completion_micros_per_1k` at open time.
    pub completion_micros_per_1k: u64,
    /// Snapshot of `Offering.cache_read_micros_per_1k` at open time.
    pub cache_read_micros_per_1k: u64,
    /// Snapshot of `Offering.cache_write_micros_per_1k` at open time.
    pub cache_write_micros_per_1k: u64,
    /// Snapshot of `Offering.request_micros` at open time.
    pub request_micros: u64,
    /// Snapshot of `Offering.ttft_bound_ms` at open time. Used by the
    /// proxy as the SLA threshold for auto-emitting
    /// `RateChannel(reason_code=TtftBreach)` events.
    pub ttft_bound_ms: u32,
    /// Snapshot of `Offering.ttot_bound_ms` at open time.
    pub ttot_bound_ms: u32,
}

/// Off-chain payment voucher signed by the channel's
/// `authorized_signer`. The voucher commits the signer to letting the
/// payee claim up to `cumulative_amount` on-chain via `Settle`.
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
///
/// Versioned via the enum tag — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Voucher {
    V1(VoucherV1),
}

/// V1 layout of [`Voucher`].
///
/// `cumulative_amount` is the only field consulted by the on-chain
/// executor (settle math). The remaining cumulative-usage fields are
/// signed alongside it for auditability — the indexer materializes
/// them per channel so off-chain oracle views can compute per-model
/// throughput, $/token effective rates, etc. without re-deriving from
/// the price snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VoucherV1 {
    pub channel_id: ObjectID,
    pub cumulative_amount: u64,
    pub cumulative_prompt_tokens: u64,
    pub cumulative_completion_tokens: u64,
    pub cumulative_cache_read_tokens: u64,
    pub cumulative_cache_write_tokens: u64,
    pub cumulative_requests: u64,
}

impl Voucher {
    /// Construct a fresh v1 voucher with explicit usage breakdown.
    /// Most callers should use this form. Settlement only consults
    /// `cumulative_amount`; the rest is auditable metadata.
    pub const fn new(
        channel_id: ObjectID,
        cumulative_amount: u64,
        cumulative_prompt_tokens: u64,
        cumulative_completion_tokens: u64,
        cumulative_cache_read_tokens: u64,
        cumulative_cache_write_tokens: u64,
        cumulative_requests: u64,
    ) -> Self {
        Self::V1(VoucherV1 {
            channel_id,
            cumulative_amount,
            cumulative_prompt_tokens,
            cumulative_completion_tokens,
            cumulative_cache_read_tokens,
            cumulative_cache_write_tokens,
            cumulative_requests,
        })
    }

    /// Voucher with no usage breakdown — useful in tests and for the
    /// `WithdrawAfterTimeout` codepath that doesn't need detail.
    pub const fn new_amount_only(channel_id: ObjectID, cumulative_amount: u64) -> Self {
        Self::V1(VoucherV1 {
            channel_id,
            cumulative_amount,
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
        })
    }

    /// `channel_id` of the voucher, regardless of variant.
    pub fn channel_id(&self) -> ObjectID {
        match self {
            Self::V1(v) => v.channel_id,
        }
    }

    /// `cumulative_amount` of the voucher, regardless of variant.
    pub fn cumulative_amount(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_amount,
        }
    }

    pub fn cumulative_prompt_tokens(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_prompt_tokens,
        }
    }

    pub fn cumulative_completion_tokens(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_completion_tokens,
        }
    }

    pub fn cumulative_cache_read_tokens(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_cache_read_tokens,
        }
    }

    pub fn cumulative_cache_write_tokens(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_cache_write_tokens,
        }
    }

    pub fn cumulative_requests(&self) -> u64 {
        match self {
            Self::V1(v) => v.cumulative_requests,
        }
    }
}

/// Off-chain HTTP-bound voucher signed for the inference marketplace
/// (proxy → provider HTTP path). Same primitive as [`Voucher`]
/// (Ed25519/MultiSig over `IntentMessage<HttpVoucher>` under
/// [`crate::intent::IntentScope::PaymentVoucherHttp`]) but additionally
/// commits to the per-request HTTP context so an adversarial provider
/// can't replay the signature against a different request.
///
/// Composition: the [`HttpVoucher`] *embeds* the on-chain [`Voucher`]
/// the payer signed (full cumulative + usage breakdown), so the provider
/// stores exactly the bytes the payer signed — no independent re-derivation
/// of usage counters that has to byte-match the payer's view across every
/// edge case (multiple SSE `usage` events, missed reconciles, upstream
/// parsing differences, etc.). The chain still verifies the on-chain sig
/// over the embedded [`Voucher`] alone; the HTTP sig adds the per-request
/// HTTP bindings on top.
///
/// Versioned via the enum tag — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpVoucher {
    V1(HttpVoucherV1),
}

/// V1 layout of [`HttpVoucher`].
///
/// Field rationale:
///   - `voucher`: the full on-chain [`Voucher`] the payer signed
///     (cumulative_amount + usage breakdown). The provider stores this
///     verbatim and submits it at settle time, so the chain's voucher
///     sig verification matches by construction.
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
pub struct HttpVoucherV1 {
    pub voucher: Voucher,
    pub expires_ms: TimestampMs,
    pub body_sha256: [u8; 32],
    pub request_id_sha256: [u8; 32],
    pub method_path_sha256: [u8; 32],
}

impl HttpVoucher {
    pub const fn new(
        voucher: Voucher,
        expires_ms: TimestampMs,
        body_sha256: [u8; 32],
        request_id_sha256: [u8; 32],
        method_path_sha256: [u8; 32],
    ) -> Self {
        Self::V1(HttpVoucherV1 {
            voucher,
            expires_ms,
            body_sha256,
            request_id_sha256,
            method_path_sha256,
        })
    }

    /// Convenience constructor that hashes the per-request strings so
    /// callers don't need to import sha2 directly.
    pub fn from_request(
        voucher: Voucher,
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
        Self::new(voucher, expires_ms, body_sha256, request_id_sha256, method_path_sha256)
    }

    /// The embedded on-chain [`Voucher`] — the exact bytes a settle
    /// will submit. Provider callers should store these fields verbatim
    /// rather than recomputing them from local observation.
    pub fn voucher(&self) -> &Voucher {
        match self {
            Self::V1(v) => &v.voucher,
        }
    }

    pub fn channel_id(&self) -> ObjectID {
        self.voucher().channel_id()
    }

    pub fn cumulative_amount(&self) -> u64 {
        self.voucher().cumulative_amount()
    }

    pub fn expires_ms(&self) -> TimestampMs {
        match self {
            Self::V1(v) => v.expires_ms,
        }
    }

    pub fn body_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(v) => v.body_sha256,
        }
    }

    pub fn request_id_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(v) => v.request_id_sha256,
        }
    }

    pub fn method_path_sha256(&self) -> [u8; 32] {
        match self {
            Self::V1(v) => v.method_path_sha256,
        }
    }
}

/// Snapshot of an offering's price + SLA columns at the moment a
/// channel is opened against it. The executor builds one of these
/// from the loaded `Offering` shared input and hands it to
/// `Channel::new`. Decoupling the shape from `ChannelV1`'s constructor
/// keeps the call sites readable when the field count grows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferingSnapshot {
    pub model_id: String,
    pub prompt_micros_per_1k: u64,
    pub completion_micros_per_1k: u64,
    pub cache_read_micros_per_1k: u64,
    pub cache_write_micros_per_1k: u64,
    pub request_micros: u64,
    pub ttft_bound_ms: u32,
    pub ttot_bound_ms: u32,
}

impl Channel {
    /// Construct a fresh v1 channel for `OpenChannel` execution.
    /// `settled_amount` starts at 0 and `close_requested_at_ms` at None.
    pub fn new(
        payer: SomaAddress,
        payee: SomaAddress,
        authorized_signer: SomaAddress,
        token: CoinType,
        deposit: u64,
        offering: OfferingSnapshot,
    ) -> Self {
        Self::V1(ChannelV1 {
            payer,
            payee,
            authorized_signer,
            token,
            deposit,
            settled_amount: 0,
            close_requested_at_ms: None,
            model_id: offering.model_id,
            prompt_micros_per_1k: offering.prompt_micros_per_1k,
            completion_micros_per_1k: offering.completion_micros_per_1k,
            cache_read_micros_per_1k: offering.cache_read_micros_per_1k,
            cache_write_micros_per_1k: offering.cache_write_micros_per_1k,
            request_micros: offering.request_micros,
            ttft_bound_ms: offering.ttft_bound_ms,
            ttot_bound_ms: offering.ttot_bound_ms,
        })
    }

    /// Construct a test channel with default-zero price snapshot.
    /// Convenience for downstream tests and SDK fixtures that don't
    /// care about pricing — they pass `model_id` for binding but
    /// don't exercise settlement math. Intentionally not cfg-gated so
    /// downstream crates' tests can call it without a feature flag.
    pub fn new_for_testing(
        payer: SomaAddress,
        payee: SomaAddress,
        authorized_signer: SomaAddress,
        token: CoinType,
        deposit: u64,
    ) -> Self {
        Self::new(
            payer,
            payee,
            authorized_signer,
            token,
            deposit,
            OfferingSnapshot {
                model_id: "test/model".to_string(),
                prompt_micros_per_1k: 0,
                completion_micros_per_1k: 0,
                cache_read_micros_per_1k: 0,
                cache_write_micros_per_1k: 0,
                request_micros: 0,
                ttft_bound_ms: 0,
                ttot_bound_ms: 0,
            },
        )
    }

    pub fn payer(&self) -> SomaAddress {
        match self {
            Self::V1(c) => c.payer,
        }
    }

    pub fn payee(&self) -> SomaAddress {
        match self {
            Self::V1(c) => c.payee,
        }
    }

    pub fn authorized_signer(&self) -> SomaAddress {
        match self {
            Self::V1(c) => c.authorized_signer,
        }
    }

    pub fn token(&self) -> CoinType {
        match self {
            Self::V1(c) => c.token,
        }
    }

    pub fn deposit(&self) -> u64 {
        match self {
            Self::V1(c) => c.deposit,
        }
    }

    pub fn settled_amount(&self) -> u64 {
        match self {
            Self::V1(c) => c.settled_amount,
        }
    }

    pub fn close_requested_at_ms(&self) -> Option<TimestampMs> {
        match self {
            Self::V1(c) => c.close_requested_at_ms,
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::V1(c) => c.model_id.as_str(),
        }
    }

    pub fn prompt_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(c) => c.prompt_micros_per_1k,
        }
    }

    pub fn completion_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(c) => c.completion_micros_per_1k,
        }
    }

    pub fn cache_read_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(c) => c.cache_read_micros_per_1k,
        }
    }

    pub fn cache_write_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(c) => c.cache_write_micros_per_1k,
        }
    }

    pub fn request_micros(&self) -> u64 {
        match self {
            Self::V1(c) => c.request_micros,
        }
    }

    pub fn ttft_bound_ms(&self) -> u32 {
        match self {
            Self::V1(c) => c.ttft_bound_ms,
        }
    }

    pub fn ttot_bound_ms(&self) -> u32 {
        match self {
            Self::V1(c) => c.ttot_bound_ms,
        }
    }

    /// `deposit + settled_amount` — the maximum legal `cumulative_amount`
    /// a voucher could carry. Anything beyond this implies overspending
    /// the escrow.
    pub fn max_cumulative_amount(&self) -> u64 {
        self.deposit().saturating_add(self.settled_amount())
    }

    /// Remainder that would flow back to the payer on a close right
    /// now (i.e., the live deposit). Equal to `self.deposit()`. Named
    /// for clarity at call sites.
    pub fn remainder_to_payer(&self) -> u64 {
        self.deposit()
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
    ///
    /// BCS auto-dispatches on the leading variant tag — readers built
    /// against a future `Channel::V2(...)` will accept v2 payloads,
    /// and stale readers built against v1 only will return `None` for
    /// any unknown variant tag (BCS errors on unknown enum
    /// discriminants).
    pub fn as_channel(&self) -> Option<Channel> {
        if *self.data.object_type() != ObjectType::Channel {
            return None;
        }
        bcs::from_bytes::<Channel>(self.data.contents()).ok()
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

    /// BCS encodes an enum as `uleb128(variant_index) || payload`. For
    /// the V1-only Channel/Voucher/HttpVoucher today, the leading byte
    /// is always 0 (the V1 tag). Pinning this makes the wire format
    /// auditable and gives us a bright-line check for the day a v2
    /// arm lands.
    const ENUM_V1_TAG: u8 = 0;

    /// Channel BCS round-trip: serialize → deserialize → equal, and
    /// the leading byte is the V1 enum tag.
    #[test]
    fn channel_bcs_round_trip() {
        let ch = Channel::new_for_testing(
            SomaAddress::random(),
            SomaAddress::random(),
            SomaAddress::random(),
            CoinType::Usdc,
            1_000_000,
        );
        let bytes = bcs::to_bytes(&ch).expect("Channel serializes");
        assert_eq!(bytes[0], ENUM_V1_TAG, "leading byte must be the V1 tag");
        let decoded: Channel = bcs::from_bytes(&bytes).expect("Channel deserializes");
        assert_eq!(decoded, ch);
        assert_eq!(decoded.settled_amount(), 0);
        assert_eq!(decoded.close_requested_at_ms(), None);
    }

    /// `as_channel` defers to BCS, which errors on an unknown enum
    /// discriminant. A reader built against V1 only must return None
    /// rather than misinterpreting future variants as v1.
    #[test]
    fn as_channel_rejects_unknown_variant() {
        let id = ObjectID::random();
        let ch = Channel::new_for_testing(
            SomaAddress::random(),
            SomaAddress::random(),
            SomaAddress::random(),
            CoinType::Usdc,
            100,
        );
        let mut obj = Object::new_channel_for_testing(id, ch);
        // Tamper the leading variant byte to an unknown value.
        let mut bytes = obj.data.contents().to_vec();
        bytes[0] = 0xFF;
        obj.data.update_contents(bytes);
        assert!(obj.as_channel().is_none(), "as_channel must reject unknown variant");
    }

    /// Voucher BCS round-trip — the on-the-wire representation must
    /// be stable.
    #[test]
    fn voucher_bcs_round_trip() {
        let v = Voucher::new_amount_only(ObjectID::random(), 12_345);
        let bytes = bcs::to_bytes(&v).expect("Voucher serializes");
        assert_eq!(bytes[0], ENUM_V1_TAG, "leading byte must be the V1 tag");
        let decoded: Voucher = bcs::from_bytes(&bytes).expect("Voucher deserializes");
        assert_eq!(decoded, v);
        assert_eq!(decoded.cumulative_amount(), 12_345);
    }

    /// HttpVoucher BCS round-trip — same wire-format pinning as
    /// `voucher_bcs_round_trip`.
    #[test]
    fn http_voucher_bcs_round_trip() {
        let chan = ObjectID::random();
        let voucher = Voucher::new(chan, 42, 11, 7, 0, 0, 1);
        let v = HttpVoucher::from_request(
            voucher,
            1_700_000_000_000,
            b"body bytes",
            "request-id-abc",
            "POST",
            "/v1/chat/completions",
        );
        let bytes = bcs::to_bytes(&v).expect("HttpVoucher serializes");
        assert_eq!(bytes[0], ENUM_V1_TAG, "leading byte must be the V1 tag");
        let decoded: HttpVoucher = bcs::from_bytes(&bytes).expect("HttpVoucher deserializes");
        assert_eq!(decoded, v);
        assert_eq!(decoded.cumulative_amount(), 42);
        assert_eq!(decoded.expires_ms(), 1_700_000_000_000);
        // The embedded voucher carries the full usage breakdown verbatim.
        assert_eq!(decoded.voucher().cumulative_prompt_tokens(), 11);
        assert_eq!(decoded.voucher().cumulative_completion_tokens(), 7);
        assert_eq!(decoded.voucher().cumulative_requests(), 1);
    }

    /// Object<->Channel helpers: type, ownership, contents survive
    /// the construction → as_channel round-trip.
    #[test]
    fn object_channel_helpers() {
        let id = ObjectID::random();
        let ch = Channel::new_for_testing(
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
        let ch1 = Channel::new_for_testing(
            SomaAddress::ZERO,
            SomaAddress::ZERO,
            SomaAddress::ZERO,
            CoinType::Usdc,
            100,
        );
        let mut obj = Object::new_channel_for_testing(id, ch1);

        // Mutate via destructuring — the V1-only enum gives an
        // irrefutable pattern today; future v2 forces this site to
        // match exhaustively.
        let mut updated = obj.as_channel().unwrap();
        let Channel::V1(inner) = &mut updated;
        inner.deposit = 80;
        inner.settled_amount = 20;
        obj.set_channel_data(&updated);

        assert_eq!(obj.id(), id, "id must not change");
        assert_eq!(*obj.type_(), ObjectType::Channel, "type must not change");
        let read_back = obj.as_channel().unwrap();
        assert_eq!(read_back.deposit(), 80);
        assert_eq!(read_back.settled_amount(), 20);
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
        let voucher = Voucher::new_amount_only(ObjectID::random(), 42);
        let intent_msg = IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);

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
        let voucher = Voucher::new_amount_only(ObjectID::random(), 1);
        let intent_msg = IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
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
        let original = Voucher::new_amount_only(ObjectID::random(), 100);
        let intent_msg =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), original);
        let sig = Signature::new_secure(&intent_msg, &kp);

        // Forge a higher amount but use the same signature — must reject.
        let tampered = Voucher::new_amount_only(original.channel_id(), 9999);
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

        let voucher_a = Voucher::new_amount_only(chan_a, 50);
        let intent_msg_a =
            IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher_a);
        let sig_a = Signature::new_secure(&intent_msg_a, &kp);

        // The same `cumulative_amount=50` for chan_b must not verify
        // with `sig_a` — the channel_id is part of the hashed payload.
        let voucher_b = Voucher::new_amount_only(chan_b, 50);
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
        let voucher = Voucher::new_amount_only(ObjectID::ZERO, 0);
        let im_voucher = IntentMessage::new(Intent::soma_app(IntentScope::PaymentVoucher), voucher);
        // ProofOfPossession over the same struct shape (BCS bytes are
        // the same content but the intent prefix differs).
        let im_pop = IntentMessage::new(Intent::soma_app(IntentScope::ProofOfPossession), voucher);
        assert_ne!(
            bcs::to_bytes(&im_voucher).unwrap(),
            bcs::to_bytes(&im_pop).unwrap(),
            "intent prefix must domain-separate scopes"
        );
    }
}
