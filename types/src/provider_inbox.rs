// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-payee channel-count tracker.
//!
//! A `ProviderInbox` is a shared on-chain object keyed deterministically
//! on the channel's `payee`. It records, per payer, how many open
//! payment channels that payer currently holds against this payee.
//! `OpenChannel` increments the count for the signer (the would-be
//! payer); `WithdrawAfterTimeout` decrements it. The executor rejects
//! `OpenChannel` if the count for that payer would exceed
//! [`MAX_CHANNELS_PER_PAIR`].
//!
//! Why this exists: without a per-pair cap, an attacker can mint
//! arbitrarily many `Channel` shared objects against any one payee from
//! a single address (or a handful of addresses), bloating shared-object
//! state for free besides gas. Capping the per-(payer, payee) count at
//! a small number — large enough for legitimate hot/cold-key splits or
//! per-SKU fan-out, small enough to make spam unprofitable — closes the
//! grief vector. The cap is fixed in code (rather than a SystemState
//! parameter) because changing it later doesn't require a chain-wide
//! migration: the cap is consulted only at `OpenChannel` time, and
//! existing inbox entries already exceeding a future lower cap simply
//! decay through normal channel withdrawal.
//!
//! Versioned via the enum tag — same pattern as `types::channel`
//! (`Channel`, `Voucher`, `HttpVoucher`). BCS encodes an enum as
//! `uleb128(variant_index) || payload`; the leading byte gives stale
//! readers a structural signal to bail safely on a future v2 layout.
//!
//! See `authority::execution::channel` for the executor side.

use fastcrypto::hash::{Blake2b256, HashFunction};
use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::digests::TransactionDigest;
use crate::object::{Object, ObjectData, ObjectID, ObjectType, Owner, Version};

/// Default per-(payer, payee) channel cap. The live value is read
/// from `SystemState.parameters.max_channels_per_pair` so it can be
/// tuned per-network (testnet vs. mainnet) without a binary upgrade;
/// this constant is just the genesis default and a fallback for
/// tests that don't bring up a SystemState.
pub const DEFAULT_MAX_CHANNELS_PER_PAIR: u32 = 8;

/// Domain tag for `ProviderInbox` ID derivation. Distinct from any
/// other derived-ID domain tag in this crate so a structural collision
/// is impossible regardless of the input shape.
const PROVIDER_INBOX_DOMAIN_TAG: &[u8] = b"soma::provider_inbox::v1";

/// Per-payee inbox object. See module docs.
///
/// Versioned via the enum tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderInbox {
    V1(ProviderInboxV1),
}

/// V1 layout of [`ProviderInbox`].
///
/// `open_counts` is stored as a `Vec<(SomaAddress, u32)>` rather than a
/// `BTreeMap` so the on-disk encoding is independent of any hash-map
/// implementation choice. Entries are kept sorted by payer for stable
/// ordering and binary-search lookup.
///
/// State growth: an entry is added when a payer opens their first
/// channel against this payee, and removed when their count reaches
/// zero. The map's size is therefore bounded by the number of
/// distinct payers with at least one open channel against this payee
/// — gas costs at OpenChannel time provide the throttle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInboxV1 {
    /// The payee this inbox is keyed on. Stored explicitly (rather
    /// than only encoded in the object id) so a reader can audit
    /// the binding without recomputing the derived id.
    pub payee: SomaAddress,
    /// `(payer, count)` entries, sorted by payer. Invariant:
    /// every `count > 0` (zero entries are removed on decrement).
    pub open_counts: Vec<(SomaAddress, u32)>,
}

impl ProviderInbox {
    /// Construct an empty v1 inbox keyed on `payee`.
    pub fn new(payee: SomaAddress) -> Self {
        Self::V1(ProviderInboxV1 { payee, open_counts: Vec::new() })
    }

    /// Derive the canonical `ObjectID` for the inbox keyed on `payee`.
    /// Mirrors `Provider::derive_id` and `BalanceAccumulator::derive_id`
    /// — Blake2b-256 over a length-prefixed domain tag plus the
    /// length-prefixed payee bytes. The length prefixes ensure that
    /// no future field addition can produce a structural collision
    /// across domains.
    pub fn derive_id(payee: SomaAddress) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update((PROVIDER_INBOX_DOMAIN_TAG.len() as u32).to_le_bytes());
        hasher.update(PROVIDER_INBOX_DOMAIN_TAG);
        let addr_bytes = payee.to_vec();
        hasher.update((addr_bytes.len() as u32).to_le_bytes());
        hasher.update(&addr_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        ObjectID::new(digest)
    }

    /// Payee this inbox is keyed on, regardless of variant.
    pub fn payee(&self) -> SomaAddress {
        match self {
            Self::V1(i) => i.payee,
        }
    }

    /// Current open-channel count for `payer`. Returns 0 if `payer`
    /// has no entry.
    pub fn count_for(&self, payer: SomaAddress) -> u32 {
        match self {
            Self::V1(i) => i
                .open_counts
                .binary_search_by_key(&payer, |(p, _)| *p)
                .map(|idx| i.open_counts[idx].1)
                .unwrap_or(0),
        }
    }

    /// Increment the count for `payer` by 1, capped at `max`.
    /// Returns `Err(current)` if the count is already at `max` — the
    /// caller surfaces this as `ChannelTooManyOpenForPair`. On
    /// success returns the new count.
    ///
    /// `max` is supplied by the caller (read from
    /// `SystemState.parameters.max_channels_per_pair` on chain) so
    /// the inbox itself stays free of policy.
    pub fn try_increment(&mut self, payer: SomaAddress, max: u32) -> Result<u32, u32> {
        match self {
            Self::V1(i) => match i.open_counts.binary_search_by_key(&payer, |(p, _)| *p) {
                Ok(idx) => {
                    let cur = i.open_counts[idx].1;
                    if cur >= max {
                        return Err(cur);
                    }
                    i.open_counts[idx].1 = cur + 1;
                    Ok(cur + 1)
                }
                Err(idx) => {
                    if max == 0 {
                        return Err(0);
                    }
                    i.open_counts.insert(idx, (payer, 1));
                    Ok(1)
                }
            },
        }
    }

    /// Saturating-decrement the count for `payer`. Removes the entry
    /// when the count reaches zero so the inbox doesn't accumulate
    /// dead rows. A no-op if `payer` has no entry.
    pub fn decrement(&mut self, payer: SomaAddress) {
        match self {
            Self::V1(i) => {
                if let Ok(idx) = i.open_counts.binary_search_by_key(&payer, |(p, _)| *p) {
                    let new = i.open_counts[idx].1.saturating_sub(1);
                    if new == 0 {
                        i.open_counts.remove(idx);
                    } else {
                        i.open_counts[idx].1 = new;
                    }
                }
            }
        }
    }

    /// True iff the inbox is empty (no payer entries).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::V1(i) => i.open_counts.is_empty(),
        }
    }
}

impl Object {
    /// Build a new v1 ProviderInbox shared object. Uses
    /// `OBJECT_START_VERSION` for `initial_shared_version` so all
    /// inboxes have a predictable shared-version key — the SDK
    /// declares the inbox as a mutable shared input at that version
    /// and the scheduler resolves to the live version.
    pub fn new_provider_inbox(
        inbox: ProviderInbox,
        previous_transaction: TransactionDigest,
    ) -> Self {
        let id = ProviderInbox::derive_id(inbox.payee());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::ProviderInbox,
            Version::MIN,
            bcs::to_bytes(&inbox).expect("ProviderInbox serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            previous_transaction,
        )
    }

    /// Test/debug constructor that places the object at
    /// `OBJECT_START_VERSION` so it can be loaded as an input without
    /// going through full execution.
    pub fn new_provider_inbox_for_testing(inbox: ProviderInbox) -> Self {
        let id = ProviderInbox::derive_id(inbox.payee());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::ProviderInbox,
            crate::object::OBJECT_START_VERSION,
            bcs::to_bytes(&inbox).expect("ProviderInbox serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            TransactionDigest::default(),
        )
    }

    /// If this object is a ProviderInbox, deserialize and return it.
    pub fn as_provider_inbox(&self) -> Option<ProviderInbox> {
        if *self.data.object_type() != ObjectType::ProviderInbox {
            return None;
        }
        bcs::from_bytes::<ProviderInbox>(self.data.contents()).ok()
    }

    /// Overwrite a ProviderInbox object's contents.
    pub fn set_provider_inbox_data(&mut self, inbox: &ProviderInbox) {
        debug_assert_eq!(
            *self.data.object_type(),
            ObjectType::ProviderInbox,
            "set_provider_inbox_data called on non-ProviderInbox object"
        );
        self.update_contents(inbox);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(seed: u8) -> SomaAddress {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SomaAddress::new(bytes)
    }

    /// Default cap for tests — matches `DEFAULT_MAX_CHANNELS_PER_PAIR`
    /// but spelled locally so tests don't silently follow if the
    /// const drifts.
    const TEST_CAP: u32 = 8;

    #[test]
    fn provider_inbox_bcs_round_trip() {
        let mut inbox = ProviderInbox::new(addr(1));
        inbox.try_increment(addr(2), TEST_CAP).unwrap();
        inbox.try_increment(addr(3), TEST_CAP).unwrap();
        let bytes = bcs::to_bytes(&inbox).expect("serializes");
        // Leading byte is the V1 enum tag (= 0).
        assert_eq!(bytes[0], 0, "leading byte must be the V1 tag");
        let decoded: ProviderInbox = bcs::from_bytes(&bytes).expect("deserializes");
        assert_eq!(decoded, inbox);
    }

    #[test]
    fn derive_id_deterministic_and_unique() {
        let a = ProviderInbox::derive_id(addr(1));
        let b = ProviderInbox::derive_id(addr(1));
        let c = ProviderInbox::derive_id(addr(2));
        assert_eq!(a, b, "same payee must produce the same id");
        assert_ne!(a, c, "different payees must produce different ids");
    }

    /// `derive_id` must not collide with `Provider::derive_id` even
    /// when given the same address — the domain tag prefix prevents
    /// cross-domain collisions structurally.
    #[test]
    fn derive_id_distinct_from_provider() {
        let inbox_id = ProviderInbox::derive_id(addr(1));
        let provider_id = crate::provider::Provider::derive_id(addr(1));
        assert_ne!(inbox_id, provider_id);
    }

    #[test]
    fn try_increment_inserts_then_increments() {
        let mut inbox = ProviderInbox::new(addr(1));
        let payer = addr(2);
        assert_eq!(inbox.count_for(payer), 0);
        assert_eq!(inbox.try_increment(payer, TEST_CAP), Ok(1));
        assert_eq!(inbox.count_for(payer), 1);
        assert_eq!(inbox.try_increment(payer, TEST_CAP), Ok(2));
        assert_eq!(inbox.count_for(payer), 2);
    }

    #[test]
    fn try_increment_rejects_at_cap() {
        let mut inbox = ProviderInbox::new(addr(1));
        let payer = addr(2);
        for expected in 1..=TEST_CAP {
            assert_eq!(inbox.try_increment(payer, TEST_CAP), Ok(expected));
        }
        // 9th attempt rejected.
        assert_eq!(inbox.try_increment(payer, TEST_CAP), Err(TEST_CAP));
        // Count unchanged.
        assert_eq!(inbox.count_for(payer), TEST_CAP);
    }

    /// `try_increment` honors the caller-supplied cap, so a tighter
    /// per-network limit (e.g. testnet runs at 4) is enforced without
    /// any change to the inbox type.
    #[test]
    fn try_increment_honors_lower_cap() {
        let mut inbox = ProviderInbox::new(addr(1));
        let payer = addr(2);
        for expected in 1..=4 {
            assert_eq!(inbox.try_increment(payer, 4), Ok(expected));
        }
        assert_eq!(inbox.try_increment(payer, 4), Err(4));
    }

    /// Different payers each get their own bucket — one payer at the
    /// cap doesn't block another.
    #[test]
    fn try_increment_independent_per_payer() {
        let mut inbox = ProviderInbox::new(addr(1));
        let alice = addr(2);
        let bob = addr(3);
        for _ in 0..TEST_CAP {
            inbox.try_increment(alice, TEST_CAP).unwrap();
        }
        assert_eq!(inbox.try_increment(alice, TEST_CAP), Err(TEST_CAP));
        // Bob still has full headroom.
        assert_eq!(inbox.try_increment(bob, TEST_CAP), Ok(1));
    }

    #[test]
    fn decrement_removes_entry_at_zero() {
        let mut inbox = ProviderInbox::new(addr(1));
        let payer = addr(2);
        inbox.try_increment(payer, TEST_CAP).unwrap();
        inbox.try_increment(payer, TEST_CAP).unwrap();
        assert_eq!(inbox.count_for(payer), 2);
        inbox.decrement(payer);
        assert_eq!(inbox.count_for(payer), 1);
        inbox.decrement(payer);
        assert_eq!(inbox.count_for(payer), 0);
        // Entry is removed — inbox is empty.
        assert!(inbox.is_empty());
    }

    /// Decrementing a payer with no entry is a no-op (defensive).
    #[test]
    fn decrement_unknown_payer_is_noop() {
        let mut inbox = ProviderInbox::new(addr(1));
        inbox.decrement(addr(99));
        assert!(inbox.is_empty());
    }

    /// Object<->ProviderInbox helpers: type, ownership, and contents
    /// survive the construction → as_provider_inbox round-trip.
    #[test]
    fn object_helpers() {
        let mut inbox = ProviderInbox::new(addr(1));
        inbox.try_increment(addr(2), TEST_CAP).unwrap();
        let obj = Object::new_provider_inbox_for_testing(inbox.clone());
        assert_eq!(obj.id(), ProviderInbox::derive_id(addr(1)));
        assert_eq!(*obj.type_(), ObjectType::ProviderInbox);
        assert!(obj.is_shared(), "ProviderInbox must be a shared object");
        assert_eq!(obj.as_provider_inbox().expect("decode"), inbox);
    }

    #[test]
    fn set_provider_inbox_data_round_trips() {
        let inbox1 = ProviderInbox::new(addr(1));
        let mut obj = Object::new_provider_inbox_for_testing(inbox1);

        let mut inbox2 = obj.as_provider_inbox().unwrap();
        inbox2.try_increment(addr(7), TEST_CAP).unwrap();
        obj.set_provider_inbox_data(&inbox2);

        let read_back = obj.as_provider_inbox().unwrap();
        assert_eq!(read_back.count_for(addr(7)), 1);
    }
}
