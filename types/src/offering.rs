// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! On-chain per-(provider, model) offering — the *menu* a provider
//! publishes for a specific model in the protocol-config
//! [`protocol_config::ModelRegistry`]. Created by `RegisterOffering`,
//! mutated by `UpdateOffering` / `DeactivateOffering`.
//!
//! ## Identity
//!
//! Each `Offering` is a shared object whose ID is derived deterministically
//! from `(provider_address, model_id)` via [`Offering::derive_id`]. So:
//!
//! - Only one active offering per `(provider, model)` pair — re-registration
//!   fails at the executor.
//! - Any caller can construct the canonical id client-side without an
//!   indexer round-trip — the shared-input scheduler can declare it
//!   from the tx kind alone.
//!
//! ## Frozen vs. live
//!
//! Offerings are *mutable* — providers update prices and SLA bounds at any
//! cadence they like. Channels however snapshot the offering's price + SLA
//! into [`crate::channel::ChannelV1`] at `OpenChannel` time, so settlement
//! math is deterministic for the channel's lifetime regardless of how the
//! menu moves underneath it.
//!
//! ## Versioning
//!
//! Same enum-tag scheme as `Channel` / `Provider`. The wire format is
//! BCS-encoded `Offering` (a single-variant enum today), so future v2
//! layouts force every reader to match exhaustively.

use fastcrypto::hash::{Blake2b256, HashFunction};
use serde::{Deserialize, Serialize};

use crate::base::{SomaAddress, TimestampMs};
use crate::digests::TransactionDigest;
use crate::object::{Object, ObjectData, ObjectID, ObjectType, Owner, Version};

/// Domain tag for `Offering` ID derivation.
const OFFERING_DOMAIN_TAG: &[u8] = b"soma::offering::v1";

/// On-chain per-(provider, model) offering. See module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Offering {
    V1(OfferingV1),
}

/// V1 layout of [`Offering`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfferingV1 {
    /// Provider's account address. Must equal the signer of every
    /// `Register/Update/DeactivateOffering` tx that touches this row.
    pub provider: SomaAddress,
    /// Canonical model id — must match an active entry in the
    /// protocol-config `ModelRegistry` at the version the executor is
    /// running.
    pub model_id: String,
    /// Price per 1k prompt tokens, in USDC microdollars (1 µUSDC = $1e-6).
    /// `0` is allowed (free tier).
    pub prompt_micros_per_1k: u64,
    /// Price per 1k completion tokens.
    pub completion_micros_per_1k: u64,
    /// Price per 1k cached-prompt-tokens (cache read). Mirrors
    /// OpenRouter's `input_cache_read`.
    pub cache_read_micros_per_1k: u64,
    /// Price per 1k cache-write tokens. Mirrors OpenRouter's
    /// `input_cache_write`.
    pub cache_write_micros_per_1k: u64,
    /// Flat per-request fee in microdollars. Stays constant across
    /// requests on a given channel (channel-snapshot semantics).
    pub request_micros: u64,
    /// Provider's declared TTFT bound — proxy emits a negative
    /// `RateChannel(reason_code=TTFTBreach)` when measured TTFT
    /// exceeds this for a request served on this offering's channel.
    pub ttft_bound_ms: u32,
    /// Provider's declared TTOT bound (avg ms/output token).
    pub ttot_bound_ms: u32,
    /// `false` deprecates the offering — new channels can't open
    /// against it; existing channels keep working until they close.
    pub active: bool,
    /// Wall-clock ms when this offering was last updated. Powers
    /// off-chain "is the listed price fresh?" displays without needing
    /// to scan event history.
    pub updated_at_ms: TimestampMs,
}

impl Offering {
    pub fn new(
        provider: SomaAddress,
        model_id: String,
        prompt_micros_per_1k: u64,
        completion_micros_per_1k: u64,
        cache_read_micros_per_1k: u64,
        cache_write_micros_per_1k: u64,
        request_micros: u64,
        ttft_bound_ms: u32,
        ttot_bound_ms: u32,
        updated_at_ms: TimestampMs,
    ) -> Self {
        Self::V1(OfferingV1 {
            provider,
            model_id,
            prompt_micros_per_1k,
            completion_micros_per_1k,
            cache_read_micros_per_1k,
            cache_write_micros_per_1k,
            request_micros,
            ttft_bound_ms,
            ttot_bound_ms,
            active: true,
            updated_at_ms,
        })
    }

    pub fn provider(&self) -> SomaAddress {
        match self {
            Self::V1(o) => o.provider,
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::V1(o) => o.model_id.as_str(),
        }
    }

    pub fn prompt_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(o) => o.prompt_micros_per_1k,
        }
    }

    pub fn completion_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(o) => o.completion_micros_per_1k,
        }
    }

    pub fn cache_read_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(o) => o.cache_read_micros_per_1k,
        }
    }

    pub fn cache_write_micros_per_1k(&self) -> u64 {
        match self {
            Self::V1(o) => o.cache_write_micros_per_1k,
        }
    }

    pub fn request_micros(&self) -> u64 {
        match self {
            Self::V1(o) => o.request_micros,
        }
    }

    pub fn ttft_bound_ms(&self) -> u32 {
        match self {
            Self::V1(o) => o.ttft_bound_ms,
        }
    }

    pub fn ttot_bound_ms(&self) -> u32 {
        match self {
            Self::V1(o) => o.ttot_bound_ms,
        }
    }

    pub fn active(&self) -> bool {
        match self {
            Self::V1(o) => o.active,
        }
    }

    pub fn updated_at_ms(&self) -> TimestampMs {
        match self {
            Self::V1(o) => o.updated_at_ms,
        }
    }

    /// Canonical `ObjectID` for an offering keyed on `(provider, model_id)`.
    /// Mirrors `Provider::derive_id` — Blake2b-256 over a domain tag +
    /// length-prefixed components.
    pub fn derive_id(provider: SomaAddress, model_id: &str) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update((OFFERING_DOMAIN_TAG.len() as u32).to_le_bytes());
        hasher.update(OFFERING_DOMAIN_TAG);
        let addr_bytes = provider.to_vec();
        hasher.update((addr_bytes.len() as u32).to_le_bytes());
        hasher.update(&addr_bytes);
        let id_bytes = model_id.as_bytes();
        hasher.update((id_bytes.len() as u32).to_le_bytes());
        hasher.update(id_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        ObjectID::new(digest)
    }
}

impl Object {
    /// Build a new `Offering` shared object for `RegisterOffering`.
    pub fn new_offering(offering: Offering, previous_transaction: TransactionDigest) -> Self {
        let id = Offering::derive_id(offering.provider(), offering.model_id());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Offering,
            Version::MIN,
            bcs::to_bytes(&offering).expect("Offering serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            previous_transaction,
        )
    }

    /// Test/debug constructor for an Offering object at
    /// `OBJECT_START_VERSION` (so tests can preload one without a
    /// register-tx execution).
    pub fn new_offering_for_testing(offering: Offering) -> Self {
        let id = Offering::derive_id(offering.provider(), offering.model_id());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Offering,
            crate::object::OBJECT_START_VERSION,
            bcs::to_bytes(&offering).expect("Offering serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            TransactionDigest::default(),
        )
    }

    /// If this object is an Offering, deserialize and return it.
    pub fn as_offering(&self) -> Option<Offering> {
        if *self.data.object_type() == ObjectType::Offering {
            bcs::from_bytes::<Offering>(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Overwrite an Offering object's contents.
    pub fn set_offering_data(&mut self, offering: &Offering) {
        debug_assert_eq!(
            *self.data.object_type(),
            ObjectType::Offering,
            "set_offering_data called on non-Offering object"
        );
        self.update_contents(offering);
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

    fn sample(provider: SomaAddress) -> Offering {
        Offering::new(
            provider,
            "anthropic/claude-opus-4".to_string(),
            8_000,
            24_000,
            800,
            8_000,
            0,
            10_000,
            50,
            1_700_000_000_000,
        )
    }

    #[test]
    fn offering_bcs_round_trip() {
        let o = sample(addr(1));
        let bytes = bcs::to_bytes(&o).expect("serializes");
        assert_eq!(bytes[0], 0, "leading byte must be the V1 enum tag");
        let decoded: Offering = bcs::from_bytes(&bytes).expect("deserializes");
        assert_eq!(decoded, o);
    }

    #[test]
    fn derive_id_keyed_on_provider_and_model() {
        let p1 = addr(1);
        let p2 = addr(2);
        let a = Offering::derive_id(p1, "anthropic/claude-opus-4");
        let b = Offering::derive_id(p1, "anthropic/claude-opus-4");
        let c = Offering::derive_id(p2, "anthropic/claude-opus-4");
        let d = Offering::derive_id(p1, "openai/gpt-5");
        assert_eq!(a, b, "deterministic");
        assert_ne!(a, c, "different provider → different id");
        assert_ne!(a, d, "different model → different id");
    }

    #[test]
    fn derive_id_distinct_from_provider_domain() {
        use crate::provider::Provider;
        let a = Offering::derive_id(addr(1), "some-model");
        let p = Provider::derive_id(addr(1));
        assert_ne!(a, p, "domain tags must keep cross-type ids disjoint");
    }

    #[test]
    fn object_offering_helpers() {
        let o = sample(addr(7));
        let obj = Object::new_offering_for_testing(o.clone());
        assert_eq!(obj.id(), Offering::derive_id(o.provider(), o.model_id()));
        assert_eq!(*obj.type_(), ObjectType::Offering);
        assert!(obj.is_shared(), "Offering must be a shared object");
        assert_eq!(obj.as_offering().expect("deserialize"), o);
    }

    #[test]
    fn set_offering_data_round_trips() {
        let o1 = sample(addr(8));
        let mut obj = Object::new_offering_for_testing(o1);
        let mut o2 = obj.as_offering().unwrap();
        match &mut o2 {
            Offering::V1(inner) => {
                inner.prompt_micros_per_1k = 9_000;
                inner.active = false;
            }
        }
        obj.set_offering_data(&o2);
        let read_back = obj.as_offering().unwrap();
        assert_eq!(read_back.prompt_micros_per_1k(), 9_000);
        assert!(!read_back.active());
    }

    #[test]
    fn as_offering_rejects_other_types() {
        let coin = Object::with_id_owner_for_testing(ObjectID::random(), SomaAddress::random());
        assert!(coin.as_offering().is_none());
    }
}
