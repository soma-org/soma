// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! On-chain provider registry.
//!
//! A `Provider` is a shared object whose ID is deterministically derived
//! from the provider's `SomaAddress`. There is at most one Provider per
//! address; re-registration fails. `RegisterProvider` creates the
//! object, `UpdateProvider` changes the advertised endpoint.
//!
//! Liveness is purely an off-chain question — the proxy probes the
//! advertised endpoint via HTTP. The on-chain registry is the set of
//! addresses *claiming* to provide; HTTP probes determine which are
//! actually responding.
//!
//! See `authority::execution::provider` for the executor.

use fastcrypto::hash::{Blake2b256, HashFunction};
use serde::{Deserialize, Serialize};

use crate::base::SomaAddress;
use crate::digests::TransactionDigest;
use crate::object::{Object, ObjectData, ObjectID, ObjectType, Owner, Version};

/// Domain tag for `Provider` ID derivation. Distinct from any other
/// derived-ID domain (`BalanceAccumulator`, `DelegationAccumulator`,
/// etc.) so collisions are structurally impossible.
const PROVIDER_DOMAIN_TAG: &[u8] = b"soma::provider::v1";

/// On-chain provider record.
///
/// Created by `RegisterProvider`, mutated by `UpdateProvider`. The
/// executor uses `Provider::derive_id(address)` to address the object
/// — so a duplicate `RegisterProvider` from the same signer fails with
/// `ProviderAlreadyExists`, and any caller can compute the canonical
/// id without an indexer round-trip.
///
/// Versioned via the enum tag (matches `Channel` / `Voucher`): BCS
/// emits a leading variant byte so a future v2 forces every read site
/// to match exhaustively. `as_provider` defers to BCS so unknown
/// variant tags safely return `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    V1(ProviderV1),
}

/// V1 layout of [`Provider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderV1 {
    /// Address of the provider. Equal to the signer of the
    /// `RegisterProvider` tx that created this object. Used by the
    /// executor as the authorization key for `UpdateProvider`.
    pub address: SomaAddress,

    /// HTTP(S) endpoint at which the provider serves
    /// `/soma/info` and the OpenAI-compatible `/v1/...` endpoints.
    /// Buyers' proxies dial this URL.
    pub endpoint: String,
}

impl Provider {
    /// Construct a fresh v1 `Provider` for `RegisterProvider` execution.
    pub fn new(address: SomaAddress, endpoint: String) -> Self {
        Self::V1(ProviderV1 { address, endpoint })
    }

    pub fn address(&self) -> SomaAddress {
        match self {
            Self::V1(p) => p.address,
        }
    }

    pub fn endpoint(&self) -> &str {
        match self {
            Self::V1(p) => &p.endpoint,
        }
    }

    pub fn set_endpoint(&mut self, endpoint: String) {
        match self {
            Self::V1(p) => p.endpoint = endpoint,
        }
    }

    /// Derive the canonical `ObjectID` for the provider keyed on
    /// `address`. Mirrors `BalanceAccumulator::derive_id` —
    /// Blake2b-256 over a domain tag + the address bytes. Length
    /// prefixes guard against any future field addition that could
    /// otherwise create a collision.
    pub fn derive_id(address: SomaAddress) -> ObjectID {
        let mut hasher = Blake2b256::default();
        hasher.update((PROVIDER_DOMAIN_TAG.len() as u32).to_le_bytes());
        hasher.update(PROVIDER_DOMAIN_TAG);
        let addr_bytes = address.to_vec();
        hasher.update((addr_bytes.len() as u32).to_le_bytes());
        hasher.update(&addr_bytes);
        let digest: [u8; 32] = hasher.finalize().into();
        ObjectID::new(digest)
    }
}

impl Object {
    /// Build a new `Provider` shared object for `RegisterProvider`.
    /// Uses `OBJECT_START_VERSION` for `initial_shared_version` so all
    /// providers share a predictable shared-version key — clients can
    /// construct subsequent `UpdateProvider` transactions without
    /// looking up the lamport-timestamped version first.
    pub fn new_provider(provider: Provider, previous_transaction: TransactionDigest) -> Self {
        let id = Provider::derive_id(provider.address());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Provider,
            Version::MIN,
            bcs::to_bytes(&provider).expect("Provider serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            previous_transaction,
        )
    }

    /// Test/debug constructor: creates a Provider object at
    /// `OBJECT_START_VERSION` so it can be loaded as an input without
    /// going through full register-tx execution.
    pub fn new_provider_for_testing(provider: Provider) -> Self {
        let id = Provider::derive_id(provider.address());
        let data = ObjectData::new_with_id(
            id,
            ObjectType::Provider,
            crate::object::OBJECT_START_VERSION,
            bcs::to_bytes(&provider).expect("Provider serialization is infallible"),
        );
        Object::new(
            data,
            Owner::Shared { initial_shared_version: crate::object::OBJECT_START_VERSION },
            TransactionDigest::default(),
        )
    }

    /// If this object is a Provider, deserialize and return it.
    /// Returns `None` for non-Provider objects or for Provider bytes
    /// whose variant tag is unknown to this binary (forward-compat).
    pub fn as_provider(&self) -> Option<Provider> {
        if *self.data.object_type() == ObjectType::Provider {
            bcs::from_bytes::<Provider>(self.data.contents()).ok()
        } else {
            None
        }
    }

    /// Overwrite a Provider object's contents. Caller must ensure this
    /// object actually IS a Provider; debug-asserts the type.
    pub fn set_provider_data(&mut self, provider: &Provider) {
        debug_assert_eq!(
            *self.data.object_type(),
            ObjectType::Provider,
            "set_provider_data called on non-Provider object"
        );
        self.update_contents(provider);
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

    #[test]
    fn provider_bcs_round_trip() {
        let p = Provider::new(addr(7), "https://example.com".to_string());
        let bytes = bcs::to_bytes(&p).expect("serializes");
        let decoded: Provider = bcs::from_bytes(&bytes).expect("deserializes");
        assert_eq!(decoded, p);
    }

    /// `derive_id` is deterministic and bijective from address.
    #[test]
    fn derive_id_deterministic_and_unique() {
        let a = Provider::derive_id(addr(1));
        let b = Provider::derive_id(addr(1));
        let c = Provider::derive_id(addr(2));
        assert_eq!(a, b, "same address must produce the same id");
        assert_ne!(a, c, "different addresses must produce different ids");
    }

    /// `derive_id` doesn't collide with `BalanceAccumulator::derive_id`
    /// (which uses a different domain tag + a CoinType field). This is
    /// a structural invariant: the domain tag prefix guarantees no
    /// cross-domain collision regardless of the input shape.
    #[test]
    fn derive_id_distinct_from_balance_accumulator() {
        use crate::accumulator::BalanceAccumulator;
        use crate::object::CoinType;
        let prov = Provider::derive_id(addr(1));
        let bal = BalanceAccumulator::derive_id(addr(1), CoinType::Usdc);
        assert_ne!(prov, bal);
    }

    #[test]
    fn object_provider_helpers() {
        let p = Provider::new(addr(3), "https://prov.test".to_string());
        let obj = Object::new_provider_for_testing(p.clone());
        assert_eq!(obj.id(), Provider::derive_id(p.address()));
        assert_eq!(*obj.type_(), ObjectType::Provider);
        assert!(obj.is_shared(), "Provider must be a shared object");
        assert_eq!(obj.as_provider().expect("deserialize"), p);
    }

    #[test]
    fn set_provider_data_round_trips() {
        let p1 = Provider::new(addr(4), "https://old.example".to_string());
        let mut obj = Object::new_provider_for_testing(p1);
        let mut p2 = obj.as_provider().unwrap();
        p2.set_endpoint("https://new.example".to_string());
        obj.set_provider_data(&p2);
        assert_eq!(*obj.type_(), ObjectType::Provider, "type must not change");
        let read_back = obj.as_provider().unwrap();
        assert_eq!(read_back.endpoint(), "https://new.example");
    }

    #[test]
    fn as_provider_rejects_other_types() {
        let coin = Object::with_id_owner_for_testing(ObjectID::random(), SomaAddress::random());
        assert!(coin.as_provider().is_none());
    }
}
