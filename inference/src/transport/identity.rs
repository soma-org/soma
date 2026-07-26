// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Mapping an iroh transport identity to a Soma on-chain identity.
//!
//! An iroh [`EndpointId`] and a Soma `SomaAddress` are two views of the same
//! Ed25519 key: the `EndpointId` *is* the raw public key, and the address is
//! `Blake2b256(ED25519_flag(0x00) ‖ pubkey)` — the exact derivation the
//! voucher verifier uses to check a signature against a channel's
//! `authorized_signer`.
//!
//! So when a buyer uses one dedicated Ed25519 key as BOTH its channel
//! `authorized_signer` and its iroh identity (the recommended model), this
//! function maps the authenticated transport peer back to that on-chain
//! signer address — which is what lets the provider enforce
//! `peer == channel.authorized_signer` at the connection boundary.

use ::types::base::SomaAddress;
use ::types::crypto::{PublicKey, SomaKeyPair};
use fastcrypto::ed25519::Ed25519PublicKey;
use fastcrypto::traits::{KeyPair as _, ToFromBytes as _};

use super::EndpointId;

/// Derive an iroh [`iroh::SecretKey`] from a Soma keypair, so the same
/// Ed25519 key is used as BOTH the voucher-signing / `authorized_signer`
/// identity AND the iroh transport identity. This is what makes the
/// provider-side `peer == authorized_signer` binding pass: the buyer dials
/// with the very key it signs vouchers with.
///
/// Only Ed25519 keys can be used as an iroh identity.
pub fn iroh_secret_from_keypair(kp: &SomaKeyPair) -> anyhow::Result<iroh::SecretKey> {
    match kp {
        SomaKeyPair::Ed25519(kp) => {
            let bytes: [u8; 32] = kp
                .copy()
                .private()
                .as_bytes()
                .try_into()
                .map_err(|_| anyhow::anyhow!("ed25519 private key must be 32 bytes"))?;
            Ok(iroh::SecretKey::from_bytes(&bytes))
        }
    }
}

/// Derive the `SomaAddress` corresponding to an iroh [`EndpointId`].
///
/// Errors only if the endpoint id is not a valid Ed25519 point (which iroh
/// guarantees for any real peer, so in practice this is infallible).
pub fn soma_address_from_endpoint_id(peer: &EndpointId) -> anyhow::Result<SomaAddress> {
    let raw: [u8; 32] = *peer.as_bytes();
    let ed = Ed25519PublicKey::from_bytes(&raw)
        .map_err(|e| anyhow::anyhow!("endpoint id is not a valid ed25519 key: {e}"))?;
    Ok(SomaAddress::from(&PublicKey::Ed25519((&ed).into())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An entity holding one Ed25519 secret has a single identity that is
    /// both its iroh `EndpointId` and (via derivation) its `SomaAddress`.
    /// This pins the mapping to the canonical Soma address derivation — the
    /// same one voucher verification uses — so `peer == authorized_signer`
    /// holds exactly when the buyer reuses its signer key as its iroh key.
    #[test]
    fn endpoint_id_maps_to_canonical_soma_address() {
        let seed = [7u8; 32];
        let secret = iroh::SecretKey::from_bytes(&seed);
        let endpoint_id = secret.public();

        // Canonical Soma address for that same raw Ed25519 public key.
        let raw: [u8; 32] = *endpoint_id.as_bytes();
        let ed = Ed25519PublicKey::from_bytes(&raw).unwrap();
        let expected = SomaAddress::from(&PublicKey::Ed25519((&ed).into()));

        assert_eq!(soma_address_from_endpoint_id(&endpoint_id).unwrap(), expected);
    }

    /// The full binding invariant: a Soma keypair's iroh identity derives
    /// back to that keypair's own `SomaAddress`. So a buyer that opens a
    /// channel with `authorized_signer = its address` and dials with the
    /// iroh secret from the same key will satisfy `peer == authorized_signer`.
    #[test]
    fn iroh_secret_from_keypair_round_trips_to_address() {
        use fastcrypto::ed25519::Ed25519KeyPair;
        use fastcrypto::traits::KeyPair as _;

        let seed = [9u8; 32];
        let ed = Ed25519KeyPair::from(
            fastcrypto::ed25519::Ed25519PrivateKey::from_bytes(&seed).unwrap(),
        );
        let soma_addr = SomaAddress::from(&PublicKey::Ed25519((ed.public()).into()));
        let kp = SomaKeyPair::Ed25519(ed);

        let secret = iroh_secret_from_keypair(&kp).unwrap();
        let endpoint_id = secret.public();
        assert_eq!(soma_address_from_endpoint_id(&endpoint_id).unwrap(), soma_addr);
    }

    /// Distinct iroh identities derive to distinct Soma addresses.
    #[test]
    fn distinct_keys_distinct_addresses() {
        let a = iroh::SecretKey::from_bytes(&[1u8; 32]).public();
        let b = iroh::SecretKey::from_bytes(&[2u8; 32]).public();
        assert_ne!(
            soma_address_from_endpoint_id(&a).unwrap(),
            soma_address_from_endpoint_id(&b).unwrap()
        );
    }
}
