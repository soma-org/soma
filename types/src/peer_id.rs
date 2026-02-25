// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use fastcrypto::{ed25519::Ed25519PublicKey, traits::ToFromBytes};

use crate::crypto::NetworkPublicKey;

/// Length of a PeerId, based on the length of an ed25519 public key
const PEER_ID_LENGTH: usize = 32;

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PeerId(pub [u8; PEER_ID_LENGTH]);

impl PeerId {
    pub fn short_display(&self, len: u8) -> impl std::fmt::Display + '_ {
        ShortPeerId(self, len)
    }

    #[cfg(test)]
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let mut bytes = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rng, &mut bytes[..]);
        Self(bytes)
    }
}

impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = f.precision().unwrap_or(PEER_ID_LENGTH);
        for byte in self.0.iter().take(len) {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PeerId({self})")
    }
}

impl<'de> serde::Deserialize<'de> for PeerId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        if deserializer.is_human_readable() {
            let s = <String>::deserialize(deserializer)?;

            hex::FromHex::from_hex(s).map_err(D::Error::custom).map(Self)
        } else {
            <[u8; PEER_ID_LENGTH]>::deserialize(deserializer).map(Self)
        }
    }
}

impl serde::Serialize for PeerId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            hex::encode(self.0).serialize(serializer)
        } else {
            self.0.serialize(serializer)
        }
    }
}

struct ShortPeerId<'a>(&'a PeerId, u8);

impl<'a> std::fmt::Display for ShortPeerId<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.len$}", self.0, len = self.1.into())
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
}
// NetworkPublicKey conversions
impl From<Ed25519PublicKey> for NetworkPublicKey {
    fn from(pk: Ed25519PublicKey) -> Self {
        Self::new(pk)
    }
}

// Conversions to PeerId
impl From<&Ed25519PublicKey> for PeerId {
    fn from(pk: &Ed25519PublicKey) -> Self {
        let bytes: [u8; PEER_ID_LENGTH] =
            pk.as_ref()[..PEER_ID_LENGTH].try_into().expect("Ed25519PublicKey should be 32 bytes");
        Self(bytes)
    }
}

impl From<Ed25519PublicKey> for PeerId {
    fn from(pk: Ed25519PublicKey) -> Self {
        Self::from(&pk)
    }
}

impl From<&NetworkPublicKey> for PeerId {
    fn from(pk: &NetworkPublicKey) -> Self {
        Self::from(pk.clone().into_inner())
    }
}

impl From<NetworkPublicKey> for PeerId {
    fn from(pk: NetworkPublicKey) -> Self {
        Self::from(pk.into_inner())
    }
}

// Conversions from PeerId
impl From<PeerId> for Ed25519PublicKey {
    fn from(peer_id: PeerId) -> Self {
        Ed25519PublicKey::from_bytes(&peer_id.0)
            .expect("PeerId bytes should always be valid Ed25519PublicKey")
    }
}

impl From<PeerId> for NetworkPublicKey {
    fn from(peer_id: PeerId) -> Self {
        Self::new(peer_id.into())
    }
}
