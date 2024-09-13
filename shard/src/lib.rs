#![doc = include_str!("../README.md")]
mod crypto;
mod error;
mod types;

pub use types::scope::{Scope, ScopedMessage};

pub use crypto::keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey};
