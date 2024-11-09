#![doc = include_str!("../README.md")]
mod core;
mod crypto;
mod error;
mod intelligence;
mod networking;
mod storage;
mod types;

pub use types::scope::{Scope, ScopedMessage};

pub use crypto::keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey};
