use std::{fmt::Display, str::FromStr};

use serde::Serialize;
use types::base::SomaAddress;

/// An address or an alias associated with a key in the wallet
/// This is used to distinguish between an address or an alias,
/// enabling a user to use an alias for any command that requires an address.
#[derive(Debug, Serialize, Clone)]
pub enum KeyIdentity {
    Address(SomaAddress),
    Alias(String),
}

impl FromStr for KeyIdentity {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("0x") {
            Ok(KeyIdentity::Address(SomaAddress::from_str(s)?))
        } else {
            Ok(KeyIdentity::Alias(s.to_string()))
        }
    }
}

impl From<SomaAddress> for KeyIdentity {
    fn from(addr: SomaAddress) -> Self {
        KeyIdentity::Address(addr)
    }
}

impl Display for KeyIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            KeyIdentity::Address(x) => x.to_string(),
            KeyIdentity::Alias(x) => x.to_string(),
        };
        write!(f, "{}", v)
    }
}
