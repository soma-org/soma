// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt::Display, str::FromStr};

use serde::Serialize;
use soma_keys::keystore::{AccountKeystore, Keystore};
use types::base::SomaAddress;

use crate::error::{CliError, CliResult};

/// An address or an alias associated with a key in the wallet
/// This is used to distinguish between an address or an alias,
/// enabling a user to use an alias for any command that requires an address.
#[derive(Serialize, Clone)]
pub enum KeyIdentity {
    Address(SomaAddress),
    Alias(String),
}

impl FromStr for KeyIdentity {
    type Err = CliError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("0x") {
            Ok(KeyIdentity::Address(
                SomaAddress::from_str(s).map_err(|e| CliError::AddressError(e.to_string()))?,
            ))
        } else {
            Ok(KeyIdentity::Alias(s.to_string()))
        }
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

/// Get the SomaAddress corresponding to this key identity.
/// If no string is provided, then the current active address is returned.
pub fn get_identity_address(
    input: Option<KeyIdentity>,
    keystore: &Keystore,
) -> CliResult<SomaAddress> {
    if let Some(addr) = input {
        get_identity_address_from_keystore(addr, keystore)
    } else {
        Err(CliError::AddressError("could not load address".to_string()))
    }
}

pub fn get_identity_address_from_keystore(
    input: KeyIdentity,
    keystore: &Keystore,
) -> CliResult<SomaAddress> {
    match input {
        KeyIdentity::Address(x) => Ok(x),
        KeyIdentity::Alias(x) => Ok(*keystore
            .get_address_by_alias(x)
            .map_err(|e| CliError::AliasError(e.to_string()))?),
    }
}
