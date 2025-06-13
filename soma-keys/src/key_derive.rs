// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use bip32::{ChildNumber, DerivationPath};

use bip39::{Language, Mnemonic, MnemonicType, Seed};
use fastcrypto::ed25519::Ed25519KeyPair;
use fastcrypto::{
    ed25519::Ed25519PrivateKey,
    traits::{KeyPair, ToFromBytes},
};
use slip10_ed25519::derive_ed25519_private_key;
use types::base::SomaAddress;
use types::crypto::{SignatureScheme, SomaKeyPair};

use crate::error::{SomaKeyError, SomaKeyResult};

pub const DERIVATION_PATH_COIN_TYPE: u32 = 791;
pub const DERVIATION_PATH_PURPOSE_ED25519: u32 = 44;

/// Ed25519 follows SLIP-0010 using hardened path: m/44'/784'/0'/0'/{index}'
/// Note that the purpose node is used to distinguish signature schemes.
pub fn derive_key_pair_from_path(
    seed: &[u8],
    derivation_path: Option<DerivationPath>,
    key_scheme: &SignatureScheme,
) -> SomaKeyResult<(SomaAddress, SomaKeyPair)> {
    let path = validate_path(key_scheme, derivation_path)?;
    match key_scheme {
        SignatureScheme::ED25519 => {
            let indexes = path.into_iter().map(|i| i.into()).collect::<Vec<_>>();
            let derived = derive_ed25519_private_key(seed, &indexes);
            let sk = Ed25519PrivateKey::from_bytes(&derived)
                .map_err(|e| SomaKeyError::FailedToGenerateKeyPair(e.to_string()))?;
            let kp: Ed25519KeyPair = sk.into();
            Ok((kp.public().into(), SomaKeyPair::Ed25519(kp)))
        }
        _ => Err(SomaKeyError::UnsupportedKeyScheme),
    }
}

pub fn validate_path(
    key_scheme: &SignatureScheme,
    path: Option<DerivationPath>,
) -> SomaKeyResult<DerivationPath> {
    match key_scheme {
        SignatureScheme::ED25519 => {
            match path {
                Some(p) => {
                    // The derivation path must be hardened at all levels with purpose = 44, coin_type = 784
                    if let &[purpose, coin_type, account, change, address] = p.as_ref() {
                        if Some(purpose)
                            == ChildNumber::new(DERVIATION_PATH_PURPOSE_ED25519, true).ok()
                            && Some(coin_type)
                                == ChildNumber::new(DERIVATION_PATH_COIN_TYPE, true).ok()
                            && account.is_hardened()
                            && change.is_hardened()
                            && address.is_hardened()
                        {
                            Ok(p)
                        } else {
                            Err(SomaKeyError::InvalidDerivationPath)
                        }
                    } else {
                        Err(SomaKeyError::InvalidDerivationPath)
                    }
                }
                None => Ok(format!(
                    "m/{DERVIATION_PATH_PURPOSE_ED25519}'/{DERIVATION_PATH_COIN_TYPE}'/0'/0'/0'"
                )
                .parse()
                .map_err(|_| SomaKeyError::InvalidDerivationPath)?),
            }
        }
        _ => Err(SomaKeyError::UnsupportedKeyScheme),
    }
}

pub fn generate_new_key(
    key_scheme: SignatureScheme,
    derivation_path: Option<DerivationPath>,
    word_length: Option<String>,
) -> SomaKeyResult<(SomaAddress, SomaKeyPair, SignatureScheme, String)> {
    let mnemonic = Mnemonic::new(parse_word_length(word_length)?, Language::English);
    let seed = Seed::new(&mnemonic, "");
    match derive_key_pair_from_path(seed.as_bytes(), derivation_path, &key_scheme) {
        Ok((address, kp)) => Ok((address, kp, key_scheme, mnemonic.phrase().to_string())),
        Err(e) => Err(SomaKeyError::FailedToGenerateKeyPair(e.to_string())),
    }
}

fn parse_word_length(s: Option<String>) -> SomaKeyResult<MnemonicType> {
    match s {
        None => Ok(MnemonicType::Words12),
        Some(s) => match s.as_str() {
            "word12" => Ok(MnemonicType::Words12),
            "word15" => Ok(MnemonicType::Words15),
            "word18" => Ok(MnemonicType::Words18),
            "word21" => Ok(MnemonicType::Words21),
            "word24" => Ok(MnemonicType::Words24),
            _ => Err(SomaKeyError::InvalidWordLength),
        },
    }
}
