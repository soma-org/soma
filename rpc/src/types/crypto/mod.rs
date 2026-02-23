// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

mod bls12381;
mod ed25519;
mod intent;
mod multisig;
mod signature;
mod validator;

pub use bls12381::Bls12381PublicKey;
pub use bls12381::Bls12381Signature;
pub use ed25519::Ed25519PublicKey;
pub use ed25519::Ed25519Signature;
pub use intent::Intent;
pub use intent::IntentAppId;
pub use intent::IntentScope;
pub use intent::IntentVersion;
pub use multisig::MultisigAggregatedSignature;
pub use multisig::MultisigCommittee;
pub use multisig::MultisigMember;
pub use multisig::MultisigMemberPublicKey;
pub use multisig::MultisigMemberSignature;
pub use signature::SignatureScheme;
pub use signature::SimpleSignature;
pub use signature::UserSignature;
pub use validator::ValidatorAggregatedSignature;
pub use validator::ValidatorCommittee;
pub use validator::ValidatorCommitteeMember;
pub use validator::ValidatorNetworkMetadata;
pub use validator::ValidatorSignature;

//
// Implement various base64 fixed-size array helpers
//

/// Utility for calculating base64 encoding lengths.
///
/// In the Base64 encoding each character is used to represent 6 bits (log2(64) = 6). This means
/// that 4 characters are used to represent 4*6 = 24 bits = 3 bytes. So you need 4*(`n`/3)
/// characters in order to represent `n` bytes, and this needs to be rounded up to a multiple of 4.
/// The number of unused padding characters resulting from the rounding will be 0, 1, 2, or 3.
const fn base64_encoded_length(len: usize) -> usize {
    ((4 * len / 3) + 3) & !3
}

macro_rules! impl_base64_helper {
    ($base:ident, $display:ident, $fromstr:ident, $test_module:ident, $array_length:literal) => {
        #[allow(unused)]
        struct $base;

        impl $base {
            const LENGTH: usize = $array_length;
            #[allow(unused)]
            const ENCODED_LENGTH: usize = base64_encoded_length(Self::LENGTH);
        }

        #[allow(unused)]
        struct $display<'a>(&'a [u8; $base::LENGTH]);

        impl<'a> std::fmt::Display for $display<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut buf = [0; $base::ENCODED_LENGTH];
                let encoded =
                    <base64ct::Base64 as base64ct::Encoding>::encode(self.0, &mut buf).unwrap();
                f.write_str(encoded)
            }
        }

        #[allow(unused)]
        #[derive(Debug, PartialEq)]
        struct $fromstr([u8; $base::LENGTH]);

        impl std::str::FromStr for $fromstr {
            type Err = base64ct::Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut buf = [0; $base::LENGTH];
                let decoded = <base64ct::Base64 as base64ct::Encoding>::decode(s, &mut buf)?;
                assert_eq!(decoded.len(), $base::LENGTH);
                Ok(Self(buf))
            }
        }

        impl serde_with::SerializeAs<[u8; Self::LENGTH]> for $base {
            fn serialize_as<S>(
                source: &[u8; Self::LENGTH],
                serializer: S,
            ) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let display = $display(source);
                serde_with::DisplayFromStr::serialize_as(&display, serializer)
            }
        }

        impl<'de> serde_with::DeserializeAs<'de, [u8; Self::LENGTH]> for $base {
            fn deserialize_as<D>(deserializer: D) -> Result<[u8; Self::LENGTH], D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let array: $fromstr = serde_with::DisplayFromStr::deserialize_as(deserializer)?;
                Ok(array.0)
            }
        }
    };
}

impl_base64_helper!(Base64Array32, Base64Display32, Base64FromStr32, test32, 32);
impl_base64_helper!(Base64Array33, Base64Display33, Base64FromStr33, test33, 33);
impl_base64_helper!(Base64Array34, Base64Display34, Base64FromStr34, test34, 34);
impl_base64_helper!(Base64Array48, Base64Display48, Base64FromStr48, test48, 48);
impl_base64_helper!(Base64Array64, Base64Display64, Base64FromStr64, test64, 64);
impl_base64_helper!(Base64Array96, Base64Display96, Base64FromStr96, test96, 96);
