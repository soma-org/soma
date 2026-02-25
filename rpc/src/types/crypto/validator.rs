// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::types::{
    EpochId, StakeUnit,
    crypto::{Bls12381PublicKey, Bls12381Signature},
};

/// The Validator Set for a particular epoch.
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ValidatorCommittee {
    pub epoch: EpochId,
    pub members: Vec<ValidatorCommitteeMember>,
}

/// A member of a Validator Committee with full authority information
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ValidatorCommitteeMember {
    pub authority_key: Vec<u8>, // BLS12381 public key bytes (will become AuthorityName)
    pub stake: StakeUnit,
    pub network_metadata: ValidatorNetworkMetadata,
}

/// Network and protocol information for a validator
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ValidatorNetworkMetadata {
    pub consensus_address: String, // Multiaddr string representation
    pub hostname: String,
    pub protocol_key: Vec<u8>, // Ed25519 protocol public key bytes
    pub network_key: Vec<u8>,  // Ed25519 network public key bytes
}

/// An aggregated signature from multiple Validators.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// validator-aggregated-signature = u64               ; epoch
///                                  bls-signature
///                                  roaring-bitmap
/// roaring-bitmap = bytes  ; where the contents of the bytes are valid
///                         ; according to the serialized spec for
///                         ; roaring bitmaps
/// ```
///
/// See [here](https://github.com/RoaringBitmap/RoaringFormatSpec) for the specification for the
/// serialized format of RoaringBitmaps.
#[derive(Clone, Debug, PartialEq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ValidatorAggregatedSignature {
    pub epoch: EpochId,
    pub signature: Bls12381Signature,
    pub bitmap: crate::types::Bitmap,
}

/// A signature from a Validator
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// validator-signature = u64               ; epoch
///                       bls-public-key
///                       bls-signature
/// ```
#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Serialize, serde_derive::Deserialize)]
pub struct ValidatorSignature {
    pub epoch: EpochId,
    pub public_key: Bls12381PublicKey,
    pub signature: Bls12381Signature,
}
