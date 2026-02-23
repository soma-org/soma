// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Intent {
    pub scope: IntentScope,
    pub version: IntentVersion,
    pub app_id: IntentAppId,
}

impl Intent {
    pub fn new(scope: IntentScope, version: IntentVersion, app_id: IntentAppId) -> Self {
        Self { scope, version, app_id }
    }

    pub fn to_bytes(self) -> [u8; 3] {
        [self.scope as u8, self.version as u8, self.app_id as u8]
    }

    pub fn scope(self) -> IntentScope {
        self.scope
    }

    pub fn version(self) -> IntentVersion {
        self.version
    }

    pub fn app_id(self) -> IntentAppId {
        self.app_id
    }
}

/// Byte signifying the scope of an [`Intent`]
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// intent-scope = u8
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum IntentScope {
    TransactionData = 0,         // Used for a user signature on a transaction data.
    TransactionEffects = 1,      // Used for an authority signature on transaction effects.
    CheckpointSummary = 2,       // Used for an authority signature on a checkpoint summary.
    PersonalMessage = 3,         // Used for a user signature on a personal message.
    SenderSignedTransaction = 4, // Used for an authority signature on a user signed transaction.
    ConsensusBlock = 5,          // Used for consensus authority signature on block's digest
}

/// Byte signifying the version of an [`Intent`]
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// intent-version = u8
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum IntentVersion {
    V0 = 0,
}

/// Byte signifying the application id of an [`Intent`]
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// intent-app-id = u8
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum IntentAppId {
    Consensus = 0,
    Soma = 1,
}
