// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

pub mod balance_change;
pub mod base;
pub mod challenge;
pub mod checkpoints;
pub mod checksum;
#[cfg(feature = "tls")]
pub mod client;
pub mod committee;
pub mod config;
pub mod consensus;
pub mod crypto;
pub mod digests;
pub mod effects;
pub mod envelope;
pub mod error;
pub mod execution;
pub mod finality;
pub mod full_checkpoint_content;
pub mod genesis;
#[cfg(feature = "ml")]
pub mod genesis_builder;
pub mod grpc_timeout;
pub mod intent;
pub mod messages_grpc;
pub mod metadata;
pub mod model;
#[cfg(feature = "ml")]
pub mod model_selection;
pub mod multiaddr;
pub mod multisig;
pub mod mutex_table;
pub mod object;
pub mod parameters;
pub mod peer_id;
pub mod quorum_driver;
pub(crate) mod serde;
pub mod signature_verification;
pub mod storage;
pub mod submission;
pub mod supported_protocol_versions;
pub mod sync;
pub mod system_state;
pub mod target;
pub mod temporary_store;
pub mod tensor;
#[cfg(feature = "tls")]
pub mod tls;
pub mod traffic_control;
pub mod transaction;
pub mod transaction_executor;
pub mod transaction_outputs;
pub mod tx_fee;
pub mod unit_tests;
pub mod validator_info;

use base::SomaAddress;
use object::{OBJECT_START_VERSION, ObjectID, Version};

macro_rules! built_in_ids {
    ($($addr:ident / $id:ident = $init:expr);* $(;)?) => {
        $(
            pub const $addr: SomaAddress = builtin_address($init);
            pub const $id: ObjectID = ObjectID::from_address($addr);
        )*
    }
}

const fn builtin_address(suffix: u16) -> SomaAddress {
    let mut addr = [0u8; SomaAddress::LENGTH];
    let [hi, lo] = suffix.to_be_bytes();
    addr[SomaAddress::LENGTH - 2] = hi;
    addr[SomaAddress::LENGTH - 1] = lo;
    SomaAddress::new(addr)
}

built_in_ids! {
    SYSTEM_STATE_ADDRESS / SYSTEM_STATE_OBJECT_ID = 0x5;
}

/// The initial shared version for the SystemState object created at genesis.
/// After genesis execution, the lamport timestamp is Version(1) = OBJECT_START_VERSION.
pub const SYSTEM_STATE_OBJECT_SHARED_VERSION: Version = OBJECT_START_VERSION;

/// The initial shared version for Target objects created at genesis.
/// All targets are created at genesis with the same lamport timestamp.
pub const TARGET_OBJECT_SHARED_VERSION: Version = OBJECT_START_VERSION;

/// The initial shared version for Challenge objects.
/// Challenges are created dynamically (not at genesis) but need an initial version for shared object handling.
pub const CHALLENGE_OBJECT_SHARED_VERSION: Version = OBJECT_START_VERSION;
