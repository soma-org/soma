pub mod actors;
pub mod balance_change;
pub mod base;
pub mod checkpoints;
pub mod checksum;
pub mod client;
pub mod committee;
pub mod config;
pub mod consensus;
pub mod crypto;
pub mod digests;
pub mod discovery;
pub mod effects;
pub mod encoder_committee;
pub mod encoder_validator;
pub mod entropy;
pub mod envelope;
pub mod error;
pub mod evaluation;
pub mod execution;
pub mod finality;
pub mod full_checkpoint_content;
pub mod genesis;
pub mod genesis_builder;
pub mod intent;
pub mod messages_grpc;
pub mod metadata;
pub mod multiaddr;
pub mod mutex_table;
pub mod object;
pub mod p2p;
pub mod parameters;
pub mod peer_id;
pub mod quorum_driver;
pub mod report;
pub(crate) mod serde;
pub mod shard;
pub mod shard_crypto;
pub mod shard_networking;
pub mod shard_verifier;
pub mod signature_verification;
pub mod storage;
pub mod submission;
pub mod system_state;
pub mod temporary_store;
pub mod tls;
pub mod traffic_control;
pub mod transaction;
pub mod transaction_executor;
pub mod transaction_outputs;
pub mod tx_fee;
pub mod unit_tests;

use base::SomaAddress;
use object::{ObjectID, Version, OBJECT_START_VERSION};

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

pub const SYSTEM_STATE_OBJECT_SHARED_VERSION: Version = OBJECT_START_VERSION;
