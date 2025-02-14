pub mod accumulator;
pub mod base;
pub mod client;
pub mod committee;
pub mod config;
pub mod consensus;
pub mod crypto;
pub mod digests;
pub mod discovery;
pub mod effects;
pub mod envelope;
pub mod error;
pub mod execution_indices;
pub mod genesis;
pub mod grpc;
pub mod intent;
pub mod multiaddr;
pub mod mutex_table;
pub mod object;
pub mod p2p;
pub mod parameters;
pub mod peer_id;
pub mod protocol;
pub mod quorum_driver;
pub(crate) mod serde;
pub mod signature_verifier;
pub mod state_sync;
pub mod storage;
pub mod system_state;
pub mod temporary_store;
pub mod tls;
pub mod transaction;
pub mod tx_outputs;

use base::SomaAddress;
use object::ObjectID;

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
