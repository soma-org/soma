use std::{collections::BTreeMap, sync::Arc};

use arc_swap::ArcSwap;
use tracing::info;
use types::multiaddr::Multiaddr;
use types::shard_crypto::keys::{EncoderPublicKey, PeerPublicKey};
mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderInternalTonicService.rs"
    ));
}

pub mod external;
pub mod internal;
