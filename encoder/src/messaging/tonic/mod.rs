use std::{collections::BTreeMap, sync::Arc};

use arc_swap::ArcSwap;
use shared::crypto::keys::{EncoderPublicKey, PeerPublicKey};
use soma_network::multiaddr::Multiaddr;
use tracing::info;
mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderInternalTonicService.rs"
    ));
}

pub mod external;
pub mod internal;
