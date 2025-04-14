use std::{collections::BTreeMap, sync::Arc};

use arc_swap::ArcSwap;
use shared::crypto::keys::{EncoderPublicKey, PeerPublicKey};
use soma_network::multiaddr::Multiaddr;

pub(crate) mod channel_pool;
pub(crate) mod external;
pub(crate) mod internal;

mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderInternalTonicService.rs"
    ));
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderExternalTonicService.rs"
    ));
}

#[derive(Debug, Clone, Default)]
pub struct NetworkingInfo {
    inner: Arc<ArcSwap<BTreeMap<EncoderPublicKey, (Multiaddr, PeerPublicKey)>>>,
}

impl NetworkingInfo {
    pub fn new(mapping: BTreeMap<EncoderPublicKey, (Multiaddr, PeerPublicKey)>) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(mapping)),
        }
    }

    pub fn update(&self, new_mapping: BTreeMap<EncoderPublicKey, (Multiaddr, PeerPublicKey)>) {
        self.inner.store(Arc::new(new_mapping));
    }

    fn lookup(&self, key: &EncoderPublicKey) -> Option<(Multiaddr, PeerPublicKey)> {
        match self.inner.load().get(key) {
            Some(k) => Some(k.clone()),
            None => None,
        }
    }
}
