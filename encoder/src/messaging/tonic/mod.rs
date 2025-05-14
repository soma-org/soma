use std::{collections::BTreeMap, sync::Arc};

use arc_swap::ArcSwap;
use shared::crypto::keys::{EncoderPublicKey, PeerPublicKey};
use soma_network::multiaddr::Multiaddr;
use tracing::info;

pub(crate) mod channel_pool;
pub mod external;
pub mod internal;

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

    pub fn add(
        &mut self,
        key: EncoderPublicKey,
        value: (soma_network::multiaddr::Multiaddr, PeerPublicKey),
    ) {
        // Get the current map
        let current = self.inner.load();

        // Clone the BTreeMap to create a mutable copy
        let mut new_map = current.as_ref().clone();

        info!("Adding to new map: {:?}", new_map);

        // Insert the new entry
        new_map.insert(key, value);

        // Store the updated map back into the ArcSwap
        self.inner.store(Arc::new(new_map));
    }

    fn lookup(&self, key: &EncoderPublicKey) -> Option<(Multiaddr, PeerPublicKey)> {
        match self.inner.load().get(key) {
            Some(k) => Some(k.clone()),
            None => None,
        }
    }
}
