use arc_swap::ArcSwap;
use soma_network::multiaddr::Multiaddr;
use std::{collections::BTreeMap, sync::Arc};

use shared::crypto::keys::{EncoderPublicKey, PeerPublicKey};

pub mod channel_pool;
pub mod external;

pub mod generated {
    include!(concat!(
        env!("OUT_DIR"),
        "/soma.EncoderExternalTonicService.rs"
    ));
}

#[derive(Debug, Clone, Default)]
pub struct EncoderNetworkingInfoInner {
    encoder_to_tls: BTreeMap<EncoderPublicKey, (PeerPublicKey, Multiaddr)>,
    tls_to_encoder: BTreeMap<PeerPublicKey, EncoderPublicKey>,
}
#[derive(Debug, Clone, Default)]
pub struct EncoderNetworkingInfo {
    inner: Arc<ArcSwap<EncoderNetworkingInfoInner>>,
}

impl EncoderNetworkingInfo {
    pub fn new(mapping: Vec<(EncoderPublicKey, (PeerPublicKey, Multiaddr))>) -> Self {
        let mut encoder_to_tls = BTreeMap::new();
        let mut tls_to_encoder = BTreeMap::new();
        for map in mapping {
            let (encoder_public_key, (peer_public_key, multiaddr)) = map;
            encoder_to_tls.insert(
                encoder_public_key.clone(),
                (peer_public_key.clone(), multiaddr),
            );
            tls_to_encoder.insert(peer_public_key, encoder_public_key);
        }
        let inner = EncoderNetworkingInfoInner {
            encoder_to_tls,
            tls_to_encoder,
        };
        Self {
            inner: Arc::new(ArcSwap::from_pointee(inner)),
        }
    }

    pub fn update(&self, new_mapping: Vec<(EncoderPublicKey, (PeerPublicKey, Multiaddr))>) {
        let mut encoder_to_tls = BTreeMap::new();
        let mut tls_to_encoder = BTreeMap::new();
        for map in new_mapping {
            let (encoder_public_key, (peer_public_key, address)) = map;
            encoder_to_tls.insert(
                encoder_public_key.clone(),
                (peer_public_key.clone(), address),
            );
            tls_to_encoder.insert(peer_public_key, encoder_public_key);
        }
        let inner = EncoderNetworkingInfoInner {
            encoder_to_tls,
            tls_to_encoder,
        };
        self.inner.store(Arc::new(inner));
    }

    pub fn add(
        &mut self,
        encoder_public_key: EncoderPublicKey,
        peer_public_key: PeerPublicKey,
        address: Multiaddr,
    ) {
        // Get the current map
        let current = self.inner.load();

        let mut inner = current.as_ref().clone();

        inner.encoder_to_tls.insert(
            encoder_public_key.clone(),
            (peer_public_key.clone(), address),
        );
        inner
            .tls_to_encoder
            .insert(peer_public_key, encoder_public_key);

        self.inner.store(Arc::new(inner));
    }

    pub fn encoder_to_tls(&self, key: &EncoderPublicKey) -> Option<(PeerPublicKey, Multiaddr)> {
        match self.inner.load().encoder_to_tls.get(key) {
            Some(k) => Some(k.clone()),
            None => None,
        }
    }
    pub fn tls_to_encoder(&self, peer: &PeerPublicKey) -> Option<EncoderPublicKey> {
        match self.inner.load().tls_to_encoder.get(peer) {
            Some(k) => Some(k.clone()),
            None => None,
        }
    }
}
