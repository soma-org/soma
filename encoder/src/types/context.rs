use std::{collections::HashMap, sync::Arc};

use arc_swap::ArcSwap;
use shared::{
    authority_committee::AuthorityCommittee,
    crypto::keys::{EncoderPublicKey, PeerPublicKey},
};
use soma_network::multiaddr::Multiaddr;

use crate::error::{ShardError, ShardResult};

use super::encoder_committee::{Encoder, EncoderCommittee, EncoderIndex, Epoch};

#[derive(Clone)]
pub(crate) struct Context {
    inner: Arc<ArcSwap<InnerContext>>,
}

impl Context {
    pub fn new(inner: InnerContext) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(inner)),
        }
    }

    pub fn update(&self, inner: InnerContext) {
        self.inner.store(Arc::new(inner));
    }

    pub fn own_encoder_index(&self, epoch: Epoch) -> ShardResult<EncoderIndex> {
        Ok(self.inner.load().committees(epoch)?.own_encoder_index)
    }

    pub fn encoder(&self, epoch: Epoch, encoder: EncoderIndex) -> ShardResult<Encoder> {
        Ok(self
            .inner
            .load()
            .committees(epoch)?
            .encoder_committee
            .encoder(encoder)
            .clone())
    }
    pub fn inner(&self) -> Arc<InnerContext> {
        self.inner.load_full()
    }
    // TODO add the top level handlers here
}

/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
#[derive(Clone)]
pub(crate) struct InnerContext {
    committees: [Committees; 2],
    current_epoch: Epoch,
    own_encoder_public_key: EncoderPublicKey,
    encoder_object_servers: HashMap<EncoderPublicKey, (PeerPublicKey, Multiaddr)>,
}

impl InnerContext {
    fn current_committees(&self) -> &Committees {
        &self.committees[1]
    }
    fn previous_committees(&self) -> &Committees {
        &self.committees[0]
    }
    pub(crate) fn committees(&self, epoch: Epoch) -> ShardResult<&Committees> {
        if epoch == self.current_epoch {
            return Ok(self.current_committees());
        } else if epoch == self.current_epoch.saturating_sub(1) {
            return Ok(self.previous_committees());
        }
        Err(ShardError::WrongEpoch)
    }
    pub(crate) fn own_encoder_key(&self) -> &EncoderPublicKey {
        &self.own_encoder_public_key
    }
    pub(crate) fn object_server(
        &self,
        encoder: &EncoderPublicKey,
    ) -> Option<(PeerPublicKey, Multiaddr)> {
        self.encoder_object_servers
            .get(encoder)
            .map(|(peer_key, address)| (peer_key.clone(), address.clone()))
    }
}

#[derive(Clone)]
pub(crate) struct Committees {
    pub epoch: Epoch,
    pub authority_committee: AuthorityCommittee,
    /// the committee of allowed network keys
    pub encoder_committee: EncoderCommittee,
    /// The services own index for each respective modality
    pub own_encoder_index: EncoderIndex,
    // TODO: move this to a more robust protocol config
    pub vdf_iterations: u64,
}

impl Committees {
    pub(crate) fn new(
        epoch: Epoch,
        authority_committee: AuthorityCommittee,
        encoder_committee: EncoderCommittee,
        own_encoder_index: EncoderIndex,
        vdf_iterations: u64,
    ) -> Self {
        Self {
            epoch,
            authority_committee,
            encoder_committee,
            own_encoder_index,
            vdf_iterations,
        }
    }
}
