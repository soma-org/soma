use std::{collections::HashMap, sync::Arc};

use arc_swap::ArcSwap;
use shared::{
    authority_committee::AuthorityCommittee,
    crypto::keys::{EncoderPublicKey, PeerPublicKey},
    multiaddr::Multiaddr,
};

use crate::error::{ShardError, ShardResult};

use super::{
    encoder_committee::{Encoder, EncoderCommittee, EncoderIndex, Epoch},
    parameters::Parameters,
};

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
    pub fn encoder_networking_details(
        &self,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<NetworkingDetails> {
        match self.inner.load().encoder_networking.get(encoder) {
            Some(networking_details) => Ok(networking_details.clone()),
            None => Err(ShardError::NotFound(
                "failed to get networking details".to_string(),
            )),
        }
    }

    // pub fn encoder_networking(
    //     &self,
    //     epoch: Epoch,
    //     encoder: EncoderIndex,
    // ) -> ShardResult<NetworkingDetails> {
    //     self.inner.load().encoder_networking_details(epoch, encoder)
    // }

    pub fn own_encoder_index(&self, epoch: Epoch) -> ShardResult<EncoderIndex> {
        Ok(self.inner.load().committees(epoch)?.own_encoder_index)
    }

    pub fn parameters(&self) -> Parameters {
        self.inner.load().parameters.clone()
    }

    pub fn own_networking_details(&self) -> ShardResult<NetworkingDetails> {
        let inner = self.inner.load();
        match inner.encoder_networking.get(&inner.own_encoder_public_key) {
            Some(networking_details) => Ok(networking_details.clone()),
            None => Err(ShardError::NotFound(
                "failed to get networking details".to_string(),
            )),
        }
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

#[derive(Clone)]
pub(crate) struct NetworkingDetails {
    pub tls_key: PeerPublicKey,
    pub address: Multiaddr,
}

/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
#[derive(Clone)]
pub(crate) struct InnerContext {
    committees: [Committees; 2],
    encoder_networking: HashMap<EncoderPublicKey, NetworkingDetails>,
    current_epoch: Epoch,
    parameters: Parameters,
    own_encoder_public_key: EncoderPublicKey,
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
    fn encoder_networking_details(
        &self,
        epoch: Epoch,
        peer: EncoderIndex,
    ) -> ShardResult<NetworkingDetails> {
        let committees = self.committees(epoch)?;
        let encoder_pub_key = &committees.encoder_committee.encoder(peer).encoder_key;
        match self.encoder_networking.get(encoder_pub_key) {
            Some(networking_details) => Ok(networking_details.clone()),
            None => Err(ShardError::NotFound(
                "failed to get networking details".to_string(),
            )),
        }
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
    pub own_encoder_public_key: EncoderPublicKey,
}

impl Committees {
    pub(crate) fn new(
        epoch: Epoch,
        authority_committee: AuthorityCommittee,
        encoder_committee: EncoderCommittee,
        own_encoder_index: EncoderIndex,
        own_encoder_public_key: EncoderPublicKey,
    ) -> Self {
        Self {
            epoch,
            authority_committee,
            encoder_committee,
            own_encoder_index,
            own_encoder_public_key,
        }
    }
}
