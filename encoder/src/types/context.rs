use std::sync::Arc;

use arc_swap::ArcSwap;
use protocol_config::{Chain, ProtocolConfig, ProtocolVersion};
use types::crypto::{NetworkKeyPair, NetworkPublicKey};
use types::metadata::{DownloadMetadata, Metadata};
use types::multiaddr::Multiaddr;
use types::{checksum::Checksum, committee::Committee, shard_crypto::keys::EncoderPublicKey};

use types::error::{ShardError, ShardResult};

use types::committee::{Epoch, NetworkingCommittee};
use types::encoder_committee::EncoderCommittee;

#[derive(Clone, Debug)]
pub struct Context {
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

    pub fn own_encoder_key(&self) -> EncoderPublicKey {
        self.inner.load().own_encoder_public_key.clone()
    }
    pub fn own_network_keypair(&self) -> NetworkKeyPair {
        self.inner.load().own_network_keypair.clone()
    }

    pub fn inner(&self) -> Arc<InnerContext> {
        self.inner.load_full()
    }

    pub fn probe(&self, epoch: Epoch, encoder: &EncoderPublicKey) -> ShardResult<DownloadMetadata> {
        match self
            .inner
            .load()
            .committees(epoch)?
            .encoder_committee
            .encoder_by_key(&encoder)
        {
            Some(e) => Ok(e.probe.clone()),
            None => Err(ShardError::NotFound(
                "probe metadata not found for encoder".to_string(),
            )),
        }
    }

    pub fn network_public_keys(
        &self,
        encoders: Vec<EncoderPublicKey>,
    ) -> ShardResult<Vec<NetworkPublicKey>> {
        let mut network_keys: Vec<NetworkPublicKey> = Vec::new();
        for encoder_key in encoders {
            if let Some(nm) = self
                .inner
                .load()
                .current_committees()
                .encoder_committee
                .network_metadata
                .get(&encoder_key)
            {
                network_keys.push(nm.network_key.clone());
            } else {
                return Err(ShardError::NotFound(
                    "encoder network key not found".to_string(),
                ));
            }
        }

        Ok(network_keys)
    }
}

/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
#[derive(Clone, Debug)]
pub struct InnerContext {
    committees: [Committees; 2],
    pub current_epoch: Epoch,
    own_encoder_public_key: EncoderPublicKey,
    own_network_keypair: NetworkKeyPair,
    chain: Chain,
}

impl InnerContext {
    pub fn new(
        committees: [Committees; 2],
        current_epoch: Epoch,
        own_encoder_public_key: EncoderPublicKey,
        own_network_keypair: NetworkKeyPair,
        chain: Chain,
    ) -> Self {
        Self {
            current_epoch,
            own_encoder_public_key,
            committees,
            own_network_keypair,
            chain,
        }
    }

    pub fn current_committees(&self) -> &Committees {
        &self.committees[1]
    }

    fn previous_committees(&self) -> &Committees {
        &self.committees[0]
    }

    pub fn committees(&self, epoch: Epoch) -> ShardResult<&Committees> {
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

    pub fn protocol_config(&self) -> ProtocolConfig {
        self.current_committees().protocol_config(self.chain)
    }

    pub fn vdf_iterations(&self) -> u64 {
        self.protocol_config().vdf_iterations()
    }

    pub fn chain(&self) -> Chain {
        self.chain
    }
}

#[derive(Clone, Debug)]
pub struct Committees {
    pub epoch: Epoch,
    pub authority_committee: Committee,
    /// the committee of allowed network keys
    pub encoder_committee: EncoderCommittee,
    /// the committee of all validators with minimum stake for networking
    pub networking_committee: NetworkingCommittee,
    pub protocol_version: ProtocolVersion,
}

impl Committees {
    pub fn new(
        epoch: Epoch,
        authority_committee: Committee,
        encoder_committee: EncoderCommittee,
        networking_committee: NetworkingCommittee,
        protocol_version: ProtocolVersion,
    ) -> Self {
        Self {
            epoch,
            authority_committee,
            encoder_committee,
            networking_committee,
            protocol_version,
        }
    }

    /// Get the protocol config for this committee's epoch
    pub fn protocol_config(&self, chain: Chain) -> ProtocolConfig {
        ProtocolConfig::get_for_version(self.protocol_version, chain)
    }

    /// Convenience method for vdf_iterations
    pub fn vdf_iterations(&self, chain: Chain) -> u64 {
        self.protocol_config(chain).vdf_iterations()
    }
}
