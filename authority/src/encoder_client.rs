use fastcrypto::traits::KeyPair;
use shared::{
    crypto::keys::{PeerKeyPair, PeerPublicKey},
    error::SharedResult,
    scope::Scope,
    signed::Signed,
    verified::Verified,
};

use std::{collections::BTreeMap, sync::Arc, time::Duration};
use types::{
    committee::EncoderCommittee,
    crypto::{AuthorityKeyPair, NetworkKeyPair},
    parameters::Parameters,
    shard::{ShardAuthToken, ShardInput, ShardInputV1},
    shard_networking::{
        external::{EncoderExternalNetworkClient, EncoderExternalTonicClient},
        EncoderNetworkingInfo,
    },
};

pub struct EncoderClientService {
    client: Arc<EncoderExternalTonicClient>,
    authority_keypair: AuthorityKeyPair,
}

impl EncoderClientService {
    pub fn new(authority_keypair: AuthorityKeyPair, network_keypair: NetworkKeyPair) -> Self {
        // Convert network keypair to PeerKeyPair for TLS
        let peer_keypair = PeerKeyPair::new(network_keypair.into_inner());

        // Create empty NetworkingInfo initially
        let networking_info = EncoderNetworkingInfo::new(Vec::new());

        // Create the tonic client
        let client = Arc::new(EncoderExternalTonicClient::new(
            networking_info,
            peer_keypair,
            Arc::new(Parameters::default()),
            100, // channel pool capacity
        ));

        Self {
            client,
            authority_keypair,
        }
    }

    /// Update networking info when encoder committee changes
    pub fn update_encoder_committee(&self, committee: &EncoderCommittee) {
        let mut network_mapping = Vec::new();

        for (encoder_key, _) in &committee.members {
            if let Some(metadata) = committee.network_metadata.get(encoder_key) {
                let peer_public_key = PeerPublicKey::new(metadata.network_key.clone().into_inner());

                // TODO: Temporary conversion between Multiaddr types
                // Convert types::multiaddr::Multiaddr to soma_network::multiaddr::Multiaddr
                let network_addr_str = metadata.external_network_address.to_string();
                let network_multiaddr =
                    soma_network::multiaddr::Multiaddr::try_from(network_addr_str)
                        .expect("Failed to convert multiaddr");

                network_mapping.push((encoder_key.clone(), (peer_public_key, network_multiaddr)))
            }
        }

        // This assumes you add an update method to NetworkingInfo
        self.client.networking_info.update(network_mapping);
    }

    /// Send shard input to all members of the shard
    pub async fn send_to_shard(
        &self,
        token: ShardAuthToken,
        timeout: Duration,
    ) -> SharedResult<()> {
        // Create and sign the shard input
        let input = ShardInput::V1(ShardInputV1::new(token.clone()));
        let signed_input = Signed::new(
            input,
            Scope::Input,
            &self.authority_keypair.copy().private(),
        )?;
        let verified_input = Verified::from_trusted(signed_input)?;

        // Send to each shard member
        for encoder_key in token.shard.encoders() {
            self.client
                .send_input(&encoder_key, &verified_input, timeout)
                .await?;
        }

        Ok(())
    }
}
