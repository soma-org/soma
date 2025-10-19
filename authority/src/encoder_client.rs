use std::{sync::Arc, time::Duration};
use types::{
    crypto::{AuthorityKeyPair, NetworkKeyPair, NetworkPublicKey},
    encoder_committee::EncoderCommittee,
    error::SharedResult,
    metadata::DownloadableMetadata,
    parameters::TonicParameters,
    shard::{Input, InputV1, ShardAuthToken},
    shard_crypto::keys::EncoderPublicKey,
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
        // Convert network keypair to NetworkKeyPair for TLS
        let peer_keypair = NetworkKeyPair::new(network_keypair.into_inner());

        // Create empty NetworkingInfo initially
        let networking_info = EncoderNetworkingInfo::new(Vec::new());

        // Create the tonic client
        let client = Arc::new(EncoderExternalTonicClient::new(
            networking_info,
            peer_keypair,
            Arc::new(TonicParameters::default()),
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

        for (encoder_key, _) in &committee.members() {
            if let Some(metadata) = committee.network_metadata.get(encoder_key) {
                let peer_public_key =
                    NetworkPublicKey::new(metadata.network_key.clone().into_inner());

                network_mapping.push((
                    encoder_key.clone(),
                    (peer_public_key, metadata.external_network_address.clone()),
                ))
            }
        }

        // This assumes you add an update method to NetworkingInfo
        self.client.networking_info.update(network_mapping);
    }

    /// Send shard input to all members of the shard
    pub async fn send_to_shard(
        &self,
        encoders: Vec<EncoderPublicKey>,
        token: ShardAuthToken,
        downloadable_metadata: DownloadableMetadata,
        timeout: Duration,
    ) -> SharedResult<()> {
        // Create and sign the shard input
        let input = Input::V1(InputV1::new(token.clone(), downloadable_metadata));
        // Send to each shard member
        for encoder_key in encoders {
            self.client
                .send_input(&encoder_key, &input, timeout)
                .await?;
        }
        Ok(())
    }
}
