use std::collections::HashMap;
use types::{
    crypto::NetworkPublicKey, encoder_committee::EncoderCommittee, multiaddr::Multiaddr,
    shard_crypto::keys::EncoderPublicKey,
};

/// Network peer information extracted from encoder committee
#[derive(Clone, Debug)]
pub struct NetworkPeerInfo {
    pub encoder_key: EncoderPublicKey,
    pub network_key: NetworkPublicKey,
    pub internal_address: Multiaddr,
    pub object_server_address: Multiaddr,
}

/// Extract network peer info from encoder committee(s).
/// This is a standalone function so it can be used both during genesis initialization
/// and when processing verified committees.
pub fn extract_network_info_from_committees(
    current_committee: &EncoderCommittee,
    previous_committee: Option<&EncoderCommittee>,
) -> (
    Vec<(EncoderPublicKey, (NetworkPublicKey, Multiaddr))>,
    HashMap<EncoderPublicKey, (NetworkPublicKey, Multiaddr)>,
) {
    let mut networking_info = Vec::new();
    let mut object_servers = HashMap::new();

    let process_committee =
        |committee: &EncoderCommittee,
         networking_info: &mut Vec<(EncoderPublicKey, (NetworkPublicKey, Multiaddr))>,
         objects: &mut HashMap<EncoderPublicKey, (NetworkPublicKey, Multiaddr)>| {
            for (encoder_key, _stake) in committee.members() {
                if let Some(metadata) = committee.network_metadata.get(&encoder_key) {
                    let peer_key = NetworkPublicKey::new(metadata.network_key.clone().into_inner());

                    networking_info.push((
                        encoder_key.clone(),
                        (peer_key.clone(), metadata.internal_network_address.clone()),
                    ));

                    objects.insert(
                        encoder_key.clone(),
                        (peer_key, metadata.object_server_address.clone()),
                    );
                }
            }
        };

    // Process previous committee first (so current can override)
    if let Some(prev) = previous_committee {
        process_committee(prev, &mut networking_info, &mut object_servers);
    }

    // Process current committee
    process_committee(current_committee, &mut networking_info, &mut object_servers);

    (networking_info, object_servers)
}

/// Convert to the simpler NetworkPeerInfo format
pub fn extract_network_peers(committee: &EncoderCommittee) -> Vec<NetworkPeerInfo> {
    let mut peers = Vec::new();

    for (encoder_key, _stake) in committee.members() {
        if let Some(metadata) = committee.network_metadata.get(&encoder_key) {
            peers.push(NetworkPeerInfo {
                encoder_key: encoder_key.clone(),
                network_key: NetworkPublicKey::new(metadata.network_key.clone().into_inner()),
                internal_address: metadata.internal_network_address.clone(),
                object_server_address: metadata.object_server_address.clone(),
            });
        }
    }

    peers
}
