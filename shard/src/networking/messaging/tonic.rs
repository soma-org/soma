use crate::types::network_committee::NetworkIdentityIndex;

#[derive(Clone, Debug)]
pub(crate) struct PeerInfo {
    pub(crate) network_index: NetworkIdentityIndex,
}
