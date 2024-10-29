use crate::types::network_committee::NetworkingIndex;

#[derive(Clone, Debug)]
pub(crate) struct PeerInfo {
    pub(crate) network_index: NetworkingIndex,
}
