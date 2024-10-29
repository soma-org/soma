use crate::types::authority_committee::AuthorityCommittee;

use super::{
    modality::{ModalityCommittee, OwnModalityIndices},
    network_committee::{NetworkCommittee, NetworkingIndex},
};

pub(crate) trait NetworkingContext: Send + Sync + 'static {
    fn network_committee(&self) -> &NetworkCommittee;
    fn own_network_index(&self) -> NetworkingIndex;
}

#[derive(Clone)]
/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
pub(crate) struct EncoderContext {
    /// the committee of validators
    pub(crate) authority_committee: AuthorityCommittee,
    /// the committee of allowed network keys
    pub(crate) network_committee: NetworkCommittee,
    /// the services own index for the network committee
    pub(crate) own_network_index: NetworkingIndex,
    /// The encoder committee for each registered modality
    pub(crate) modality_committees: ModalityCommittee,
    /// The services own index for each respective modality
    pub(crate) own_modality_indices: OwnModalityIndices,
    // TODO: add configs
    // TODO: add metrics back
}

impl EncoderContext {
    pub(crate) fn new(
        authority_committee: AuthorityCommittee,
        network_committee: NetworkCommittee,
        own_network_index: NetworkingIndex,
        modality_committees: ModalityCommittee,
        own_modality_indices: OwnModalityIndices,
    ) -> Self {
        Self {
            authority_committee,
            network_committee,
            own_network_index,
            modality_committees,
            own_modality_indices,
        }
    }
}

impl NetworkingContext for EncoderContext {
    fn network_committee(&self) -> &NetworkCommittee {
        &self.network_committee
    }
    fn own_network_index(&self) -> NetworkingIndex {
        self.own_network_index
    }
}

#[derive(Clone)]
/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
pub(crate) struct LeaderContext {
    /// the committee of validators
    pub(crate) authority_committee: AuthorityCommittee,
    /// the committee of allowed network keys
    pub(crate) network_committee: NetworkCommittee,
    /// the services own index for the network committee
    pub(crate) own_network_index: NetworkingIndex,
    /// The encoder committee for each registered modality
    pub(crate) modality_committees: ModalityCommittee,
    /// The services own index for each respective modality
    pub(crate) own_modality_indices: OwnModalityIndices,
    // TODO: add configs
    // TODO: add metrics back
}

impl LeaderContext {
    pub(crate) fn new(
        authority_committee: AuthorityCommittee,
        network_committee: NetworkCommittee,
        own_network_index: NetworkingIndex,
        modality_committees: ModalityCommittee,
        own_modality_indices: OwnModalityIndices,
    ) -> Self {
        Self {
            authority_committee,
            network_committee,
            own_network_index,
            modality_committees,
            own_modality_indices,
        }
    }
}

impl NetworkingContext for LeaderContext {
    fn network_committee(&self) -> &NetworkCommittee {
        &self.network_committee
    }
    fn own_network_index(&self) -> NetworkingIndex {
        self.own_network_index
    }
}
