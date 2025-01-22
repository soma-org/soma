use shared::{
    authority_committee::AuthorityCommittee,
    network_committee::{NetworkCommittee, NetworkingIndex},
};

use super::encoder_committee::{EncoderCommittee, EncoderIndex};

type Quorum = usize;

#[derive(Clone)]
/// EncoderContext is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
pub(crate) struct EncoderContext {
    /// the committee of validators
    pub authority_committee: AuthorityCommittee,
    /// the committee of allowed network keys
    pub network_committee: NetworkCommittee,
    /// the services own index for the network committee
    pub own_network_index: NetworkingIndex,
    /// The encoder committee for each registered modality
    pub encoder_committee: EncoderCommittee,
    /// The services own index for each respective modality
    pub own_encoder_index: EncoderIndex,

    // TODO change this quorum to be better
    pub shard_quorum: Quorum,
    // TODO: add configs
    // TODO: add metrics back
}

impl EncoderContext {
    pub(crate) fn new(
        authority_committee: AuthorityCommittee,
        network_committee: NetworkCommittee,
        own_network_index: NetworkingIndex,
        encoder_committee: EncoderCommittee,
        own_encoder_index: EncoderIndex,
        shard_quorum: Quorum,
    ) -> Self {
        Self {
            authority_committee,
            network_committee,
            own_network_index,
            encoder_committee,
            own_encoder_index,
            shard_quorum,
        }
    }
}
