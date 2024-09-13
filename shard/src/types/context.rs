use std::collections::HashMap;

use crate::types::authority_committee::AuthorityCommittee;

use super::{
    encoder_committee::EncoderIndex,
    modality::{Modality, ModalityCommittee, ModalityIndex},
    network_committee::{NetworkCommittee, NetworkIdentityIndex},
};

#[derive(Clone)]
/// Context is updated each epoch and provides the various services running
/// information on committeees, configurations, and access to common metric reporting
pub(crate) struct Context {
    /// the committee of validators
    pub(crate) authority_committee: AuthorityCommittee,
    /// the committee of allowed network keys
    pub(crate) network_committee: NetworkCommittee,
    /// the services own index for the network committee
    pub(crate) own_network_index: NetworkIdentityIndex,
    /// The encoder committee for each registered modality
    pub(crate) modality_committees: HashMap<Modality, ModalityCommittee>,
    /// The services own index for each respective modality
    pub(crate) own_modality_indices: HashMap<Modality, ModalityIndex>,
    // TODO: add configs
    // /// Parameters of this authority.
    //  parameters: Parameters,
    // /// Protocol configuration of current epoch.
    //  protocol_config: ProtocolConfig,

    // TODO: add metrics back
    // /// Metrics of this authority.
    //  metrics: Arc<Metrics>,
}

impl Context {
    //  fn new(
    //     own_index: AuthorityIndex,
    //     committee: AuthorityCommittee,

    //     // parameters: Parameters,
    //     // protocol_config: ProtocolConfig,
    //     // metrics: Arc<Metrics>,
    //     // clock: Arc<Clock>,
    // ) -> Self {
    //     Self {
    //         own_index,
    //         committee,
    //         // parameters,
    //         // protocol_config,
    //         // metrics,
    //         // clock,
    //     }
    // }

    // /// Create a test context with a committee of given size and even stake
    // #[cfg(test)]
    //  fn new_for_test(
    //     committee_size: usize,
    // ) -> (Self, Vec<(NetworkKeyPair, ProtocolKeyPair)>) {
    //     let (committee, keypairs) =
    //         consensus_config::local_committee_and_keys(0, vec![1; committee_size]);
    //     let metrics = test_metrics();
    //     let temp_dir = TempDir::new().unwrap();
    //     let clock = Arc::new(Clock::new());

    //     let context = Context::new(
    //         AuthorityIndex::new_for_test(0),
    //         committee,
    //         Parameters {
    //             db_path: temp_dir.into_path(),
    //             ..Default::default()
    //         },
    //         ProtocolConfig::get_for_max_version_UNSAFE(),
    //         metrics,
    //         clock,
    //     );
    //     (context, keypairs)
    // }

    // #[cfg(test)]
    //  fn with_authority_index(mut self, authority: AuthorityIndex) -> Self {
    //     self.own_index = authority;
    //     self
    // }

    // #[cfg(test)]
    //  fn with_committee(mut self, committee: Committee) -> Self {
    //     self.committee = committee;
    //     self
    // }

    // #[cfg(test)]
    //  fn with_parameters(mut self, parameters: Parameters) -> Self {
    //     self.parameters = parameters;
    //     self
    // }
}
