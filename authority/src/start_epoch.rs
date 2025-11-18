use serde::{Deserialize, Serialize};
use types::{
    checkpoints::CheckpointTimestamp,
    digests::CheckpointDigest,
    system_state::epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait},
};

pub trait EpochStartConfigTrait {
    fn epoch_start_state(&self) -> &EpochStartSystemState;
    fn epoch_digest(&self) -> CheckpointDigest;
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EpochStartConfiguration {
    system_state: EpochStartSystemState,
    epoch_digest: CheckpointDigest,
}

impl EpochStartConfiguration {
    pub fn new(system_state: EpochStartSystemState, epoch_digest: CheckpointDigest) -> Self {
        EpochStartConfiguration {
            system_state,
            epoch_digest,
        }
    }

    // pub fn epoch_data(&self) -> EpochData {
    //     EpochData::new(
    //         self.epoch_start_state().epoch(),
    //         self.epoch_start_state().epoch_start_timestamp_ms(),
    //     )
    // }

    pub fn new_at_next_epoch_for_testing(&self) -> Self {
        // We only need to implement this function for the latest version.
        // When a new version is introduced, this function should be updated.

        match self {
            config => EpochStartConfiguration {
                system_state: config.system_state.clone(),
                epoch_digest: config.epoch_digest,
            },
            _ => panic!(
                "This function is only implemented for the latest version of \
                 EpochStartConfiguration"
            ),
        }
    }

    pub fn epoch_start_timestamp_ms(&self) -> CheckpointTimestamp {
        self.epoch_start_state().epoch_start_timestamp_ms()
    }
}

impl EpochStartConfigTrait for EpochStartConfiguration {
    fn epoch_start_state(&self) -> &EpochStartSystemState {
        &self.system_state
    }

    fn epoch_digest(&self) -> CheckpointDigest {
        self.epoch_digest
    }
}
