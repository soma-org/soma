use serde::{Deserialize, Serialize};
use types::{
    state_sync::CommitTimestamp,
    system_state::epoch_start::{EpochStartSystemState, EpochStartSystemStateTrait},
};

pub trait EpochStartConfigTrait {
    fn epoch_start_state(&self) -> &EpochStartSystemState;
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EpochStartConfiguration {
    system_state: EpochStartSystemState,
    // epoch_digest is defined as following
    // (1) For the genesis epoch it is set to 0
    // (2) For all other epochs it is a digest of the last commit of a previous epoch
    // Note that this is in line with how epoch start timestamp is defined
    // epoch_digest: CheckpointDigest,
}

impl EpochStartConfiguration {
    pub fn new(system_state: EpochStartSystemState) -> Self {
        EpochStartConfiguration { system_state }
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
            },
            _ => panic!(
                "This function is only implemented for the latest version of \
                 EpochStartConfiguration"
            ),
        }
    }

    pub fn epoch_start_timestamp_ms(&self) -> CommitTimestamp {
        self.epoch_start_state().epoch_start_timestamp_ms()
    }
}

impl EpochStartConfigTrait for EpochStartConfiguration {
    fn epoch_start_state(&self) -> &EpochStartSystemState {
        &self.system_state
    }
}
