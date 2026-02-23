// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use types::{
    checkpoints::{CheckpointSummary, CheckpointTimestamp},
    committee::EpochId,
    digests::CheckpointDigest,
    envelope::Message as _,
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
        EpochStartConfiguration { system_state, epoch_digest }
    }

    pub fn epoch_data(&self) -> EpochData {
        EpochData::new(
            self.epoch_start_state().epoch(),
            self.epoch_start_state().epoch_start_timestamp_ms(),
            self.epoch_digest(),
        )
    }

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

/// The static epoch information that is accessible to move smart contracts
#[derive(Default)]
pub struct EpochData {
    epoch_id: EpochId,
    epoch_start_timestamp: CheckpointTimestamp,
    epoch_digest: CheckpointDigest,
}

impl EpochData {
    pub fn new(
        epoch_id: EpochId,
        epoch_start_timestamp: CheckpointTimestamp,
        epoch_digest: CheckpointDigest,
    ) -> Self {
        Self { epoch_id, epoch_start_timestamp, epoch_digest }
    }

    pub fn new_genesis(epoch_start_timestamp: CheckpointTimestamp) -> Self {
        Self { epoch_id: 0, epoch_start_timestamp, epoch_digest: Default::default() }
    }

    pub fn new_from_epoch_checkpoint(
        epoch_id: EpochId,
        epoch_checkpoint: &CheckpointSummary,
    ) -> Self {
        Self {
            epoch_id,
            epoch_start_timestamp: epoch_checkpoint.timestamp_ms,
            epoch_digest: epoch_checkpoint.digest(),
        }
    }

    pub fn new_test() -> Self {
        Default::default()
    }

    pub fn epoch_id(&self) -> EpochId {
        self.epoch_id
    }

    pub fn epoch_start_timestamp(&self) -> CheckpointTimestamp {
        self.epoch_start_timestamp
    }

    pub fn epoch_digest(&self) -> CheckpointDigest {
        self.epoch_digest
    }
}
