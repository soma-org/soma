// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::api::RpcService;
use crate::api::error::Result;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::Epoch;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::GetEpochRequest;
use crate::proto::soma::GetEpochResponse;
// use crate::proto::soma::ProtocolConfig;
use crate::proto::timestamp_ms_to_proto;
use crate::types::EpochId;
use crate::utils::field::FieldMaskTree;
use crate::utils::field::FieldMaskUtil;
use crate::utils::merge::Merge;
use prost_types::FieldMask;

pub const READ_MASK_DEFAULT: &str = "epoch,committee,first_checkpoint,last_checkpoint,start,end,reference_gas_price,protocol_config.protocol_version";

#[tracing::instrument(skip(service))]
pub fn get_epoch(service: &RpcService, request: GetEpochRequest) -> Result<GetEpochResponse> {
    let read_mask = {
        let read_mask = request
            .read_mask
            .unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Epoch>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let mut message = Epoch::default();

    let current_epoch = service.reader.inner().get_highest_synced_commit()?.epoch();
    let epoch = request.epoch.unwrap_or(current_epoch);

    let mut system_state =
        if epoch == current_epoch && read_mask.contains(Epoch::SYSTEM_STATE_FIELD.name) {
            Some(service.reader.get_system_state()?)
        } else {
            None
        };

    if read_mask.contains(Epoch::EPOCH_FIELD.name) {
        message.epoch = Some(epoch);
    }

    if let Some(epoch_info) = service
        .reader
        .inner()
        .indexes()
        .and_then(|indexes| indexes.get_epoch_info(epoch).ok().flatten())
    {
        if read_mask.contains(Epoch::START_FIELD.name) {
            message.start = epoch_info.start_timestamp_ms.map(timestamp_ms_to_proto);
        }

        if read_mask.contains(Epoch::END_FIELD.name) {
            message.end = epoch_info.end_timestamp_ms.map(timestamp_ms_to_proto);
        }

        // If we're not loading the current epoch then grab the indexed snapshot of the system
        // state at the start of the epoch.
        if system_state.is_none() {
            system_state = epoch_info.system_state;
        }
    }

    if let Some(system_state) = system_state {
        if read_mask.contains(Epoch::SYSTEM_STATE_FIELD.name) {
            message.system_state = Some(Box::new(
                system_state
                    .try_into()
                    .map_err(|e| SystemStateNotFoundError::new(epoch, e))?,
            ));
        }
    }

    if read_mask.contains(Epoch::COMMITTEE_FIELD.name) {
        message.committee = Some(
            service
                .reader
                .get_committee(epoch)
                .ok_or_else(|| CommitteeNotFoundError::new(epoch))?
                .into(),
        );
    }

    Ok(GetEpochResponse::new(message))
}

#[derive(Debug)]
pub struct CommitteeNotFoundError {
    epoch: EpochId,
}

impl CommitteeNotFoundError {
    pub fn new(epoch: EpochId) -> Self {
        Self { epoch }
    }
}

impl std::fmt::Display for CommitteeNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Committee for epoch {} not found", self.epoch)
    }
}

impl std::error::Error for CommitteeNotFoundError {}

impl From<CommitteeNotFoundError> for crate::api::error::RpcError {
    fn from(value: CommitteeNotFoundError) -> Self {
        Self::new(tonic::Code::NotFound, value.to_string())
    }
}

#[derive(Debug)]
struct ProtocolVersionNotFoundError {
    version: u64,
}

impl ProtocolVersionNotFoundError {
    pub fn new(version: u64) -> Self {
        Self { version }
    }
}

impl std::fmt::Display for ProtocolVersionNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Protocol version {} not found", self.version)
    }
}

impl std::error::Error for ProtocolVersionNotFoundError {}

impl From<ProtocolVersionNotFoundError> for crate::api::error::RpcError {
    fn from(value: ProtocolVersionNotFoundError) -> Self {
        Self::new(tonic::Code::NotFound, value.to_string())
    }
}

#[derive(Debug)]
struct SystemStateNotFoundError {
    epoch: EpochId,
    error: String,
}

impl SystemStateNotFoundError {
    pub fn new(epoch: EpochId, error: String) -> Self {
        Self { epoch, error }
    }
}

impl std::fmt::Display for SystemStateNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SystemState not found for epoch {}: {}",
            self.epoch, self.error
        )
    }
}

impl std::error::Error for SystemStateNotFoundError {}

impl From<SystemStateNotFoundError> for crate::api::error::RpcError {
    fn from(value: SystemStateNotFoundError) -> Self {
        Self::new(tonic::Code::NotFound, value.to_string())
    }
}
