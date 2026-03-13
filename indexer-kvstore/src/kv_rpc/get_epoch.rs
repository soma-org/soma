// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::api::error::RpcError;
use rpc::proto::google::rpc::bad_request::FieldViolation;
use rpc::proto::soma::{Epoch, ErrorReason, GetEpochRequest, GetEpochResponse};
use rpc::proto::timestamp_ms_to_proto;
use rpc::utils::field::{FieldMask, FieldMaskTree, FieldMaskUtil};

use crate::KeyValueStoreReader;

use super::KvRpcServer;

pub const READ_MASK_DEFAULT: &str =
    "epoch,first_checkpoint,last_checkpoint,start,end";

pub async fn get_epoch(
    server: &KvRpcServer,
    request: GetEpochRequest,
) -> Result<GetEpochResponse, RpcError> {
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

    let mut client = server.client.clone();

    let epoch_data = match request.epoch {
        Some(epoch_id) => client
            .get_epoch(epoch_id)
            .await?
            .ok_or_else(|| {
                RpcError::new(
                    rpc_tonic::Code::NotFound,
                    format!("Epoch {epoch_id} not found"),
                )
            })?,
        None => client.get_latest_epoch().await?.ok_or_else(|| {
            RpcError::new(rpc_tonic::Code::NotFound, "No epoch data available")
        })?,
    };

    let mut message = Epoch::default();

    if read_mask.contains(Epoch::EPOCH_FIELD.name) {
        message.epoch = epoch_data.epoch;
    }

    if read_mask.contains(Epoch::FIRST_CHECKPOINT_FIELD.name) {
        message.first_checkpoint = epoch_data.start_checkpoint;
    }

    if read_mask.contains(Epoch::LAST_CHECKPOINT_FIELD.name) {
        message.last_checkpoint = epoch_data.end_checkpoint;
    }

    if read_mask.contains(Epoch::START_FIELD.name) {
        message.start = epoch_data.start_timestamp_ms.map(timestamp_ms_to_proto);
    }

    if read_mask.contains(Epoch::END_FIELD.name) {
        message.end = epoch_data.end_timestamp_ms.map(timestamp_ms_to_proto);
    }

    if read_mask.contains(Epoch::SYSTEM_STATE_FIELD.name) {
        if let Some(system_state_bcs) = &epoch_data.system_state_bcs {
            // Try to deserialize and convert to proto SystemState
            if let Ok(system_state) =
                bcs::from_bytes::<types::system_state::SystemState>(system_state_bcs)
            {
                message.system_state = system_state.try_into().ok().map(Box::new);
            }
        }
    }

    Ok(GetEpochResponse::new(message))
}
