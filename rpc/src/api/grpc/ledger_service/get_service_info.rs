// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use tap::Pipe;

use crate::api::RpcService;
use crate::api::error::RpcError;
use crate::proto::soma::GetServiceInfoResponse;
use crate::proto::timestamp_ms_to_proto;
use crate::types::Digest;

#[tracing::instrument(skip(service))]
pub fn get_service_info(service: &RpcService) -> Result<GetServiceInfoResponse, RpcError> {
    let latest_checkpoint = service.reader.inner().get_latest_checkpoint()?;
    let lowest_available_checkpoint =
        service.reader.inner().get_lowest_available_checkpoint()?.pipe(Some);
    let lowest_available_checkpoint_objects =
        service.reader.inner().get_lowest_available_checkpoint_objects()?.pipe(Some);

    let message = GetServiceInfoResponse {
        chain_id: Some(Digest::new(service.chain_id().as_bytes().to_owned()).to_string()),
        chain: Some(service.chain_id().chain().as_str().into()),
        epoch: Some(latest_checkpoint.epoch()),
        checkpoint_height: Some(latest_checkpoint.sequence_number),
        timestamp: Some(timestamp_ms_to_proto(latest_checkpoint.timestamp_ms)),
        lowest_available_checkpoint,
        lowest_available_checkpoint_objects,
        server: service.server_version().map(ToString::to_string),
        ..Default::default()
    };
    Ok(message)
}
