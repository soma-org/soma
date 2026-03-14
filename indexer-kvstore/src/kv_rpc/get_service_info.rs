// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::api::error::RpcError;
use rpc::proto::soma::GetServiceInfoResponse;
use rpc::proto::timestamp_ms_to_proto;

use crate::KeyValueStoreReader;

use super::KvRpcServer;

pub async fn build_response(server: &KvRpcServer) -> Result<GetServiceInfoResponse, RpcError> {
    let mut client = server.client.clone();

    let watermark = client.get_watermark().await?;

    let (epoch, checkpoint_height, timestamp) = match watermark {
        Some(wm) => (
            Some(wm.epoch_hi_inclusive),
            Some(wm.checkpoint_hi_inclusive),
            Some(timestamp_ms_to_proto(wm.timestamp_ms_hi_inclusive)),
        ),
        None => (None, None, None),
    };

    let mut response = GetServiceInfoResponse::default();
    response.chain_id = server.chain_id.clone();
    response.epoch = epoch;
    response.checkpoint_height = checkpoint_height;
    response.timestamp = timestamp;
    response.server = server.server_version.clone();
    Ok(response)
}
