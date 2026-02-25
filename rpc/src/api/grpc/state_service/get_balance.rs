// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::base::SomaAddress;

use crate::api::RpcService;
use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{ErrorReason, GetBalanceRequest, GetBalanceResponse};

#[tracing::instrument(skip(service))]
pub fn get_balance(service: &RpcService, request: GetBalanceRequest) -> Result<GetBalanceResponse> {
    let indexes = service.reader.inner().indexes().ok_or_else(RpcError::not_found)?;

    let owner = request
        .owner
        .as_ref()
        .ok_or_else(|| {
            FieldViolation::new("owner")
                .with_description("missing owner")
                .with_reason(ErrorReason::FieldMissing)
        })?
        .parse::<SomaAddress>()
        .map_err(|e| {
            FieldViolation::new("owner")
                .with_description(format!("invalid owner: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    let balance_info = indexes.get_balance(&owner)?.unwrap_or_default(); // Use default (zero) if no balance found

    let response = GetBalanceResponse { balance: Some(balance_info.balance), ..Default::default() };
    Ok(response)
}
