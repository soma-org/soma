// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use types::base::SomaAddress;
use types::object::CoinType;

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

    // Stage 13c: balances live in the per-(owner, coin_type)
    // accumulator. Default to USDC when the request omits coin_type
    // — that's the gas / typical transferable currency.
    let coin_type = match request.coin_type.as_deref() {
        None | Some("") | Some("USDC") => CoinType::Usdc,
        Some("SOMA") => CoinType::Soma,
        Some(other) => {
            return Err(FieldViolation::new("coin_type")
                .with_description(format!("unknown coin_type: {other}"))
                .with_reason(ErrorReason::FieldInvalid)
                .into());
        }
    };

    let balance_info = indexes.get_balance(&owner, coin_type)?.unwrap_or_default();

    let response = GetBalanceResponse { balance: Some(balance_info.balance), ..Default::default() };
    Ok(response)
}
