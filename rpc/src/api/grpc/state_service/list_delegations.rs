// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `StateService::ListDelegations` — read every active delegation for
//! a staker from the on-chain `delegations` column family. Stage 9d
//! lays the read path; Stage 9e (separate session) will switch
//! consumers off the `ListOwnedObjects` + StakedSomaV1 scan.

use types::base::SomaAddress;

use crate::api::RpcService;
use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{
    DelegationEntry, ErrorReason, ListDelegationsRequest, ListDelegationsResponse,
};

#[tracing::instrument(skip(service))]
pub fn list_delegations(
    service: &RpcService,
    request: ListDelegationsRequest,
) -> Result<ListDelegationsResponse> {
    let indexes = service.reader.inner().indexes().ok_or_else(RpcError::not_found)?;

    let staker = request
        .staker
        .as_ref()
        .ok_or_else(|| {
            FieldViolation::new("staker")
                .with_description("missing staker")
                .with_reason(ErrorReason::FieldMissing)
        })?
        .parse::<SomaAddress>()
        .map_err(|e| {
            FieldViolation::new("staker")
                .with_description(format!("invalid staker: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    let rows = indexes.list_delegations(&staker)?;

    // Pre-sum total here so the client can show "total stake" without
    // a separate round-trip and without summing on the client (where
    // a malicious node could omit rows).
    let mut total: u64 = 0;
    let mut delegations = Vec::with_capacity(rows.len());
    for row in rows {
        total = total.saturating_add(row.principal);
        // Auto-compound: the proto schema's `last_collected_period`
        // is a legacy F1 mark that no longer maps cleanly — the real
        // baseline is `index_at_last_collect` (u128). Hardcode 0 as a
        // placeholder until the proto is reshaped.
        delegations.push(DelegationEntry {
            pool_id: Some(row.pool_id.to_string()),
            principal: Some(row.principal.saturating_add(row.pending_principal)),
            last_collected_period: Some(0),
        });
    }

    Ok(ListDelegationsResponse {
        delegations,
        total_principal: Some(total),
    })
}
