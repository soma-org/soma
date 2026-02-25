// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use types::digests::TransactionDigest;

use crate::response::{ClientCommandResponse, TransactionQueryResponse};

/// Execute the tx command (get transaction details by digest)
pub async fn execute(
    context: &WalletContext,
    digest: TransactionDigest,
) -> Result<ClientCommandResponse> {
    let client = context.get_client().await?;

    let result = client
        .get_transaction(digest)
        .await
        .map_err(|e| anyhow!("Failed to get transaction: {}", e))?;

    Ok(ClientCommandResponse::TransactionQuery(TransactionQueryResponse::from_query_result(
        &result,
    )))
}
