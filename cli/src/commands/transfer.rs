// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the transfer command (transfer an object to a recipient)
pub async fn execute(
    context: &mut WalletContext,
    to: KeyIdentity,
    object_id: ObjectID,
    gas: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.get_object_owner(&object_id).await?;
    let recipient = context.get_identity_address(Some(to))?;
    let client = context.get_client().await?;

    // Get the object reference
    let object =
        client.get_object(object_id).await.map_err(|e| anyhow!("Failed to get object: {}", e))?;
    let object_ref = object.compute_object_reference();

    let kind = TransactionKind::TransferObjects { objects: vec![object_ref], recipient };

    // Resolve gas payment
    let gas_ref = match gas {
        Some(gas_id) => {
            let gas_obj = client
                .get_object(gas_id)
                .await
                .map_err(|e| anyhow!("Failed to get gas object: {}", e))?;
            Some(gas_obj.compute_object_reference())
        }
        None => None,
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_ref, tx_args).await
}
