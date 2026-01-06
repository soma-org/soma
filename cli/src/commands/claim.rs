use anyhow::{anyhow, bail, Result};
use sdk::wallet_context::WalletContext;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the claim command (claim escrow from failed shards or rewards from targets)
pub async fn execute(
    context: &mut WalletContext,
    escrow: Option<ObjectID>,
    reward: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    // Validate that exactly one claim type is specified
    if escrow.is_none() && reward.is_none() {
        bail!("Must specify either --escrow (shard ID) or --reward (target ID)");
    }
    if escrow.is_some() && reward.is_some() {
        bail!("Cannot specify both --escrow and --reward");
    }

    let sender = context.active_address()?;
    let client = context.get_client().await?;

    let kind = if let Some(shard_id) = escrow {
        // Claim escrow from a failed/expired shard
        let shard_obj = client
            .get_object(shard_id)
            .await
            .map_err(|e| anyhow!("Failed to get shard: {}", e))?;
        let shard_ref = shard_obj.compute_object_reference();

        TransactionKind::ClaimEscrow { shard_ref }
    } else if let Some(target_id) = reward {
        // Claim reward from a completed target
        let target_obj = client
            .get_object(target_id)
            .await
            .map_err(|e| anyhow!("Failed to get target: {}", e))?;
        let target_ref = target_obj.compute_object_reference();

        TransactionKind::ClaimReward { target_ref }
    } else {
        unreachable!()
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, None, tx_args).await
}
