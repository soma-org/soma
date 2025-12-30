use anyhow::{anyhow, Result};
use clap::Parser;
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::object::ObjectID;

use crate::response::{
    ClaimableItem, ClaimableListOutput, ShardListOutput, ShardStatus, ShardStatusOutput,
    ShardSummary, ShardsQueryResponse, TargetSummary, TargetsOutput,
};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum ShardsCommand {
    /// Get detailed status of a specific shard
    #[clap(name = "status")]
    Status {
        /// Shard object ID
        shard_id: ObjectID,
    },

    /// List shards by submitter address
    #[clap(name = "list")]
    List {
        /// Submitter address (defaults to active address)
        #[clap(long)]
        submitter: Option<SomaAddress>,
        /// Filter by epoch
        #[clap(long)]
        epoch: Option<u64>,
    },

    /// List shards won by an encoder
    #[clap(name = "by-encoder")]
    ByEncoder {
        /// Encoder address
        encoder: SomaAddress,
    },

    /// List valid targets for competition
    #[clap(name = "targets")]
    Targets {
        /// Epoch (defaults to current epoch)
        #[clap(long)]
        epoch: Option<u64>,
    },

    /// List escrows that can be claimed
    #[clap(name = "claimable-escrows")]
    ClaimableEscrows,

    /// List rewards that can be claimed
    #[clap(name = "claimable-rewards")]
    ClaimableRewards,
}

/// Execute the shards command
pub async fn execute(
    context: &mut WalletContext,
    cmd: ShardsCommand,
) -> Result<ShardsQueryResponse> {
    let client = context.get_client().await?;

    match cmd {
        ShardsCommand::Status { shard_id } => {
            let shard = client
                .get_shard(shard_id)
                .await
                .map_err(|e| anyhow!("Failed to get shard: {}", e))?;

            // Determine shard status based on its state
            let status = if shard.winning_encoder.is_some() {
                ShardStatus::Completed
            } else {
                // Could be pending or claimable - would need to check epoch
                ShardStatus::PendingEncoding
            };

            Ok(ShardsQueryResponse::Status(ShardStatusOutput {
                shard_id,
                status,
                created_epoch: shard.created_epoch,
                data_submitter: shard.data_submitter,
                amount: shard.amount,
                winning_encoder: shard.winning_encoder,
                target_id: shard.target.map(|t| t.0),
            }))
        }

        ShardsCommand::List { submitter, epoch } => {
            let address = submitter.unwrap_or(context.active_address()?);

            let response = client
                .get_shards_by_submitter_address(&address, epoch)
                .await
                .map_err(|e| anyhow!("Failed to get shards: {}", e))?;

            let shards: Vec<ShardSummary> = response
                .shards
                .into_iter()
                .filter_map(|s| {
                    let shard_id = s.shard_id.and_then(|id| ObjectID::from_bytes(&id).ok())?;
                    Some(ShardSummary {
                        shard_id,
                        created_epoch: s.created_epoch?,
                        status: if s.winning_encoder.is_some() {
                            ShardStatus::Completed
                        } else {
                            ShardStatus::PendingEncoding
                        },
                        amount: s.amount?,
                    })
                })
                .collect();

            let total_count = shards.len();
            Ok(ShardsQueryResponse::List(ShardListOutput {
                shards,
                total_count,
            }))
        }

        ShardsCommand::ByEncoder { encoder } => {
            let response = client
                .get_shards_by_encoder(encoder.as_ref())
                .await
                .map_err(|e| anyhow!("Failed to get shards by encoder: {}", e))?;

            let shards: Vec<ShardSummary> = response
                .shards
                .into_iter()
                .filter_map(|s| {
                    let shard_id = s.shard_id.and_then(|id| ObjectID::from_bytes(&id).ok())?;
                    Some(ShardSummary {
                        shard_id,
                        created_epoch: s.created_epoch?,
                        status: ShardStatus::Completed, // Won shards are completed
                        amount: s.amount?,
                    })
                })
                .collect();

            let total_count = shards.len();
            Ok(ShardsQueryResponse::List(ShardListOutput {
                shards,
                total_count,
            }))
        }

        ShardsCommand::Targets { epoch } => {
            // Get current epoch if not specified
            let current_epoch = match epoch {
                Some(e) => e,
                None => {
                    let system_state = client
                        .get_latest_system_state()
                        .await
                        .map_err(|e| anyhow!("Failed to get system state: {}", e))?;
                    system_state.epoch
                }
            };

            let response = client
                .get_valid_targets(current_epoch)
                .await
                .map_err(|e| anyhow!("Failed to get valid targets: {}", e))?;

            let targets: Vec<TargetSummary> = response
                .targets
                .into_iter()
                .filter_map(|t| {
                    let has_winner = t.has_winner();
                    let target_id = t.target_id.and_then(|id| ObjectID::from_bytes(&id).ok())?;
                    Some(TargetSummary {
                        target_id,
                        created_epoch: t.created_epoch?,
                        has_winning_shard: has_winner,
                    })
                })
                .collect();

            Ok(ShardsQueryResponse::Targets(TargetsOutput {
                epoch: current_epoch,
                targets,
            }))
        }

        ShardsCommand::ClaimableEscrows => {
            // Get current epoch
            let system_state = client
                .get_latest_system_state()
                .await
                .map_err(|e| anyhow!("Failed to get system state: {}", e))?;

            let response = client
                .get_claimable_escrows(system_state.epoch)
                .await
                .map_err(|e| anyhow!("Failed to get claimable escrows: {}", e))?;

            let items: Vec<ClaimableItem> = response
                .shards
                .into_iter()
                .filter_map(|s| {
                    let object_id = s.shard_id.and_then(|id| ObjectID::from_bytes(&id).ok())?;
                    Some(ClaimableItem {
                        object_id,
                        claimable_amount: s.amount?,
                    })
                })
                .collect();

            let total_count = items.len();
            Ok(ShardsQueryResponse::ClaimableEscrows(ClaimableListOutput {
                items,
                total_count,
            }))
        }

        ShardsCommand::ClaimableRewards => {
            // Get current epoch
            let system_state = client
                .get_latest_system_state()
                .await
                .map_err(|e| anyhow!("Failed to get system state: {}", e))?;

            let response = client
                .get_claimable_rewards(system_state.epoch)
                .await
                .map_err(|e| anyhow!("Failed to get claimable rewards: {}", e))?;

            let items: Vec<ClaimableItem> = response
                .targets
                .into_iter()
                .filter_map(|t| {
                    let reward_amount = t.reward_amount();
                    let object_id = t.target_id.and_then(|id| ObjectID::from_bytes(&id).ok())?;

                    Some(ClaimableItem {
                        object_id,
                        claimable_amount: reward_amount,
                    })
                })
                .collect();

            let total_count = items.len();
            Ok(ShardsQueryResponse::ClaimableRewards(ClaimableListOutput {
                items,
                total_count,
            }))
        }
    }
}
