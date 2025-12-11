use std::collections::HashSet;
use tracing::{debug, info};
use types::{
    base::SomaAddress,
    committee::EpochId,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ConsensusError, ExecutionResult, SomaError},
    metadata::{DownloadMetadata, MetadataAPI as _},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version},
    report::Report,
    shard::ShardAuthToken,
    shard_crypto::{
        keys::{EncoderAggregateSignature, EncoderPublicKey},
        scope::{Scope, ScopedMessage},
    },
    shard_verifier::ShardVerifier,
    system_state::{
        shard::{Shard, Target, TargetOrigin, WinningShardInfo},
        SystemState,
    },
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::{FeeCalculator, TransactionExecutor};

/// Basis points for percentage calculations (10000 = 100%)
const BPS_DENOMINATOR: u64 = 10000;

/// Percentage of escrow/reward given to claim submitter (basis points)
/// e.g., 50 = 0.5%
const CLAIM_INCENTIVE_BPS: u64 = 50;

pub struct ShardExecutor {
    shard_verifier: ShardVerifier,
}

impl ShardExecutor {
    pub fn new() -> Self {
        Self {
            shard_verifier: ShardVerifier::new(100),
        }
    }

    // =========================================================================
    // EMBED DATA
    // =========================================================================
    fn execute_embed_data(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        download_metadata: DownloadMetadata,
        coin_ref: ObjectRef,
        target_ref: Option<ObjectRef>,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;
        let is_gas_coin = store.gas_object_id == Some(coin_id);

        // 1. Validate metadata
        let data_size = download_metadata.metadata().size() as u64;
        if data_size == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Data size cannot be zero".to_string(),
            });
        }

        // 2. Load and validate system state
        let state = self.load_system_state(store)?;
        let current_epoch = state.epoch;
        let byte_price = state.encoders.reference_byte_price;

        // 3. Calculate embed price
        let embed_price = byte_price.saturating_mul(data_size);
        if embed_price == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Calculated embed price is zero".to_string(),
            });
        }

        // 4. Validate target if provided
        if let Some(target) = &target_ref {
            let target_object = store.read_object(&target.0).ok_or_else(|| {
                ExecutionFailureStatus::ObjectNotFound {
                    object_id: target.0,
                }
            })?;

            let target_data = target_object.as_target().ok_or_else(|| {
                ExecutionFailureStatus::InvalidObjectType {
                    object_id: target.0,
                    expected_type: ObjectType::Target,
                    actual_type: target_object.type_().clone(),
                }
            })?;

            // Target validity check: valid at created_epoch + 1
            if current_epoch < target_data.created_epoch + 1 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: format!(
                        "Target {} not yet valid. Created: {}, Current: {}, Valid at: {}",
                        target.0,
                        target_data.created_epoch,
                        current_epoch,
                        target_data.created_epoch + 1
                    ),
                });
            }

            // Shard created_epoch must equal target.created_epoch + 1
            // (Target created in E, valid in E+1, shards competing must be from E+1)
            if current_epoch != target_data.created_epoch + 1 {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: format!(
                        "Target epoch mismatch. Target created: {}, Current: {}, Required: {}",
                        target_data.created_epoch,
                        current_epoch,
                        target_data.created_epoch + 1
                    ),
                });
            }
        }

        // 5. Load and validate source coin
        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        self.check_ownership(&source_object, signer)?;

        let source_balance =
            source_object
                .as_coin()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: coin_id,
                    expected_type: ObjectType::Coin,
                    actual_type: source_object.type_().clone(),
                })?;

        // 6. Calculate total required
        let operation_fee = self.calculate_operation_fee(2);
        let total_required = if is_gas_coin {
            embed_price
                .saturating_add(value_fee)
                .saturating_add(operation_fee)
        } else {
            embed_price
        };

        if source_balance < total_required {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance);
        }

        // 7. Create Shard object as shared
        let shard_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let shard = Object::new_shard(
            shard_id,
            download_metadata,
            embed_price,
            current_epoch,
            signer,
            target_ref,
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            tx_digest,
        );
        store.create_object(shard);

        // 8. Update source coin
        let remaining_balance = source_balance - embed_price;
        if remaining_balance == 0 && !is_gas_coin {
            store.delete_input_object(&coin_id);
        } else {
            let mut updated_source = source_object.clone();
            updated_source.update_coin_balance(remaining_balance);
            store.mutate_input_object(updated_source);
        }

        info!(
            "EmbedData: Created shard {} with amount {} in epoch {}",
            shard_id, embed_price, current_epoch
        );

        Ok(())
    }

    // =========================================================================
    // REPORT WINNER
    // =========================================================================
    fn execute_report_winner(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        shard_ref: ObjectRef,
        target_ref: Option<ObjectRef>,
        report_bytes: Vec<u8>,
        signature_bytes: Vec<u8>,
        signers: Vec<EncoderPublicKey>,
        shard_auth_token_bytes: Vec<u8>,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let shard_id = shard_ref.0;

        // 1. Load shard object
        let shard_object =
            store
                .read_object(&shard_id)
                .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                    object_id: shard_id,
                })?;

        let shard_data =
            shard_object
                .as_shard()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: shard_id,
                    expected_type: ObjectType::Shard,
                    actual_type: shard_object.type_().clone(),
                })?;

        // 2. Check if shard already has a winner
        if shard_data.winning_encoder.is_some() {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Shard already has a winning encoder".to_string(),
            });
        }

        // 3. Load system state and check epoch
        let state = self.load_system_state(store)?;
        let current_epoch = state.epoch;

        // Valid reporting window: created_epoch or created_epoch + 1
        if current_epoch > shard_data.created_epoch + 1 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Shard reporting window closed. Created: {}, Current: {}, Deadline: {}",
                    shard_data.created_epoch,
                    current_epoch,
                    shard_data.created_epoch + 1
                ),
            });
        }

        // 4. Deserialize report
        let report: Report = bcs::from_bytes(&report_bytes).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to deserialize report: {}",
                e
            )))
        })?;

        // 5. Deserialize shard auth token
        let shard_auth_token: ShardAuthToken =
            bcs::from_bytes(&shard_auth_token_bytes).map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize shard auth token: {}",
                    e
                )))
            })?;

        // 6. Verify shard auth token
        let token_epoch = shard_auth_token.epoch();
        let committees = state
            .committees(token_epoch)
            .map_err(|e| ExecutionFailureStatus::SomaError(e))?;

        let authority_committee = committees.build_validator_committee().committee().clone();
        let encoder_committee = committees.build_encoder_committee();
        let vdf_iterations = state.parameters.vdf_iterations;

        let (verified_shard, _) = self
            .shard_verifier
            .verify(
                authority_committee,
                encoder_committee,
                vdf_iterations,
                &shard_auth_token,
            )
            .map_err(|e| ExecutionFailureStatus::InvalidArguments {
                reason: format!("Shard verification failed: {:?}", e),
            })?;

        // 7. Verify signers meet quorum
        let unique_signers: HashSet<&EncoderPublicKey> = signers.iter().collect();

        for encoder in &unique_signers {
            if !verified_shard.encoders().contains(encoder) {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: format!("Encoder {:?} is not in the shard", encoder),
                });
            }
        }

        let quorum_threshold = verified_shard.quorum_threshold();
        if (unique_signers.len() as u32) < quorum_threshold {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Insufficient signers: got {}, required {}",
                    unique_signers.len(),
                    quorum_threshold
                ),
            });
        }

        // 8. Verify aggregate signature
        let message = bcs::to_bytes(&ScopedMessage::new(Scope::ShardReport, report.clone()))
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to serialize scoped message: {}",
                    e
                )))
            })?;

        let signature = EncoderAggregateSignature::from_bytes(&signature_bytes).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to deserialize signature: {}",
                e
            )))
        })?;

        signature.verify(&signers, &message).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                ConsensusError::SignatureVerificationFailure(e),
            ))
        })?;

        // 9. Update shard object with winner and embeddings
        let mut updated_shard_data = shard_data.clone();
        updated_shard_data.winning_encoder = Some(report.winner.clone());
        updated_shard_data.embeddings_download_metadata =
            Some(report.embeddings_download_metadata.clone());
        updated_shard_data.evaluation_scores = Some(report.scores);
        updated_shard_data.target_scores = Some(report.distance);
        updated_shard_data.sampled_embedding = Some(report.sampled_embedding.clone());
        updated_shard_data.summary_embedding = Some(report.summary_embedding.clone());

        let mut updated_shard = shard_object.clone();
        updated_shard
            .data
            .update_contents(bcs::to_bytes(&updated_shard_data).map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to serialize shard: {}",
                    e
                )))
            })?);
        store.mutate_input_object(updated_shard);

        // 11. Update target if provided
        if let Some(target) = target_ref {
            self.try_update_target_winner(
                store,
                target,
                shard_ref,
                &shard_data,
                &report,
                &state,
                current_epoch,
            )?;
        }

        info!(
            "ReportWinner: Shard {} won by encoder {:?}",
            shard_id, report.winner
        );

        Ok(())
    }

    // =========================================================================
    // TRY UPDATE TARGET WINNER
    // =========================================================================
    fn try_update_target_winner(
        &self,
        store: &mut TemporaryStore,
        target_ref: ObjectRef,
        shard_ref: ObjectRef,
        shard_data: &Shard,
        report: &Report,
        state: &SystemState,
        current_epoch: EpochId,
    ) -> ExecutionResult<()> {
        let target_id = target_ref.0;

        // 1. Load target object
        let target_object = store.read_object(&target_id).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound {
                object_id: target_id,
            }
        })?;

        let target_data =
            target_object
                .as_target()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: target_id,
                    expected_type: ObjectType::Target,
                    actual_type: target_object.type_().clone(),
                })?;

        // 2. Verify target is valid (valid at created_epoch + 1)
        if current_epoch < target_data.created_epoch + 1 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Target {} not yet valid. Created: {}, Current: {}, Valid at: {}",
                    target_id,
                    target_data.created_epoch,
                    current_epoch,
                    target_data.created_epoch + 1
                ),
            });
        }

        // 3. Verify epoch compatibility
        // Shard must be created in target.created_epoch + 1
        if shard_data.created_epoch != target_data.created_epoch + 1 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Shard epoch {} incompatible with target. Expected shard from epoch {}",
                    shard_data.created_epoch,
                    target_data.created_epoch + 1
                ),
            });
        }

        // 4. Determine if we should update
        let should_update = match &target_data.winning_shard {
            None => true, // No winner yet
            Some(current_winner) => {
                // Check if current winner's encoder was tallied
                if state.is_encoder_tallied(&current_winner.winning_encoder) {
                    // Current winner was slashed, allow replacement
                    true
                } else {
                    // Compare distances - lower is better
                    report.distance < current_winner.distance
                }
            }
        };

        if !should_update {
            debug!(
                "Target {} not updated - current winner has better distance",
                target_id
            );
            return Ok(());
        }

        // 5. Create winning shard info
        let winning_info = WinningShardInfo {
            shard_ref,
            data_submitter: shard_data.data_submitter,
            winning_encoder: report.winner.clone(),
            distance: report.distance,
            shard_created_epoch: shard_data.created_epoch,
        };

        // 6. Update target
        let mut updated_target_data = target_data.clone();
        updated_target_data.winning_shard = Some(winning_info);

        let mut updated_target = target_object.clone();
        updated_target
            .data
            .update_contents(bcs::to_bytes(&updated_target_data).map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to serialize target: {}",
                    e
                )))
            })?);
        store.mutate_input_object(updated_target);

        info!(
            "Updated target {} winning shard - encoder: {:?}, distance: {}",
            target_id, report.winner, report.distance
        );

        Ok(())
    }

    // =========================================================================
    // CLAIM ESCROW
    // =========================================================================
    fn execute_claim_escrow(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        shard_ref: ObjectRef,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let shard_id = shard_ref.0;

        // 1. Load shard object
        let shard_object =
            store
                .read_object(&shard_id)
                .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                    object_id: shard_id,
                })?;

        let shard_data =
            shard_object
                .as_shard()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: shard_id,
                    expected_type: ObjectType::Shard,
                    actual_type: shard_object.type_().clone(),
                })?;

        // 2. Load system state
        let mut state = self.load_system_state(store)?;
        let current_epoch = state.epoch;

        // 3. Verify timing - must be at created_epoch + 2 or later
        if current_epoch < shard_data.created_epoch + 2 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Cannot claim escrow yet. Current: {}, Claimable at: {}",
                    current_epoch,
                    shard_data.created_epoch + 2
                ),
            });
        }

        // 4. Calculate claim incentive for signer
        let claim_incentive = (shard_data.amount * CLAIM_INCENTIVE_BPS) / BPS_DENOMINATOR;
        let remaining_amount = shard_data.amount.saturating_sub(claim_incentive);

        // 5. Pay claim incentive to signer
        if claim_incentive > 0 {
            let incentive_coin = Object::new_coin(
                ObjectID::derive_id(tx_digest, store.next_creation_num()),
                claim_incentive,
                Owner::AddressOwner(signer),
                tx_digest,
            );
            store.create_object(incentive_coin);
        }

        // 6. Determine recipient and distribution type
        let (distribution, should_create_target) =
            self.determine_escrow_distribution(&shard_data, &state, remaining_amount)?;

        // 7. Execute distribution
        match distribution {
            EscrowDistribution::CoinToSubmitter { amount } => {
                let coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    amount,
                    Owner::AddressOwner(shard_data.data_submitter),
                    tx_digest,
                );
                store.create_object(coin);

                info!(
                    "ClaimEscrow: {} returned to data_submitter {}",
                    amount, shard_data.data_submitter
                );
            }
            EscrowDistribution::StakeToEncoder { encoder, amount } => {
                // Get encoder's address from public key
                let encoder_address = state.get_encoder_address(&encoder).ok_or_else(|| {
                    ExecutionFailureStatus::InvalidArguments {
                        reason: format!("Encoder {:?} address not found", encoder),
                    }
                })?;

                // Use the same staking flow as StakingExecutor
                // Encoder receives stake into their own pool
                let staked_soma = state
                    .request_add_stake_to_encoder(
                        encoder_address, // staker = encoder (receiving reward)
                        encoder_address, // pool = encoder's own pool
                        amount,
                    )
                    .map_err(|e| {
                        ExecutionFailureStatus::SomaError(SomaError::from(format!(
                            "Failed to add stake to encoder pool: {}",
                            e
                        )))
                    })?;

                // Create StakedSoma object owned by the encoder
                let staked_obj = Object::new_staked_soma_object(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    staked_soma,
                    Owner::AddressOwner(encoder_address),
                    tx_digest,
                );
                store.create_object(staked_obj);

                info!(
                    "ClaimEscrow: {} staked to encoder {:?} (address: {})",
                    amount, encoder, encoder_address
                );
            }
        }

        // 8. Maybe create target (only if winner was valid)
        if should_create_target {
            self.maybe_create_target(
                store,
                &shard_data,
                shard_ref,
                &mut state,
                current_epoch,
                tx_digest,
            )?;
        }

        // 9. Delete the shard object
        store.delete_input_object(&shard_id);

        // 10. Save updated system state
        self.save_system_state(store, state)?;

        Ok(())
    }

    // =========================================================================
    // DETERMINE ESCROW DISTRIBUTION
    // =========================================================================
    fn determine_escrow_distribution(
        &self,
        shard: &Shard,
        state: &SystemState,
        amount: u64,
    ) -> ExecutionResult<(EscrowDistribution, bool)> {
        match &shard.winning_encoder {
            None => {
                // No winner - return to data submitter
                Ok((EscrowDistribution::CoinToSubmitter { amount }, false))
            }
            Some(winner) => {
                // Check if winner has been tallied (slashed)
                if state.is_encoder_tallied(winner) {
                    // Winner was slashed - return to data submitter, no target creation
                    Ok((EscrowDistribution::CoinToSubmitter { amount }, false))
                } else {
                    // Valid winner - stake to encoder's pool
                    Ok((
                        EscrowDistribution::StakeToEncoder {
                            encoder: winner.clone(),
                            amount,
                        },
                        true, // May create target
                    ))
                }
            }
        }
    }

    // =========================================================================
    // MAYBE CREATE TARGET
    // =========================================================================
    // Target validity is determined by epoch check (created_epoch + 1), not a boolean
    // =========================================================================
    fn maybe_create_target(
        &self,
        store: &mut TemporaryStore,
        shard: &Shard,
        shard_ref: ObjectRef,
        state: &mut SystemState,
        current_epoch: EpochId,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Only shards with sampled embeddings can become targets
        let sampled_embedding_ref = match &shard.sampled_embedding {
            Some(e) => e.clone(),
            None => {
                debug!("Shard has no sampled embedding, skipping target creation");
                return Ok(());
            }
        };

        // Get the epoch seed for deterministic randomness
        // Seed is from the epoch transition at created_epoch + 1 -> created_epoch + 2
        let seed_epoch = shard.created_epoch + 2;
        let epoch_seed = state.get_epoch_seed(seed_epoch).ok_or_else(|| {
            ExecutionFailureStatus::InvalidArguments {
                reason: format!("Epoch seed not found for epoch {}", seed_epoch),
            }
        })?;

        // Get target selection rate from protocol params
        let selection_rate_bps = state.parameters.target_selection_rate_bps;

        // Generate deterministic random value
        let random_value = self.derive_random_value(&epoch_seed, &shard_ref);

        // Check against threshold
        // random_value % BPS_DENOMINATOR < selection_rate_bps means selected
        if random_value % BPS_DENOMINATOR >= selection_rate_bps {
            debug!(
                "Shard {} not selected as target (random: {}, threshold: {})",
                shard_ref.0,
                random_value % BPS_DENOMINATOR,
                selection_rate_bps
            );
            return Ok(());
        }

        // Create target object as shared
        // Validity is determined by epoch check: valid when current_epoch >= created_epoch + 1
        let target_id = ObjectID::derive_id(tx_digest, store.next_creation_num());
        let target = Object::new_target(
            target_id,
            None,          // Network-created target (no creator)
            current_epoch, // created_epoch
            sampled_embedding_ref,
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            tx_digest,
        );
        store.create_object(target);

        // Increment target counter for this epoch in system state
        state.increment_target_count(current_epoch);

        info!(
            "Created new target {} from shard {} in epoch {} (valid at epoch {})",
            target_id,
            shard_ref.0,
            current_epoch,
            current_epoch + 1
        );

        Ok(())
    }

    /// Derives a deterministic random value from epoch seed and shard reference
    fn derive_random_value(&self, epoch_seed: &[u8], shard_ref: &ObjectRef) -> u64 {
        use fastcrypto::hash::HashFunction;

        let mut hasher = types::crypto::DefaultHash::default();
        hasher.update(epoch_seed);
        hasher.update(shard_ref.0.as_ref());
        hasher.update(&shard_ref.1.value().to_le_bytes());
        hasher.update(shard_ref.2);

        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash.as_ref()[0..8].try_into().unwrap();
        u64::from_le_bytes(bytes)
    }

    // =========================================================================
    // CLAIM REWARD
    // =========================================================================
    fn execute_claim_reward(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        target_ref: ObjectRef,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let target_id = target_ref.0;

        // 1. Load target object
        let target_object = store.read_object(&target_id).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound {
                object_id: target_id,
            }
        })?;

        let target_data =
            target_object
                .as_target()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: target_id,
                    expected_type: ObjectType::Target,
                    actual_type: target_object.type_().clone(),
                })?;

        // 2. Load system state
        let mut state = self.load_system_state(store)?;
        let current_epoch = state.epoch;

        // 3. Verify timing - must be at created_epoch + 2 or later
        if current_epoch < target_data.created_epoch + 2 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Cannot claim reward yet. Current: {}, Claimable at: {}",
                    current_epoch,
                    target_data.created_epoch + 2
                ),
            });
        }

        // Determine reward amount based on origin
        let reward_amount = match &target_data.origin {
            TargetOrigin::System => {
                let reward_epoch = target_data.created_epoch + 1;
                state.get_target_reward(reward_epoch).ok_or_else(|| {
                    ExecutionFailureStatus::InvalidArguments {
                        reason: format!("Target reward not found for epoch {}", reward_epoch),
                    }
                })?
            }
            TargetOrigin::User { reward_amount, .. } => *reward_amount,
        };

        // Calculate claim incentive
        let claim_incentive = (reward_amount * CLAIM_INCENTIVE_BPS) / BPS_DENOMINATOR;
        let remaining_amount = reward_amount.saturating_sub(claim_incentive);

        // Pay claim incentive
        if claim_incentive > 0 {
            let incentive_coin = Object::new_coin(
                ObjectID::derive_id(tx_digest, store.next_creation_num()),
                claim_incentive,
                Owner::AddressOwner(signer),
                tx_digest,
            );
            store.create_object(incentive_coin);
        }

        // Determine recipient
        let recipient = self.determine_reward_recipient(&target_data, &state)?;

        // Distribute
        match recipient {
            RewardRecipient::Address(addr) => {
                let coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    remaining_amount,
                    Owner::AddressOwner(addr),
                    tx_digest,
                );
                store.create_object(coin);
            }
            RewardRecipient::EmissionsPool => {
                // Only happens for system targets with no winner
                state.return_to_emissions_pool(remaining_amount);
            }
        }

        // Delete target object
        store.delete_input_object(&target_id);

        // Save updated system state
        self.save_system_state(store, state)?;

        Ok(())
    }

    // =========================================================================
    // DETERMINE REWARD RECIPIENT
    // =========================================================================
    fn determine_reward_recipient(
        &self,
        target: &Target,
        state: &SystemState,
    ) -> ExecutionResult<RewardRecipient> {
        match &target.winning_shard {
            Some(winner_info) if !state.is_encoder_tallied(&winner_info.winning_encoder) => {
                // Valid winner - reward goes to data submitter
                Ok(RewardRecipient::Address(winner_info.data_submitter))
            }
            _ => {
                // No winner or winner slashed - return based on origin
                match &target.origin {
                    TargetOrigin::User { creator, .. } => Ok(RewardRecipient::Address(*creator)),
                    TargetOrigin::System => Ok(RewardRecipient::EmissionsPool),
                }
            }
        }
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    fn load_system_state(&self, store: &TemporaryStore) -> ExecutionResult<SystemState> {
        let state_object = store.read_object(&SYSTEM_STATE_OBJECT_ID).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            }
        })?;

        bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents()).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to deserialize system state: {}",
                e
            )))
        })
    }

    fn save_system_state(
        &self,
        store: &mut TemporaryStore,
        state: SystemState,
    ) -> ExecutionResult<()> {
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize system state: {}",
                e
            )))
        })?;

        let mut updated_state = state_object;
        updated_state.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state);

        Ok(())
    }

    fn check_ownership(&self, object: &Object, expected_owner: SomaAddress) -> ExecutionResult<()> {
        match object.owner() {
            Owner::AddressOwner(owner) if *owner == expected_owner => Ok(()),
            Owner::AddressOwner(owner) => Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: object.id(),
                expected_owner,
                actual_owner: Some(*owner),
            }),
            Owner::Shared { .. } => Ok(()),
            Owner::Immutable => Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Cannot modify immutable object".to_string(),
            }),
        }
    }
}

// =========================================================================
// HELPER TYPES
// =========================================================================

enum EscrowDistribution {
    CoinToSubmitter {
        amount: u64,
    },
    StakeToEncoder {
        encoder: EncoderPublicKey,
        amount: u64,
    },
}

enum RewardRecipient {
    Address(SomaAddress),
    EmissionsPool,
}

// =========================================================================
// TRAIT IMPLEMENTATIONS
// =========================================================================

impl TransactionExecutor for ShardExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::EmbedData {
                download_metadata,
                coin_ref,
                target_ref,
            } => self.execute_embed_data(
                store,
                signer,
                download_metadata,
                coin_ref,
                target_ref,
                tx_digest,
                value_fee,
            ),

            TransactionKind::ClaimEscrow { shard_ref } => {
                self.execute_claim_escrow(store, signer, shard_ref, tx_digest)
            }

            TransactionKind::ReportWinner {
                shard_ref,
                target_ref,
                report,
                signature,
                signers,
                shard_auth_token,
            } => self.execute_report_winner(
                store,
                signer,
                shard_ref,
                target_ref,
                report,
                signature,
                signers,
                shard_auth_token,
                tx_digest,
            ),

            TransactionKind::ClaimReward { target_ref } => {
                self.execute_claim_reward(store, signer, target_ref, tx_digest)
            }

            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for ShardExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        match kind {
            TransactionKind::EmbedData {
                download_metadata, ..
            } => {
                if let Ok(state) = self.load_system_state(store) {
                    let byte_price = state.encoders.reference_byte_price;
                    let embed_cost =
                        byte_price.saturating_mul(download_metadata.metadata().size() as u64);
                    let fee = (embed_cost * 5) / 10000;
                    std::cmp::max(fee, self.base_fee())
                } else {
                    self.base_fee()
                }
            }

            TransactionKind::ClaimEscrow { shard_ref } => {
                if let Some(shard_obj) = store.read_object(&shard_ref.0) {
                    if let Some(shard) = shard_obj.as_shard() {
                        let fee = (shard.amount * 5) / 10000;
                        return std::cmp::max(fee, self.base_fee());
                    }
                }
                self.base_fee()
            }

            TransactionKind::ReportWinner { .. } => self.base_fee(),

            TransactionKind::ClaimReward { target_ref } => {
                if let Ok(state) = self.load_system_state(store) {
                    if let Some(target_obj) = store.read_object(&target_ref.0) {
                        if let Some(target) = target_obj.as_target() {
                            let reward_epoch = target.created_epoch + 1;
                            if let Some(reward) = state.get_target_reward(reward_epoch) {
                                let fee = (reward * 5) / 10000;
                                return std::cmp::max(fee, self.base_fee());
                            }
                        }
                    }
                }
                self.base_fee()
            }

            _ => 0,
        }
    }

    fn base_fee(&self) -> u64 {
        1500
    }

    fn write_fee_per_object(&self) -> u64 {
        300
    }
}
