//! Submission executor for data submissions to targets.
//!
//! Handles `SubmitData` and `ClaimRewards` transactions:
//! - `SubmitData`: Validate submission, fill target, record hit, spawn replacement
//! - `ClaimRewards`: Check challenge window, distribute rewards, return bond

use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    committee::EpochId,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    metadata::{ManifestAPI, MetadataAPI},
    object::{Object, ObjectID, ObjectType, Owner},
    submission::Submission,
    system_state::SystemState,
    target::{Target, TargetStatus, generate_target, make_target_seed},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use tracing::info;

use crate::execution::BPS_DENOMINATOR;

use super::{FeeCalculator, TransactionExecutor, object::check_ownership};

pub struct SubmissionExecutor;

impl SubmissionExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Deserialize SystemState from the temporary store.
    fn load_system_state(store: &TemporaryStore) -> ExecutionResult<(Object, SystemState)> {
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        let state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        Ok((state_object, state))
    }

    /// Serialize and write back the updated SystemState.
    fn save_system_state(
        store: &mut TemporaryStore,
        state_object: Object,
        state: &SystemState,
    ) -> ExecutionResult<()> {
        let state_bytes = bcs::to_bytes(state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated = state_object;
        updated.data.update_contents(state_bytes);
        store.mutate_input_object(updated);
        Ok(())
    }

    /// Load a Target from the temporary store.
    fn load_target(store: &TemporaryStore, target_id: &ObjectID) -> ExecutionResult<(Object, Target)> {
        let target_object = store
            .read_object(target_id)
            .ok_or(ExecutionFailureStatus::TargetNotFound)?
            .clone();

        let target = bcs::from_bytes::<Target>(target_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize target: {}",
                    e
                )))
            })?;

        Ok((target_object, target))
    }

    /// Save an updated Target back to the store.
    fn save_target(
        store: &mut TemporaryStore,
        target_object: Object,
        target: &Target,
    ) -> ExecutionResult<()> {
        let target_bytes = bcs::to_bytes(target).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated target: {}",
                e
            )))
        })?;

        let mut updated = target_object;
        updated.data.update_contents(target_bytes);
        store.mutate_input_object(updated);
        Ok(())
    }

    /// Execute SubmitData: validate submission, fill target, update hit counter, spawn replacement.
    fn execute_submit_data(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        _value_fee: u64,
    ) -> ExecutionResult<()> {
        let TransactionKind::SubmitData(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        info!(
            "SubmitData: tx_digest={:?}, target_id={:?}, model_id={:?}",
            tx_digest, args.target_id, args.model_id
        );

        // Load system state and target
        let (state_object, mut state) = Self::load_system_state(store)?;
        let (target_object, mut target) = Self::load_target(store, &args.target_id)?;

        info!(
            "SubmitData: loaded state epoch={}, emission_pool={}, target_status={:?}",
            state.epoch, state.emission_pool.balance, target.status
        );

        // Get current epoch from system state
        let current_epoch = state.epoch;

        // 1. Validate target is open
        if !target.is_open() {
            return Err(ExecutionFailureStatus::TargetNotOpen);
        }

        // 2. Validate target hasn't expired (must be in same epoch as generation)
        if current_epoch > target.generation_epoch {
            return Err(ExecutionFailureStatus::TargetExpired {
                generation_epoch: target.generation_epoch,
                current_epoch,
            });
        }

        // 3. Validate model is in target's model_ids
        if !target.model_ids.contains(&args.model_id) {
            return Err(ExecutionFailureStatus::ModelNotInTarget {
                model_id: args.model_id,
                target_id: args.target_id,
            });
        }

        // 4. Validate embedding dimension
        let expected_dim = target.embedding.len() as u64;
        let actual_dim = args.embedding.len() as u64;
        if actual_dim != expected_dim {
            return Err(ExecutionFailureStatus::EmbeddingDimensionMismatch {
                expected: expected_dim,
                actual: actual_dim,
            });
        }

        // 5. Validate distance score
        if args.distance_score > target.distance_threshold {
            return Err(ExecutionFailureStatus::DistanceExceedsThreshold {
                score: args.distance_score,
                threshold: target.distance_threshold,
            });
        }

        // 6. Validate reconstruction score
        if args.reconstruction_score > target.reconstruction_threshold {
            return Err(ExecutionFailureStatus::ReconstructionExceedsThreshold {
                score: args.reconstruction_score,
                threshold: target.reconstruction_threshold,
            });
        }

        // 7. Calculate required bond from protocol config and validate
        let data_size = args.data_manifest.manifest.metadata().size() as u64;
        let bond_per_byte = state.parameters.submission_bond_per_byte;
        let required_bond = data_size * bond_per_byte;

        // Get and validate bond coin
        let bond_coin_id = args.bond_coin.0;
        let bond_object = store
            .read_object(&bond_coin_id)
            .ok_or(ExecutionFailureStatus::ObjectNotFound { object_id: bond_coin_id })?;

        check_ownership(&bond_object, signer)?;

        let bond_balance = bond_object
            .as_coin()
            .ok_or(ExecutionFailureStatus::InvalidObjectType {
                object_id: bond_coin_id,
                expected_type: ObjectType::Coin,
                actual_type: bond_object.type_().clone(),
            })?;

        if bond_balance < required_bond {
            return Err(ExecutionFailureStatus::InsufficientBond {
                required: required_bond,
                provided: bond_balance,
            });
        }

        // 8. Consume bond coin (deduct bond amount)
        let is_gas_coin = store.gas_object_id == Some(bond_coin_id);
        if bond_balance == required_bond && !is_gas_coin {
            // Exact amount and not gas coin - delete
            store.delete_input_object(&bond_coin_id);
        } else {
            // More than needed or is gas coin - update balance
            let remaining = bond_balance - required_bond;
            let mut updated_bond = bond_object.clone();
            updated_bond.update_coin_balance(remaining);
            store.mutate_input_object(updated_bond);
        }

        // 9. Create submission object
        let submission = Submission::new(
            signer,
            args.data_commitment,
            args.data_manifest,
            args.model_id,
            args.embedding,
            args.distance_score,
            args.reconstruction_score,
            required_bond,
            current_epoch,
        );

        // 10. Update target status to Filled and record miner/model/bond for rewards
        target.status = TargetStatus::Filled { fill_epoch: current_epoch };
        target.miner = Some(signer);
        target.winning_model_id = Some(args.model_id);
        // Capture model owner at fill time - model must be active
        let model = state
            .model_registry
            .active_models
            .get(&args.model_id)
            .ok_or(ExecutionFailureStatus::ModelNotActive)?;
        target.winning_model_owner = Some(model.owner);
        target.bond_amount = required_bond; // Store bond on target for refund/forfeit

        // 11. Record hit in target_state (for difficulty adjustment at epoch boundary)
        state.target_state.record_hit();

        // 12. Create submission object that holds the submission data
        // Submission is stored as a shared object associated with the target
        let submission_creation_num = store.next_creation_num();
        let submission_id = ObjectID::derive_id(tx_digest, submission_creation_num);
        info!(
            "SubmitData: creating submission object with id={:?}, creation_num={}",
            submission_id, submission_creation_num
        );
        let submission_object = Object::new_submission_object(
            submission_id,
            submission,
            tx_digest,
        );
        store.create_object(submission_object);

        // 13. Spawn replacement target if there are active models and emission pool has funds
        let reward_per_target = state.target_state.reward_per_target;
        if !state.model_registry.active_models.is_empty()
            && state.emission_pool.balance >= reward_per_target
        {
            // Deduct reward from emission pool for the new target
            state.emission_pool.balance -= reward_per_target;

            let seed_creation_num = store.next_creation_num();
            let seed = make_target_seed(&tx_digest, seed_creation_num);
            info!(
                "SubmitData: generating replacement target with seed={}, seed_creation_num={}",
                seed, seed_creation_num
            );

            // Get target parameters from system state
            let models_per_target = state.parameters.target_models_per_target;
            let embedding_dim = state.parameters.target_embedding_dim;

            let new_target = generate_target(
                seed,
                &state.model_registry,
                &state.target_state,
                models_per_target,
                embedding_dim,
                current_epoch,
            )?;

            // Record that a new target was generated (for difficulty adjustment)
            state.target_state.record_target_generated();

            // Create replacement target as shared object
            let target_creation_num = store.next_creation_num();
            let new_target_id = ObjectID::derive_id(tx_digest, target_creation_num);
            info!(
                "SubmitData: creating replacement target with id={:?}, creation_num={}, embedding[0..3]={:?}",
                new_target_id, target_creation_num, &new_target.embedding.as_slice().unwrap()[0..3]
            );
            let new_target_object = Object::new_target_object(
                new_target_id,
                new_target,
                tx_digest,
            );
            store.create_object(new_target_object);
        }

        // 14. Save updated state and target
        Self::save_target(store, target_object, &target)?;
        Self::save_system_state(store, state_object, &state)?;

        Ok(())
    }

    /// Execute ClaimRewards: check challenge window, distribute rewards, return bond.
    ///
    /// Handles three cases:
    /// 1. **Filled target (no challenge)**: After challenge window closes, distribute rewards
    ///    to miner/model/claimer and return bond to miner.
    /// 2. **Filled target (successful challenge)**: When challenges are implemented, this will
    ///    forfeit the bond to the emission pool and return rewards to emission pool.
    /// 3. **Expired unfilled target**: Return reward pool to emissions, pay claimer incentive.
    fn execute_claim_rewards(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let TransactionKind::ClaimRewards(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        // Load system state and target
        let (state_object, mut state) = Self::load_system_state(store)?;
        let (target_object, mut target) = Self::load_target(store, &args.target_id)?;

        // Get current epoch from system state
        let current_epoch = state.epoch;

        match target.status {
            TargetStatus::Claimed => {
                return Err(ExecutionFailureStatus::TargetAlreadyClaimed);
            }
            TargetStatus::Filled { fill_epoch } => {
                // Handle filled target claim
                self.claim_filled_target(
                    store,
                    &mut state,
                    &mut target,
                    signer,
                    tx_digest,
                    fill_epoch,
                    current_epoch,
                )?;
            }
            TargetStatus::Open => {
                // Handle expired unfilled target claim
                self.claim_expired_target(
                    store,
                    &mut state,
                    &mut target,
                    signer,
                    tx_digest,
                    current_epoch,
                )?;
            }
        }

        // Save updated target and state
        Self::save_target(store, target_object, &target)?;
        Self::save_system_state(store, state_object, &state)?;

        Ok(())
    }

    /// Claim rewards from a filled target after the challenge window closes.
    fn claim_filled_target(
        &self,
        store: &mut TemporaryStore,
        state: &mut SystemState,
        target: &mut Target,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
        fill_epoch: EpochId,
        current_epoch: EpochId,
    ) -> ExecutionResult<()> {
        // Validate challenge window is closed
        // Challenge window = fill_epoch + 1
        // Can claim when current_epoch > fill_epoch + 1
        let challenge_window_end = fill_epoch + 1;
        if current_epoch <= challenge_window_end {
            return Err(ExecutionFailureStatus::ChallengeWindowOpen {
                fill_epoch,
                current_epoch,
            });
        }

        // Get reward amount and recipient info
        let reward = target.reward_pool;
        let bond = target.bond_amount;
        let miner = target.miner.ok_or(ExecutionFailureStatus::TargetNotFilled)?;
        // Model owner was captured at fill time, so rewards work even if model is now inactive
        let model_owner = target.winning_model_owner;

        // Mark target as claimed
        target.status = TargetStatus::Claimed;

        // TODO: When challenges are implemented, check if there was a successful challenge.
        // If challenged successfully:
        //   - Forfeit bond to emission pool: state.emission_pool.balance += bond
        //   - Return reward to emission pool: state.emission_pool.balance += reward
        //   - Pay claimer incentive only
        //   - Return early
        let _challenge_succeeded = false; // Placeholder for future challenge system

        // No successful challenge - distribute rewards normally
        // Full reward goes to miner, model owner, and claimer (100% distributed)
        if reward > 0 {
            let params = &state.parameters;
            let miner_share = (reward * params.target_miner_reward_share_bps) / BPS_DENOMINATOR;
            let model_share = (reward * params.target_model_reward_share_bps) / BPS_DENOMINATOR;
            let claimer_share = (reward * params.target_claimer_incentive_bps) / BPS_DENOMINATOR;
            // Remainder after rounding goes to miner (ensures 100% distribution)
            let remainder = reward - miner_share - model_share - claimer_share;
            let miner_total = miner_share + remainder;

            // Miner reward (includes any rounding remainder)
            if miner_total > 0 {
                let miner_coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    miner_total,
                    Owner::AddressOwner(miner),
                    tx_digest,
                );
                store.create_object(miner_coin);
            }

            // Model owner reward (captured at fill time)
            if model_share > 0 {
                if let Some(owner) = model_owner {
                    let model_coin = Object::new_coin(
                        ObjectID::derive_id(tx_digest, store.next_creation_num()),
                        model_share,
                        Owner::AddressOwner(owner),
                        tx_digest,
                    );
                    store.create_object(model_coin);
                }
                // Note: If model_owner is None (shouldn't happen), model share is not distributed
            }

            // Claimer incentive (incentivizes anyone to call ClaimRewards)
            if claimer_share > 0 {
                let claimer_coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    claimer_share,
                    Owner::AddressOwner(signer),
                    tx_digest,
                );
                store.create_object(claimer_coin);
            }
        }

        // Return bond to miner (no challenge succeeded)
        if bond > 0 {
            let bond_return_coin = Object::new_coin(
                ObjectID::derive_id(tx_digest, store.next_creation_num()),
                bond,
                Owner::AddressOwner(miner),
                tx_digest,
            );
            store.create_object(bond_return_coin);
        }

        Ok(())
    }

    /// Claim an expired unfilled target - return reward pool to emissions.
    fn claim_expired_target(
        &self,
        store: &mut TemporaryStore,
        state: &mut SystemState,
        target: &mut Target,
        signer: SomaAddress,
        tx_digest: TransactionDigest,
        current_epoch: EpochId,
    ) -> ExecutionResult<()> {
        // Target must be expired (generation_epoch < current_epoch) to be claimable when Open
        if current_epoch <= target.generation_epoch {
            return Err(ExecutionFailureStatus::TargetNotFilled);
        }

        let reward = target.reward_pool;

        // Mark target as claimed
        target.status = TargetStatus::Claimed;

        // Return most of the reward pool to emissions, pay claimer incentive
        if reward > 0 {
            let claimer_share = (reward * state.parameters.target_claimer_incentive_bps) / BPS_DENOMINATOR;
            let return_to_pool = reward - claimer_share;

            // Return to emission pool
            state.emission_pool.balance += return_to_pool;

            // Pay claimer incentive for cleanup
            if claimer_share > 0 {
                let claimer_coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    claimer_share,
                    Owner::AddressOwner(signer),
                    tx_digest,
                );
                store.create_object(claimer_coin);
            }
        }

        // Note: No bond to handle since target was never filled

        Ok(())
    }
}

impl TransactionExecutor for SubmissionExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::SubmitData(_) => {
                self.execute_submit_data(store, signer, kind, tx_digest, value_fee)
            }
            TransactionKind::ClaimRewards(_) => {
                self.execute_claim_rewards(store, signer, kind, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for SubmissionExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        // Value fee based on target reward being claimed
        // Using halved fee rate similar to staking transactions
        let value_fee_bps = store.fee_parameters.value_fee_bps / 2;

        match kind {
            TransactionKind::ClaimRewards(args) => {
                // Value fee on the reward amount
                // Load target to get reward_pool - if not found, return 0
                let target = match store.read_object(&args.target_id) {
                    Some(obj) => {
                        match bcs::from_bytes::<Target>(obj.as_inner().data.contents()) {
                            Ok(t) => t,
                            Err(_) => return 0,
                        }
                    }
                    None => return 0,
                };

                let reward = target.reward_pool;
                if reward == 0 {
                    return 0;
                }

                (reward * value_fee_bps) / BPS_DENOMINATOR
            }
            // SubmitData has no value fee (bond is separate and refundable)
            _ => 0,
        }
    }
}
