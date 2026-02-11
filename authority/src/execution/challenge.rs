//! Challenge executor for dispute resolution.
//!
//! Handles challenge transactions using the tally-based approach.
//!
//! ## InitiateChallenge
//! - Validate target is filled and in challenge window (same epoch as fill)
//! - Calculate and lock challenger bond (challenger_bond_per_byte * data_size)
//! - Create Challenge shared object
//! - Set challenger and challenge_id on the Target object
//!
//! ## ReportChallenge
//! - Validators submit reports indicating the challenger is wrong (submission is valid)
//! - Reports accumulate on the Challenge object
//!
//! ## ClaimChallengeBond
//! - After challenge window closes, distribute challenger's bond based on report quorum
//! - 2f+1 reports: challenger loses, bond → reporting validators
//! - No quorum: challenger wins (benefit of doubt), bond returned

use tracing::info;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    challenge::{
        Challenge, ChallengeStatus,
    },
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    metadata::{ManifestAPI, MetadataAPI},
    object::{Object, ObjectID, ObjectType, Owner},
    system_state::SystemState,
    target::{Target, TargetStatus},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
};

use super::{FeeCalculator, TransactionExecutor, object::check_ownership};

pub struct ChallengeExecutor;

impl ChallengeExecutor {
    pub fn new() -> Self {
        Self
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
    fn load_target(
        store: &TemporaryStore,
        target_id: &ObjectID,
    ) -> ExecutionResult<(Object, Target)> {
        let target_object = store
            .read_object(target_id)
            .ok_or(ExecutionFailureStatus::TargetNotFound)?
            .clone();

        let target = bcs::from_bytes::<Target>(target_object.as_inner().data.contents()).map_err(
            |e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize target: {}",
                    e
                )))
            },
        )?;

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

    /// Load a Challenge from the temporary store.
    fn load_challenge(
        store: &TemporaryStore,
        challenge_id: &ObjectID,
    ) -> ExecutionResult<(Object, Challenge)> {
        let challenge_object = store.read_object(challenge_id).ok_or_else(|| {
            ExecutionFailureStatus::ChallengeNotFound {
                challenge_id: *challenge_id,
            }
        })?;

        let challenge =
            bcs::from_bytes::<Challenge>(challenge_object.as_inner().data.contents()).map_err(
                |e| {
                    ExecutionFailureStatus::SomaError(SomaError::from(format!(
                        "Failed to deserialize challenge: {}",
                        e
                    )))
                },
            )?;

        Ok((challenge_object.clone(), challenge))
    }

    /// Save an updated Challenge back to the store.
    fn save_challenge(
        store: &mut TemporaryStore,
        challenge_object: Object,
        challenge: &Challenge,
    ) -> ExecutionResult<()> {
        let challenge_bytes = bcs::to_bytes(challenge).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated challenge: {}",
                e
            )))
        })?;

        let mut updated = challenge_object;
        updated.data.update_contents(challenge_bytes);
        store.mutate_input_object(updated);
        Ok(())
    }

    /// Execute InitiateChallenge: validate target is filled, lock bond, create fraud challenge.
    ///
    /// All challenges are fraud challenges (availability issues handled via submission reports).
    fn execute_initiate_challenge(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let TransactionKind::InitiateChallenge(args) = kind else {
            return Err(ExecutionFailureStatus::InvalidTransactionType);
        };

        info!(
            "InitiateChallenge: tx_digest={:?}, target_id={:?}",
            tx_digest, args.target_id
        );

        // Load system state for epoch and protocol config
        let (state_object, state) = Self::load_system_state(store)?;
        let current_epoch = state.epoch;

        // Load target
        let (target_object, mut target) = Self::load_target(store, &args.target_id)?;

        info!(
            "InitiateChallenge: loaded target status={:?}, generation_epoch={}",
            target.status, target.generation_epoch
        );

        // 1. Validate target is filled and get fill epoch
        let fill_epoch = match target.status {
            TargetStatus::Filled { fill_epoch } => fill_epoch,
            TargetStatus::Open => return Err(ExecutionFailureStatus::TargetNotFilled),
            TargetStatus::Claimed => return Err(ExecutionFailureStatus::TargetAlreadyClaimed),
        };

        // 2. Validate challenge window is open (same epoch as fill)
        // Challenge window = fill_epoch only
        if current_epoch != fill_epoch {
            return Err(ExecutionFailureStatus::ChallengeWindowClosed {
                fill_epoch,
                current_epoch,
            });
        }

        // 2a. Validate target doesn't already have a challenger (first one wins)
        if target.challenger.is_some() {
            return Err(ExecutionFailureStatus::ChallengeAlreadyExists);
        }

        // 3. Calculate and lock challenger bond (challenger_bond_per_byte * data_size)
        // Retrieve data size from target's winning submission manifest
        let data_size = target
            .winning_data_manifest
            .as_ref()
            .ok_or_else(|| {
                ExecutionFailureStatus::SomaError(SomaError::from(
                    "Target missing winning_data_manifest for challenge",
                ))
            })?
            .manifest
            .metadata()
            .size() as u64;

        let bond_per_byte = state.parameters.challenger_bond_per_byte;
        let required_bond = data_size * bond_per_byte;

        info!(
            "InitiateChallenge: data_size={}, bond_per_byte={}, required_bond={}",
            data_size, bond_per_byte, required_bond
        );

        // Get and validate bond coin
        let bond_coin_id = args.bond_coin.0;
        let bond_object = store
            .read_object(&bond_coin_id)
            .ok_or(ExecutionFailureStatus::ObjectNotFound {
                object_id: bond_coin_id,
            })?;

        check_ownership(&bond_object, signer)?;

        let bond_balance = bond_object.as_coin().ok_or_else(|| {
            ExecutionFailureStatus::InvalidObjectType {
                object_id: bond_coin_id,
                expected_type: ObjectType::Coin,
                actual_type: bond_object.type_().clone(),
            }
        })?;

        if bond_balance < required_bond {
            return Err(ExecutionFailureStatus::InsufficientChallengerBond {
                required: required_bond,
                provided: bond_balance,
            });
        }

        // 5. Consume bond coin (deduct bond amount)
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

        // 6. Extract audit data from target (so Challenge is self-contained for auditing)
        let winning_model_id = target.winning_model_id.ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Target missing winning_model_id for challenge",
            ))
        })?;
        let winning_data_manifest = target.winning_data_manifest.clone().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Target missing winning_data_manifest for challenge",
            ))
        })?;
        let winning_data_commitment = target.winning_data_commitment.clone().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Target missing winning_data_commitment for challenge",
            ))
        })?;
        let winning_embedding = target.winning_embedding.clone().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Target missing winning_embedding for challenge",
            ))
        })?;
        let winning_distance_score = target.winning_distance_score.clone().ok_or_else(|| {
            ExecutionFailureStatus::SomaError(SomaError::from(
                "Target missing winning_distance_score for challenge",
            ))
        })?;

        // 7. Create Challenge object
        // Derive challenge ID from tx_digest first (ignore client-provided challenge_id)
        let challenge_creation_num = store.next_creation_num();
        let challenge_id = ObjectID::derive_id(tx_digest, challenge_creation_num);

        info!(
            "InitiateChallenge: creating challenge with id={:?}, creation_num={}",
            challenge_id, challenge_creation_num
        );

        let challenge = Challenge::new(
            challenge_id,
            args.target_id,
            signer,
            required_bond,
            current_epoch,
            // Audit data from target
            target.model_ids.clone(),
            target.embedding.clone(),
            target.distance_threshold.clone(),
            winning_model_id,
            winning_data_manifest,
            winning_data_commitment,
            winning_embedding,
            winning_distance_score,
        );

        let challenge_object = Object::new_challenge_object(challenge_id, challenge, tx_digest);
        store.create_object(challenge_object);

        // 8. Update target with challenger and challenge_id (for tally-based reports)
        target.challenger = Some(signer);
        target.challenge_id = Some(challenge_id);

        // Save updated target
        Self::save_target(store, target_object, &target)?;

        // Save system state (no changes needed, but ensures consistency)
        Self::save_system_state(store, state_object, &state)?;

        Ok(())
    }

    // =========================================================================
    // Tally-based challenge transaction handlers
    // =========================================================================

    /// Execute ReportChallenge: Record that a validator finds the challenger wrong.
    ///
    /// When a validator audits the challenge and determines the submission is valid
    /// (challenger is wrong), they submit this report. Reports accumulate on the
    /// Challenge object. If 2f+1 validators report, the challenger loses at ClaimChallengeBond.
    fn execute_report_challenge(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        challenge_id: ObjectID,
    ) -> ExecutionResult<()> {
        // Load system state to validate signer is active validator
        let (_, state) = Self::load_system_state(store)?;

        if !state.validators.is_active_validator(signer) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Load challenge
        let (challenge_object, mut challenge) = Self::load_challenge(store, &challenge_id)?;

        // Validate challenge is pending
        if !challenge.is_pending() {
            return Err(ExecutionFailureStatus::ChallengeNotPending { challenge_id });
        }

        // Validate challenge is from current epoch (not expired)
        if challenge.challenge_epoch != state.epoch {
            return Err(ExecutionFailureStatus::ChallengeExpired {
                challenge_epoch: challenge.challenge_epoch,
                current_epoch: state.epoch,
            });
        }

        // Record the report on the Challenge object (validator says challenger is wrong)
        challenge.report_challenge(signer);

        info!(
            "ReportChallenge: validator {:?} reported challenge {:?} (challenger is wrong)",
            signer, challenge_id
        );

        // Save updated challenge
        Self::save_challenge(store, challenge_object, &challenge)?;

        Ok(())
    }

    /// Execute UndoReportChallenge: Remove a validator's verdict from a challenge.
    ///
    /// Validation:
    /// - Signer must be an active validator
    /// - Challenge must be in Pending status
    /// - Challenge must be from current epoch (not expired)
    fn execute_undo_report_challenge(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        challenge_id: ObjectID,
    ) -> ExecutionResult<()> {
        // Load system state to validate signer is active validator
        let (_, state) = Self::load_system_state(store)?;

        if !state.validators.is_active_validator(signer) {
            return Err(ExecutionFailureStatus::NotAValidator);
        }

        // Load challenge
        let (challenge_object, mut challenge) = Self::load_challenge(store, &challenge_id)?;

        // Validate challenge is pending
        if !challenge.is_pending() {
            return Err(ExecutionFailureStatus::ChallengeNotPending { challenge_id });
        }

        // Validate challenge is from current epoch (not expired)
        if challenge.challenge_epoch != state.epoch {
            return Err(ExecutionFailureStatus::ChallengeExpired {
                challenge_epoch: challenge.challenge_epoch,
                current_epoch: state.epoch,
            });
        }

        // Remove the report
        if !challenge.undo_report_challenge(signer) {
            return Err(ExecutionFailureStatus::ReportRecordNotFound);
        }

        info!(
            "UndoReportChallenge: validator {:?} removed report for challenge {:?}",
            signer, challenge_id
        );

        // Save updated challenge
        Self::save_challenge(store, challenge_object, &challenge)?;

        Ok(())
    }

    /// Execute ClaimChallengeBond: Resolve the challenge based on validator reports.
    ///
    /// Called after the challenge window closes (epoch after challenge_epoch).
    /// Distributes the challenger's bond based on whether 2f+1 validators reported:
    /// - 2f+1 reports: challenger loses, bond → reporting validators
    /// - No quorum: challenger wins (benefit of doubt), bond returned
    ///
    /// Note: The miner's bond and target rewards are handled separately by ClaimRewards
    /// based on submission reports (ReportSubmission).
    fn execute_claim_challenge_bond(
        &self,
        store: &mut TemporaryStore,
        _signer: SomaAddress,
        challenge_id: ObjectID,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // Load system state
        let (state_object, mut state) = Self::load_system_state(store)?;
        let current_epoch = state.epoch;

        // Load challenge
        let (challenge_object, mut challenge) = Self::load_challenge(store, &challenge_id)?;

        // Validate challenge is pending (not already resolved)
        if !challenge.is_pending() {
            return Err(ExecutionFailureStatus::ChallengeNotPending { challenge_id });
        }

        // Validate challenge window is closed (current_epoch > challenge_epoch)
        if current_epoch <= challenge.challenge_epoch {
            return Err(ExecutionFailureStatus::ChallengeWindowOpen {
                fill_epoch: challenge.challenge_epoch,
                current_epoch,
            });
        }

        // Check tally-based quorum on Challenge object
        let (has_quorum, reporting_validators) =
            challenge.get_challenge_report_quorum(&state.validators);

        info!(
            "ClaimChallengeBond: challenge {:?}, has_quorum={}, reporters={:?}",
            challenge_id, has_quorum, reporting_validators
        );

        // Clear challenge reports
        challenge.clear_challenge_reports();

        if has_quorum {
            // 2f+1 validators reported against challenger - challenger loses
            info!("ClaimChallengeBond: challenger loses (2f+1 validators reported against)");

            // Challenger bond → reporting validators
            Self::distribute_bond_to_validators(
                store,
                &mut state,
                challenge.challenger_bond,
                &reporting_validators,
                tx_digest,
            );

            // Update challenge status
            challenge.status = ChallengeStatus::Resolved { challenger_lost: true };
        } else {
            // No quorum - challenger wins (benefit of doubt), bond returned
            info!("ClaimChallengeBond: no quorum, returning bond to challenger");

            if challenge.challenger_bond > 0 {
                let bond_return = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    challenge.challenger_bond,
                    Owner::AddressOwner(challenge.challenger),
                    tx_digest,
                );
                store.create_object(bond_return);
            }

            // Update challenge status
            challenge.status = ChallengeStatus::Resolved { challenger_lost: false };
        }

        // Save challenge and state (no target modification needed)
        Self::save_challenge(store, challenge_object, &challenge)?;
        Self::save_system_state(store, state_object, &state)?;

        Ok(())
    }

    /// Distribute a bond to reporting validators' staking pools.
    fn distribute_bond_to_validators(
        store: &mut TemporaryStore,
        state: &mut SystemState,
        bond: u64,
        validators: &[SomaAddress],
        tx_digest: TransactionDigest,
    ) {
        if validators.is_empty() || bond == 0 {
            return;
        }

        let per_validator = bond / validators.len() as u64;
        let remainder = bond % validators.len() as u64;

        for (i, validator_addr) in validators.iter().enumerate() {
            // First validator gets the remainder (rounding dust)
            let amount = if i == 0 {
                per_validator + remainder
            } else {
                per_validator
            };

            if amount > 0 {
                // Create a coin for this validator
                let coin = Object::new_coin(
                    ObjectID::derive_id(tx_digest, store.next_creation_num()),
                    amount,
                    Owner::AddressOwner(*validator_addr),
                    tx_digest,
                );
                store.create_object(coin);
            }
        }
    }
}

impl TransactionExecutor for ChallengeExecutor {
    fn execute(
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        _value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::InitiateChallenge(_) => {
                self.execute_initiate_challenge(store, signer, kind, tx_digest)
            }
            TransactionKind::ReportChallenge { challenge_id } => {
                self.execute_report_challenge(store, signer, challenge_id)
            }
            TransactionKind::UndoReportChallenge { challenge_id } => {
                self.execute_undo_report_challenge(store, signer, challenge_id)
            }
            TransactionKind::ClaimChallengeBond { challenge_id } => {
                self.execute_claim_challenge_bond(store, signer, challenge_id, tx_digest)
            }
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for ChallengeExecutor {
    // Use default fee calculation - no special value-based fees for challenge transactions
}

// ===========================================================================
// Pure functions for unit testing
// ===========================================================================

/// Calculate the required challenger bond based on data size.
/// Formula: data_size_bytes * bond_per_byte
pub fn calculate_challenger_bond(data_size_bytes: u64, bond_per_byte: u64) -> u64 {
    data_size_bytes.saturating_mul(bond_per_byte)
}

/// Calculate the per-validator reward share when distributing a bond.
/// Returns (per_validator_amount, remainder)
/// The remainder can be handled separately (e.g., to emission pool or first validator).
pub fn calculate_per_validator_reward(total_bond: u64, num_validators: usize) -> (u64, u64) {
    if num_validators == 0 {
        return (0, total_bond);
    }
    let per_validator = total_bond / num_validators as u64;
    let distributed = per_validator * num_validators as u64;
    let remainder = total_bond - distributed;
    (per_validator, remainder)
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Challenger Bond Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_challenger_bond_basic() {
        // 1 MB at 10 tokens/byte = 10 million tokens
        let bond = calculate_challenger_bond(1_000_000, 10);
        assert_eq!(bond, 10_000_000);
    }

    #[test]
    fn test_calculate_challenger_bond_zero_size() {
        let bond = calculate_challenger_bond(0, 100);
        assert_eq!(bond, 0);
    }

    #[test]
    fn test_calculate_challenger_bond_zero_rate() {
        let bond = calculate_challenger_bond(1_000_000, 0);
        assert_eq!(bond, 0);
    }

    #[test]
    fn test_calculate_challenger_bond_overflow_protection() {
        // Should saturate instead of overflowing
        let bond = calculate_challenger_bond(u64::MAX, u64::MAX);
        assert_eq!(bond, u64::MAX);
    }

    #[test]
    fn test_calculate_challenger_bond_typical_values() {
        // Typical case: 10 MB file at 1 token/byte = 10,485,760 tokens
        let bond = calculate_challenger_bond(10 * 1024 * 1024, 1);
        assert_eq!(bond, 10 * 1024 * 1024); // 10,485,760

        // 100 KB file at 100 tokens/byte = 10,240,000 tokens
        let bond = calculate_challenger_bond(100 * 1024, 100);
        assert_eq!(bond, 100 * 1024 * 100); // 10,240,000
    }

    // =========================================================================
    // Per-Validator Reward Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_per_validator_reward_even_split() {
        // 1000 tokens split among 4 validators = 250 each, no remainder
        let (per_validator, remainder) = calculate_per_validator_reward(1000, 4);
        assert_eq!(per_validator, 250);
        assert_eq!(remainder, 0);
    }

    #[test]
    fn test_calculate_per_validator_reward_with_remainder() {
        // 1000 tokens split among 3 validators = 333 each, 1 remainder
        let (per_validator, remainder) = calculate_per_validator_reward(1000, 3);
        assert_eq!(per_validator, 333);
        assert_eq!(remainder, 1);
    }

    #[test]
    fn test_calculate_per_validator_reward_large_remainder() {
        // 10 tokens split among 3 validators = 3 each, 1 remainder
        let (per_validator, remainder) = calculate_per_validator_reward(10, 3);
        assert_eq!(per_validator, 3);
        assert_eq!(remainder, 1);

        // 7 tokens split among 3 validators = 2 each, 1 remainder
        let (per_validator, remainder) = calculate_per_validator_reward(7, 3);
        assert_eq!(per_validator, 2);
        assert_eq!(remainder, 1);
    }

    #[test]
    fn test_calculate_per_validator_reward_zero_validators() {
        // Edge case: no validators means all goes to remainder
        let (per_validator, remainder) = calculate_per_validator_reward(1000, 0);
        assert_eq!(per_validator, 0);
        assert_eq!(remainder, 1000);
    }

    #[test]
    fn test_calculate_per_validator_reward_zero_bond() {
        let (per_validator, remainder) = calculate_per_validator_reward(0, 4);
        assert_eq!(per_validator, 0);
        assert_eq!(remainder, 0);
    }

    #[test]
    fn test_calculate_per_validator_reward_single_validator() {
        // Single validator gets everything
        let (per_validator, remainder) = calculate_per_validator_reward(1000, 1);
        assert_eq!(per_validator, 1000);
        assert_eq!(remainder, 0);
    }

    #[test]
    fn test_calculate_per_validator_reward_more_validators_than_tokens() {
        // 3 tokens split among 10 validators = 0 each, 3 remainder
        let (per_validator, remainder) = calculate_per_validator_reward(3, 10);
        assert_eq!(per_validator, 0);
        assert_eq!(remainder, 3);
    }

    // =========================================================================
    // Reward Distribution Scenarios
    // =========================================================================

    #[test]
    fn test_reward_distribution_challenger_loses_scenario() {
        // Scenario: Challenger loses (2f+1 reported against them), challenger bond = 500, 3 validators
        let challenger_bond = 500u64;
        let num_validators = 3usize;

        // Challenger bond distributed to validators who reported
        let (per_validator, remainder) = calculate_per_validator_reward(challenger_bond, num_validators);
        assert_eq!(per_validator, 166);
        assert_eq!(remainder, 2);

        // Total distributed (excluding remainder)
        let total_to_validators = per_validator * num_validators as u64;
        assert_eq!(total_to_validators, 498);

        // 2 tokens are "dust" - would typically go to first validator or emission pool
    }

    #[test]
    fn test_reward_distribution_quorum_size() {
        // With 4 validators, quorum is 3 (2f+1 where f=1)
        // If only 3 validators vote, reward is split among 3
        let challenger_bond = 1000u64;
        let quorum_voters = 3usize;

        let (per_validator, remainder) = calculate_per_validator_reward(challenger_bond, quorum_voters);
        assert_eq!(per_validator, 333);
        assert_eq!(remainder, 1);
    }

    #[test]
    fn test_reward_distribution_large_bond() {
        // 10 million tokens split among 7 validators
        let large_bond = 10_000_000u64;
        let num_validators = 7usize;

        let (per_validator, remainder) = calculate_per_validator_reward(large_bond, num_validators);
        assert_eq!(per_validator, 1_428_571);
        assert_eq!(remainder, 3);

        // Verify math: 1,428,571 * 7 = 9,999,997, remainder = 3
        let total = per_validator * num_validators as u64 + remainder;
        assert_eq!(total, large_bond);
    }

    #[test]
    fn test_reward_distribution_challenger_wins_benefit_of_doubt() {
        // Scenario: No quorum reached (challenger gets benefit of doubt)
        // Challenger bond = 500 is returned to challenger
        let challenger_bond = 500u64;

        // If no quorum, challenger bond goes back to challenger
        // No validators receive rewards in this case
        let (per_validator, remainder) = calculate_per_validator_reward(challenger_bond, 0);
        assert_eq!(per_validator, 0);
        assert_eq!(remainder, 500); // All goes to remainder (returned to challenger)
    }
}
