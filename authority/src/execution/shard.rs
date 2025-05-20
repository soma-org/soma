use types::{
    base::SomaAddress,
    committee::EpochId,
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ExecutionResult, SomaError},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version},
    system_state::{get_system_state, shard::ScoreSet, SystemState, SystemStateTrait},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::{object::check_ownership, FeeCalculator, TransactionExecutor};

/// Executor for shard-related transactions
pub struct ShardExecutor;

impl ShardExecutor {
    pub fn new() -> Self {
        Self {}
    }

    /// Execute EmbedData transaction
    fn execute_embed_data(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        metadata_commitment_digest: [u8; 32],
        data_size_bytes: u64,
        coin_ref: ObjectRef,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;

        // Get system state object to determine encoder byte price and current epoch
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        // Get encoder byte price from reference price in EncoderSet
        let byte_price = state.encoders.reference_byte_price;
        let current_epoch = state.epoch;

        // Calculate total price for the data size
        let total_price = byte_price.saturating_mul(data_size_bytes);

        if total_price == 0 {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: "Total price cannot be 0. Data size too small or byte price is 0!"
                    .to_string(),
            });
        }

        // Get source coin
        let source_object = store
            .read_object(&coin_id)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound { object_id: coin_id })?;

        // Check ownership
        check_ownership(&source_object, signer)?;

        // Check this is a coin object and get balance
        let source_balance =
            source_object
                .as_coin()
                .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                    object_id: source_object.id(),
                    expected_type: ObjectType::Coin,
                    actual_type: source_object.type_().clone(),
                })?;

        // Check sufficient balance
        if source_balance < total_price {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        // Calculate remaining balance
        let remaining_balance = source_balance - total_price;

        // Create ShardInput object as a shared object
        // Shard expiration = current epoch + 2
        let expiration_epoch = current_epoch + 2;

        let shard_input = Object::new_shard_input(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            metadata_commitment_digest,
            data_size_bytes,
            total_price,
            expiration_epoch,
            signer,
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            tx_digest,
        );

        store.create_object(shard_input);

        // Update source coin with remaining balance
        let mut updated_source = source_object.clone();
        updated_source.update_coin_balance(remaining_balance);
        store.mutate_input_object(updated_source);

        Ok(())
    }

    /// Execute ClaimEscrow transaction
    fn execute_claim_escrow(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        shard_input_ref: ObjectRef,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let shard_input_id = shard_input_ref.0;

        // Get system state object to determine current epoch
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        let current_epoch = state.epoch;

        // Get ShardInput object
        let shard_input_object = store.read_object(&shard_input_id).ok_or_else(|| {
            ExecutionFailureStatus::ObjectNotFound {
                object_id: shard_input_id,
            }
        })?;

        // Extract ShardInput data
        let shard_input = shard_input_object.as_shard_input().ok_or_else(|| {
            ExecutionFailureStatus::InvalidObjectType {
                object_id: shard_input_id,
                expected_type: ObjectType::ShardInput,
                actual_type: shard_input_object.type_().clone(),
            }
        })?;

        // Check if caller is the submitter
        if shard_input.submitter != signer {
            return Err(ExecutionFailureStatus::InvalidSigner.into());
        }

        // Check if shard has expired
        if current_epoch < shard_input.expiration_epoch {
            return Err(ExecutionFailureStatus::InvalidArguments {
                reason: format!(
                    "Shard has not expired yet. Current epoch: {}, expiration epoch: {}",
                    current_epoch, shard_input.expiration_epoch
                ),
            });
        }

        // Create a new coin with the escrowed amount
        let new_coin = Object::new_coin(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            shard_input.amount,
            Owner::AddressOwner(signer),
            tx_digest,
        );

        store.create_object(new_coin);

        // Delete the ShardInput object
        store.delete_input_object(&shard_input_id);

        Ok(())
    }

    /// Execute ReportScores transaction
    fn execute_report_scores(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        shard_input_ref: ObjectRef,
        score_set: ScoreSet,
        aggregated_signature: AggregatedSignature,
        shard_token: ShardToken,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        let shard_input_id = shard_input_ref.0;

        // Get system state object
        let state_object = store
            .read_object(&SYSTEM_STATE_OBJECT_ID)
            .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
                object_id: SYSTEM_STATE_OBJECT_ID,
            })?
            .clone();

        // Deserialize system state
        let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
            .map_err(|e| {
                ExecutionFailureStatus::SomaError(SomaError::from(format!(
                    "Failed to deserialize system state: {}",
                    e
                )))
            })?;

        // Verify signers are in encoder committee
        let encoder_committee = state.get_current_epoch_encoder_committee();

        for signer_addr in &aggregated_signature.signers {
            if !encoder_committee.members.contains_key(signer_addr) {
                return Err(ExecutionFailureStatus::InvalidSigners {
                    reason: format!("Signer {} is not in the encoder committee", signer_addr),
                });
            }
        }

        // Verify the number of signers meets quorum requirement
        let encoder_committee_size = encoder_committee.members.len();
        if aggregated_signature.signers.len() < (2 * encoder_committee_size / 3) {
            return Err(ExecutionFailureStatus::InsufficientQuorum.into());
        }

        // Try to get ShardInput object
        let shard_input_object = store.read_object(&shard_input_id);

        // Verify ShardInput object exists and is valid
        let shard_input = if let Some(obj) = shard_input_object {
            // Extract ShardInput data
            let shard_input =
                obj.as_shard_input()
                    .ok_or_else(|| ExecutionFailureStatus::InvalidObjectType {
                        object_id: shard_input_id,
                        expected_type: ObjectType::ShardInput,
                        actual_type: obj.type_().clone(),
                    })?;

            // Verify that the ShardToken matches the ShardInput
            if shard_token.token_id != shard_input_id {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "ShardToken does not match ShardInput".to_string(),
                });
            }

            // Verify the metadata commitment digests match
            if shard_input.digest != score_set.digest {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Metadata commitment digest mismatch".to_string(),
                });
            }

            // Verify the data sizes match
            if shard_input.data_size_bytes != score_set.data_size_bytes {
                return Err(ExecutionFailureStatus::InvalidArguments {
                    reason: "Data size mismatch".to_string(),
                });
            }

            // Delete the ShardInput object
            store.delete_input_object(&shard_input_id);

            shard_input
        } else {
            // ShardInput not found
            return Err(ExecutionFailureStatus::ObjectNotFound {
                object_id: shard_input_id,
            });
        };

        // Update the ScoreSet with the current epoch
        let mut updated_score_set = score_set;
        updated_score_set.reported_epoch = state.epoch;

        // Add the score set to the system state
        state.add_score_set(updated_score_set)?;

        // Serialize and update the system state
        let state_bytes = bcs::to_bytes(&state).map_err(|e| {
            ExecutionFailureStatus::SomaError(SomaError::from(format!(
                "Failed to serialize updated system state: {}",
                e
            )))
        })?;

        let mut updated_state_object = state_object;
        updated_state_object.data.update_contents(state_bytes);
        store.mutate_input_object(updated_state_object);

        Ok(())
    }
}

impl TransactionExecutor for ShardExecutor {
    fn execute(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        kind: TransactionKind,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        match kind {
            TransactionKind::EmbedData {
                digest,
                data_size_bytes,
                coin_ref,
            } => {
                self.execute_embed_data(store, signer, digest, data_size_bytes, coin_ref, tx_digest)
            }
            TransactionKind::ClaimEscrow { shard_input_ref } => {
                self.execute_claim_escrow(store, signer, shard_input_ref, tx_digest)
            }
            TransactionKind::ReportScores {
                shard_input_ref,
                // scores,
                signature,
                signers,
            } => self.execute_report_scores(
                store,
                signer,
                shard_input_ref,
                score_set,
                aggregated_signature,
                shard_token,
                tx_digest,
            ),
            _ => Err(ExecutionFailureStatus::InvalidTransactionType),
        }
    }
}

impl FeeCalculator for ShardExecutor {
    fn calculate_value_fee(&self, store: &TemporaryStore, kind: &TransactionKind) -> u64 {
        match kind {
            TransactionKind::EmbedData {
                data_size_bytes, ..
            } => {
                // Fee based on data size, e.g., 1 fee unit per 10KB
                let base_fee = self.base_fee();
                let size_fee = (data_size_bytes / 10240) * 100; // 100 units per 10KB
                base_fee + size_fee
            }
            TransactionKind::ClaimEscrow { .. } => {
                // Standard base fee for claiming escrow
                self.base_fee()
            }
            TransactionKind::ReportScores { .. } => {
                // Standard base fee for reporting scores
                self.base_fee()
            }
            _ => 0, // Default for non-matching types
        }
    }

    fn base_fee(&self) -> u64 {
        1500 // Higher than standard due to complexity
    }
}
