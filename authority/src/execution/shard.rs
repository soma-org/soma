use std::{collections::HashSet, fmt::format, sync::Arc};

use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey},
    digest::Digest,
    metadata::{DownloadableMetadataAPI, MetadataAPI, MetadataCommitment},
    scope::{Scope, ScopedMessage},
    signed::Signed,
    verified::Verified,
};
use tracing::{debug, error};
use types::{
    base::SomaAddress,
    committee::{Committee, EncoderCommittee, EpochId},
    digests::TransactionDigest,
    effects::ExecutionFailureStatus,
    error::{ConsensusError, ExecutionResult, SomaError},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version},
    score_set::ScoreSetAPI,
    shard_verifier::ShardVerifier,
    system_state::{get_system_state, shard::ShardResult, SystemState, SystemStateTrait},
    temporary_store::TemporaryStore,
    transaction::TransactionKind,
    SYSTEM_STATE_OBJECT_ID,
};

use super::{object::check_ownership, FeeCalculator, TransactionExecutor};

/// Executor for shard-related transactions
pub struct ShardExecutor {
    shard_verifier: ShardVerifier,
}

impl ShardExecutor {
    pub fn new() -> Self {
        let shard_verifier = ShardVerifier::new(100, None);

        Self { shard_verifier }
    }

    /// Execute EmbedData transaction
    fn execute_embed_data(
        &self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        digest: Digest<MetadataCommitment>,
        data_size_bytes: usize,
        coin_ref: ObjectRef,
        tx_digest: TransactionDigest,
        value_fee: u64,
    ) -> ExecutionResult<()> {
        let coin_id = coin_ref.0;

        // Check if the coin is also the gas coin
        let is_gas_coin = store.gas_object_id == Some(coin_id);

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
        let embed_price = byte_price.saturating_mul(data_size_bytes as u64);

        if embed_price == 0 {
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

        // Calculate operation fees
        // We're creating one ShardInput object and updating the source coin (2 operations)
        let operation_fee = self.calculate_operation_fee(2);

        // If this is the gas coin, we need to ensure there's enough for both embed_price and fees
        let total_required = if is_gas_coin {
            // Total fee = value_fee (passed in) + operation_fee (calculated)
            let gas_fee = value_fee + operation_fee;
            embed_price + gas_fee
        } else {
            // Just need enough for the embed price
            embed_price
        };

        // Check sufficient balance
        if source_balance < total_required {
            return Err(ExecutionFailureStatus::InsufficientCoinBalance.into());
        }

        // Shard expiration = current epoch + 2
        let expiration_epoch = current_epoch + 2;

        // Create ShardInput object as a shared object
        let shard_input = Object::new_shard_input(
            ObjectID::derive_id(tx_digest, store.next_creation_num()),
            digest,
            data_size_bytes,
            embed_price,
            expiration_epoch,
            signer,
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            tx_digest,
        );

        store.create_object(shard_input);

        // Update source coin with remaining balance after deducting the embed price ONLY
        // (gas fees are handled separately by the transaction system)
        let remaining_balance = source_balance - embed_price;

        // If remaining balance is 0, delete the coin
        if remaining_balance == 0 && !is_gas_coin {
            // Only delete if it's not the gas coin
            store.delete_input_object(&coin_id);
        } else {
            // Otherwise update the coin
            let mut updated_source = source_object.clone();
            updated_source.update_coin_balance(remaining_balance);
            store.mutate_input_object(updated_source);
        }

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
            return Err(ExecutionFailureStatus::InvalidOwnership {
                object_id: shard_input_id,
                expected_owner: shard_input.submitter,
                actual_owner: Some(signer),
            }
            .into());
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
        &mut self,
        store: &mut TemporaryStore,
        signer: SomaAddress,
        shard_input_ref: ObjectRef,
        scores_bytes: Vec<u8>,
        signature: Vec<u8>,
        signers: Vec<EncoderPublicKey>,
        tx_digest: TransactionDigest,
    ) -> ExecutionResult<()> {
        // let shard_input_id = shard_input_ref.0;
        // // Try to get ShardInput object
        // let shard_input_object = store.read_object(&shard_input_id).ok_or_else(|| {
        //     ExecutionFailureStatus::ObjectNotFound {
        //         object_id: shard_input_id,
        //     }
        // })?;

        // let shard_input = shard_input_object.as_shard_input().ok_or_else(|| {
        //     ExecutionFailureStatus::InvalidObjectType {
        //         object_id: shard_input_id,
        //         expected_type: ObjectType::ShardInput,
        //         actual_type: shard_input_object.type_().clone(),
        //     }
        // })?;

        // // Get system state object
        // let state_object = store
        //     .read_object(&SYSTEM_STATE_OBJECT_ID)
        //     .ok_or_else(|| ExecutionFailureStatus::ObjectNotFound {
        //         object_id: SYSTEM_STATE_OBJECT_ID,
        //     })?
        //     .clone();

        // // Deserialize system state
        // let mut state = bcs::from_bytes::<SystemState>(state_object.as_inner().data.contents())
        //     .map_err(|e| {
        //         ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //             "Failed to deserialize system state: {}",
        //             e
        //         )))
        //     })?;

        // let current_epoch = state.epoch;

        // // Check expiration epoch
        // if current_epoch >= shard_input.expiration_epoch {
        //     return Err(ExecutionFailureStatus::InvalidArguments {
        //         reason: format!(
        //             "Shard input has expired. Current epoch: {}, Expiration epoch: {}",
        //             current_epoch, shard_input.expiration_epoch
        //         ),
        //     });
        // }

        // // Deserialize scores bytes
        // tracing::debug!("Deserializing scores bytes");
        // let scores: Signed<ShardScore, min_sig::BLS12381Signature> =
        //     match bcs::from_bytes(&scores_bytes) {
        //         Ok(s) => {
        //             tracing::debug!("Successfully deserialized scores");
        //             s
        //         }
        //         Err(e) => {
        //             tracing::error!("Failed to deserialize scores: {:?}", e);
        //             return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //                 "Failed to deserialize scores: {}",
        //                 e
        //             ))));
        //         }
        //     };

        // let committees = state.committees(scores.auth_token().epoch())?;
        // let authority_committee = Committee::convert_to_authority_committee(
        //     committees.build_validator_committee().committee(),
        // );
        // let encoder_committee = EncoderCommittee::convert_encoder_committee(
        //     &committees.build_encoder_committee(),
        //     scores.auth_token().epoch(),
        // );
        // let vdf_iterations = state.parameters.vdf_iterations; // TODO: Revisit if this needs to be queried by the auth token epoch too

        // let (shard, _) = match self.shard_verifier.verify(
        //     authority_committee,
        //     encoder_committee,
        //     vdf_iterations,
        //     scores.auth_token(),
        // ) {
        //     Ok(s) => {
        //         debug!("Shard verification succeeded");
        //         s
        //     }
        //     Err(e) => {
        //         error!("Shard verification failed: {:?}", e);
        //         return Err(ExecutionFailureStatus::InvalidArguments {
        //             reason: format!("Shard verification failed: {:?}", e),
        //         });
        //     }
        // };

        // // Create verified scores object
        // let verified_scores = match Verified::new(scores, scores_bytes.into(), |scores| {
        //     tracing::debug!("Verifying signed scores");
        //     verify_signed_score(scores, &shard)
        // }) {
        //     Ok(v) => {
        //         tracing::debug!("Verified scores created successfully");
        //         v
        //     }
        //     Err(e) => {
        //         tracing::error!("Failed to create verified scores: {:?}", e);
        //         return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //             "Failed to create verified scores: {}",
        //             e
        //         ))));
        //     }
        // };

        // // 2.1 Verify signers are in the shard
        // let unique_signers: HashSet<&EncoderPublicKey> = signers.iter().collect();

        // for encoder in &unique_signers {
        //     if !shard.encoders().contains(encoder) {
        //         return Err(ExecutionFailureStatus::InvalidArguments {
        //             reason: format!("Encoder {:?} is not in the shard", encoder),
        //         });
        //     }
        // }

        // // 2.2 Verify the number of signers meets quorum requirement
        // if (unique_signers.len() as u32) < shard.quorum_threshold() {
        //     return Err(ExecutionFailureStatus::InvalidArguments {
        //         reason: format!(
        //             "Insufficient signers. Got: {}, Required: {}",
        //             signers.len(),
        //             shard.quorum_threshold()
        //         ),
        //     });
        // }

        // // Verify the aggregate signature
        // let message = bcs::to_bytes(&ScopedMessage::new(Scope::Score, verified_scores.digest()))
        //     .map_err(|e| {
        //         ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //             "Failed to deserialize system state: {}",
        //             e
        //         )))
        //     })?;
        // let sig = match EncoderAggregateSignature::from_bytes(&signature) {
        //     Ok(s) => {
        //         tracing::debug!("Successfully deserialized scores");
        //         s
        //     }
        //     Err(e) => {
        //         tracing::error!("Failed to deserialize scores: {:?}", e);
        //         return Err(ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //             "Failed to deserialize scores: {}",
        //             e
        //         ))));
        //     }
        // };
        // sig.verify(&signers, &message).map_err(|e| {
        //     ExecutionFailureStatus::SomaError(SomaError::from(
        //         ConsensusError::SignatureVerificationFailure(e),
        //     ))
        // })?;

        // // 2.3 Verify the metadata commitment digests match
        // let metadata_commitment_digest = verified_scores
        //     .auth_token()
        //     .metadata_commitment
        //     .digest()
        //     .map_err(|e| {
        //         ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //             "Failed to get metadata commitment digest: {}",
        //             e
        //         )))
        //     })?;

        // if shard_input.digest != metadata_commitment_digest {
        //     return Err(ExecutionFailureStatus::InvalidArguments {
        //         reason: "Metadata commitment digest mismatch".to_string(),
        //     });
        // }

        // // 2.4 Verify the data sizes match
        // let data_size = verified_scores
        //     .auth_token()
        //     .metadata_commitment()
        //     .downloadable_metadata()
        //     .metadata()
        //     .size();

        // if shard_input.data_size_bytes != data_size {
        //     return Err(ExecutionFailureStatus::InvalidArguments {
        //         reason: format!(
        //             "Data size mismatch. ShardInput: {}, Scores: {}",
        //             shard_input.data_size_bytes, data_size
        //         ),
        //     });
        // }

        // // Delete the ShardInput object
        // store.delete_input_object(&shard_input_id);

        // // Add ShardResult to the system state
        // let shard_digest = verified_scores
        //     .signed_score_set()
        //     .into_inner()
        //     .shard_digest();
        // let score_set = verified_scores.signed_score_set().into_inner();
        // state.add_shard_result(
        //     shard_digest,
        //     ShardResult {
        //         digest: metadata_commitment_digest,
        //         data_size_bytes: data_size,
        //         amount: shard_input.amount,
        //         score_set,
        //     },
        // );

        // // Serialize updated system state
        // let state_bytes = bcs::to_bytes(&state).map_err(|e| {
        //     ExecutionFailureStatus::SomaError(SomaError::from(format!(
        //         "Failed to serialize updated system state: {}",
        //         e
        //     )))
        // })?;

        // // Update the system state object
        // let mut updated_state_object = state_object;
        // updated_state_object.data.update_contents(state_bytes);
        // store.mutate_input_object(updated_state_object);

        Ok(())
    }
}

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
                digest,
                data_size_bytes,
                coin_ref,
            } => self.execute_embed_data(
                store,
                signer,
                digest,
                data_size_bytes,
                coin_ref,
                tx_digest,
                value_fee,
            ),
            TransactionKind::ClaimEscrow { shard_input_ref } => {
                self.execute_claim_escrow(store, signer, shard_input_ref, tx_digest)
            }
            TransactionKind::ReportScores {
                shard_input_ref,
                scores,
                signature,
                signers,
            } => self.execute_report_scores(
                store,
                signer,
                shard_input_ref,
                scores,
                signature,
                signers,
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
                data_size_bytes,
                coin_ref,
                ..
            } => {
                // Get the actual embedding cost
                if let Some(state_obj) = store.read_object(&SYSTEM_STATE_OBJECT_ID) {
                    if let Ok(state) =
                        bcs::from_bytes::<SystemState>(state_obj.as_inner().data.contents())
                    {
                        let byte_price = state.encoders.reference_byte_price;
                        let embed_cost = byte_price.saturating_mul(*data_size_bytes as u64);

                        // Fee is 0.05% (5 basis points) of the embedding cost
                        let fee = (embed_cost * 5) / 10000;
                        return std::cmp::max(fee, self.base_fee());
                    }
                }
                // Default if we can't determine the state
                self.base_fee()
            }
            TransactionKind::ClaimEscrow { shard_input_ref } => {
                // Fee based on the amount being claimed
                if let Some(shard_obj) = store.read_object(&shard_input_ref.0) {
                    if let Some(shard_input) = shard_obj.as_shard_input() {
                        // Fee is 0.05% (5 basis points) of the claimed amount
                        let fee = (shard_input.amount * 5) / 10000;
                        return std::cmp::max(fee, self.base_fee());
                    }
                }
                // Default if we can't determine the value
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

    fn write_fee_per_object(&self) -> u64 {
        300 // Standard per-write fee
    }
}
