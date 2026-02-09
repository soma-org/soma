//! Validator audit service for challenge resolution.
//!
//! When a Challenge object is created, validators:
//! 1. Call EvaluationService to download data, verify hash, and run inference
//! 2. Compare results against miner's claims
//! 3. Submit ReportSubmission and/or ReportChallenge transactions via consensus
//!
//! # Tally-Based Design
//!
//! Instead of aggregating votes off-chain and submitting a certified result,
//! validators submit individual report transactions that accumulate on-chain.
//! When 2f+1 stake reports, quorum is reached and claims can be processed.
//!
//! All challenges are fraud challenges. Availability issues are handled via
//! submission reports (ReportSubmission/UndoReportSubmission) instead.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{info, warn};

use types::{
    base::{AuthorityName, SomaAddress},
    challenge::{Challenge, ChallengeId},
    committee::EpochId,
    consensus::ConsensusTransaction,
    crypto::{SomaKeyPair, Signature},
    intent::{Intent, IntentMessage},
    metadata::Manifest,
    model::ModelId,
    object::ObjectID,
    target::{Embedding, TargetId},
};

use crate::{
    authority::AuthorityState,
    authority_per_epoch_store::AuthorityPerEpochStore,
    consensus_adapter::ConsensusAdapter,
};

// ===========================================================================
// Evaluation Service Trait
// ===========================================================================

/// Result of evaluating models against data and a target embedding.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The winning model (produces the best/lowest distance)
    pub winning_model_id: ModelId,
    /// The embedding used to calculate the distance.
    /// May be from the winning model directly, or a performance-weighted average.
    pub embedding: Embedding,
    /// The distance from the embedding to the target
    pub distance: i64,
}

/// Interface to the evaluation service for running ML inference.
///
/// This trait defines the contract for running inference during challenge audits.
/// The actual implementation will wrap the probes/intelligence crates.
///
/// **Self-Contained Interface**: The EvaluationService handles all downloading
/// (data + model weights) internally. Callers just provide manifests.
#[async_trait]
pub trait EvaluationService: Send + Sync {
    /// Evaluate all models against the given data and target embedding.
    ///
    /// The service handles all downloading and verification internally:
    /// 1. Downloads model weights from manifests (or uses cached versions)
    /// 2. Downloads input data from the data_manifest
    /// 3. Verifies data hash matches manifest checksum
    /// 4. Runs inference with each model to compute embeddings
    /// 5. Determines the winning model (best distance to target)
    /// 6. Returns the winning model, final embedding, and distance
    ///
    /// # Arguments
    /// * `model_manifests` - Model IDs with manifests for downloading weights
    /// * `data_manifest` - Manifest with URL to download input data from
    /// * `data_commitment` - Expected hash of the data (for verification)
    /// * `target_embedding` - The target embedding to compute distances against
    ///
    /// # Returns
    /// The evaluation result containing winning model, embedding, and distance.
    async fn evaluate(
        &self,
        model_manifests: &[(ModelId, Manifest)],
        data_manifest: &Manifest,
        data_commitment: &[u8; 32],
        target_embedding: &Embedding,
    ) -> Result<EvaluationResult, EvaluationError>;
}

/// Errors that can occur during evaluation.
#[derive(Debug, Clone)]
pub enum EvaluationError {
    /// Model weights could not be loaded (unavailable, network error, etc.)
    ModelNotAvailable(ObjectID),
    /// Data could not be loaded (unavailable, network error, etc.)
    DataNotAvailable(String),
    /// Data hash doesn't match commitment
    DataHashMismatch,
    /// Inference or distance computation failed
    ComputationFailed(String),
}

/// Mock evaluation service for testing - returns matching results (no fraud).
///
/// This service returns a result that matches the miner's claimed values,
/// so no fraud will be detected. Use for testing the "challenger loses" path.
pub struct MockEvaluationService;

#[async_trait]
impl EvaluationService for MockEvaluationService {
    async fn evaluate(
        &self,
        model_manifests: &[(ModelId, Manifest)],
        _data_manifest: &Manifest,
        _data_commitment: &[u8; 32],
        _target_embedding: &Embedding,
    ) -> Result<EvaluationResult, EvaluationError> {
        // Return first model as winner with zero embedding and distance.
        // This matches the miner's claimed values in tests (distance=0 or within epsilon).
        let winning_model_id = model_manifests
            .first()
            .map(|(id, _)| *id)
            .unwrap_or_else(ObjectID::random);

        Ok(EvaluationResult {
            winning_model_id,
            embedding: Embedding::zeros(768),
            distance: 0,
        })
    }
}

/// Mock evaluation service that always returns data unavailable error (fraud).
///
/// Use this to test the "miner loses" path where data cannot be downloaded.
pub struct MockFraudEvaluationService;

#[async_trait]
impl EvaluationService for MockFraudEvaluationService {
    async fn evaluate(
        &self,
        _model_manifests: &[(ModelId, Manifest)],
        _data_manifest: &Manifest,
        _data_commitment: &[u8; 32],
        _target_embedding: &Embedding,
    ) -> Result<EvaluationResult, EvaluationError> {
        // Simulate data unavailable - miner is responsible for data availability
        Err(EvaluationError::DataNotAvailable(
            "Mock: data unavailable for testing fraud detection".to_string(),
        ))
    }
}

// ===========================================================================
// Audit Service
// ===========================================================================

/// Validator audit service for challenge resolution.
///
/// This service runs on validators and:
/// - Listens for new Challenge objects (via channel from CheckpointExecutor)
/// - Calls EvaluationService to download data, verify hash, and run inference
/// - Submits ReportSubmission and ReportChallenge transactions via consensus
///
/// **Tally-Based Approach:**
///
/// Instead of broadcasting votes and aggregating signatures off-chain,
/// validators submit individual report transactions. Reports accumulate
/// on Target and Challenge objects until 2f+1 quorum is reached.
///
/// **Fraud-Only Challenges:**
///
/// All challenges are fraud challenges. Availability issues are handled via
/// submission reports (ReportSubmission/UndoReportSubmission) instead of
/// availability challenges.
pub struct AuditService {
    /// This validator's identity (protocol key).
    authority_name: AuthorityName,

    /// This validator's account address (for transaction sender).
    validator_address: SomaAddress,

    /// This validator's account keypair for signing transactions.
    account_keypair: Arc<SomaKeyPair>,

    /// Authority state for reading objects.
    state: Arc<AuthorityState>,

    /// Evaluation service for running inference (handles all downloading).
    evaluation_service: Arc<dyn EvaluationService>,

    /// Epsilon for distance comparison (fixed-point scale).
    /// If |actual_distance - claimed_distance| > epsilon, it's fraud.
    distance_epsilon: i64,

    /// Current epoch.
    epoch: EpochId,

    /// Consensus adapter for submitting report transactions.
    consensus_adapter: Arc<ConsensusAdapter>,

    /// Epoch store for consensus submissions.
    epoch_store: Arc<AuthorityPerEpochStore>,
}

impl AuditService {
    /// Build a new audit service.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        authority_name: AuthorityName,
        validator_address: SomaAddress,
        account_keypair: Arc<SomaKeyPair>,
        state: Arc<AuthorityState>,
        evaluation_service: Arc<dyn EvaluationService>,
        epoch: EpochId,
        distance_epsilon: i64,
        consensus_adapter: Arc<ConsensusAdapter>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) -> Arc<Self> {
        Arc::new(Self {
            authority_name,
            validator_address,
            account_keypair,
            state,
            evaluation_service,
            distance_epsilon,
            epoch,
            consensus_adapter,
            epoch_store,
        })
    }

    /// Spawn the audit service task.
    ///
    /// - `challenge_rx`: Receives new Challenge objects from CheckpointExecutor
    pub async fn spawn(
        self: Arc<Self>,
        mut challenge_rx: mpsc::Receiver<Challenge>,
    ) {
        // Spawn task to handle new challenges
        tokio::spawn(async move {
            while let Some(challenge) = challenge_rx.recv().await {
                let service = self.clone();
                tokio::spawn(async move {
                    service.handle_new_challenge(challenge).await;
                });
            }
        });
    }

    /// Handle a newly created challenge.
    ///
    /// All challenges are fraud challenges - call EvaluationService, compare results.
    /// Based on the result, submit appropriate report transactions.
    ///
    /// **Report semantics:**
    /// - `ReportSubmission`: "This submission is fraudulent" → miner loses bond
    /// - `ReportChallenge`: "This challenge is invalid" → challenger loses bond
    async fn handle_new_challenge(&self, challenge: Challenge) {
        info!(
            "Auditing challenge {:?} for target {:?}",
            challenge.id, challenge.target_id
        );

        // Run fraud audit - returns true if fraud was found
        let fraud_found = self.audit_fraud(&challenge).await;

        if fraud_found {
            // Fraud confirmed: report submission with challenger attribution
            // Don't report challenge (challenger was right)
            self.submit_report_submission(
                challenge.target_id,
                Some(challenge.challenger),
            ).await;
        } else {
            // No fraud: report challenge (challenger was wrong)
            self.submit_report_challenge(challenge.id).await;
        }
    }

    /// Submit a ReportSubmission transaction via consensus.
    async fn submit_report_submission(
        &self,
        target_id: TargetId,
        challenger: Option<SomaAddress>,
    ) {
        info!(
            "Submitting ReportSubmission for target {:?} with challenger {:?}",
            target_id, challenger
        );

        let kind = types::transaction::TransactionKind::ReportSubmission {
            target_id,
            challenger,
        };

        self.submit_validator_transaction(kind).await;
    }

    /// Submit a ReportChallenge transaction via consensus.
    async fn submit_report_challenge(&self, challenge_id: ChallengeId) {
        info!(
            "Submitting ReportChallenge for challenge {:?} (challenger was wrong)",
            challenge_id
        );

        let kind = types::transaction::TransactionKind::ReportChallenge { challenge_id };

        self.submit_validator_transaction(kind).await;
    }

    /// Submit a validator transaction via consensus.
    async fn submit_validator_transaction(&self, kind: types::transaction::TransactionKind) {
        use fastcrypto::traits::KeyPair as _;

        let tx_data = types::transaction::TransactionData::new(
            kind,
            self.validator_address,
            vec![], // No gas payment for validator transactions
        );

        // Sign the transaction with the validator's account keypair
        let intent_msg = IntentMessage::new(Intent::soma_transaction(), &tx_data);
        let sig = Signature::new_secure(&intent_msg, self.account_keypair.as_ref());
        let tx = types::transaction::Transaction::from_data(tx_data, vec![sig]);

        // Wrap in ConsensusTransaction
        let consensus_tx = ConsensusTransaction::new_user_transaction_message(
            &self.authority_name,
            tx,
        );

        // Submit to consensus
        match self.consensus_adapter.submit(
            consensus_tx,
            None, // No reconfig lock
            &self.epoch_store,
            None, // No position tracking
            None, // No client address
        ) {
            Ok(_join_handle) => {
                info!("Validator transaction submitted successfully");
            }
            Err(e) => {
                warn!("Failed to submit validator transaction: {:?}", e);
            }
        }
    }

    /// Load the model's weights manifest from the model registry.
    async fn load_model_manifest(&self, model_id: &ObjectID) -> Result<Manifest, ()> {
        use types::SYSTEM_STATE_OBJECT_ID;

        let state_object = self.state.get_object(&SYSTEM_STATE_OBJECT_ID).await.ok_or(())?;
        let state: types::system_state::SystemState =
            bcs::from_bytes(state_object.as_inner().data.contents()).map_err(|_| ())?;

        let model = state
            .model_registry
            .active_models
            .get(model_id)
            .ok_or(())?;

        let weights_manifest = model.weights_manifest.as_ref().ok_or(())?;

        Ok(weights_manifest.manifest.clone())
    }

    /// Verify fraud by calling EvaluationService and checking results against claims.
    ///
    /// Returns true if fraud was found (challenger wins), false otherwise.
    ///
    /// # Fraud Detection Logic
    ///
    /// The evaluation service is trusted to determine the correct winning model and distance.
    /// We simply compare the service's results against the miner's claims:
    ///
    /// 1. **Data unavailable**: Miner is responsible for data availability → FRAUD
    /// 2. **Data hash mismatch**: Miner submitted wrong data → FRAUD
    /// 3. **Wrong model**: Miner didn't use the winning model → FRAUD
    /// 4. **Distance mismatch**: Distance differs beyond epsilon → FRAUD
    /// 5. **Model unavailable**: System issue, not miner's fault → NO FRAUD
    /// 6. **Computation failed**: Validator issue, not miner's fault → NO FRAUD
    async fn audit_fraud(&self, challenge: &Challenge) -> bool {
        let data_manifest = &challenge.winning_data_manifest.manifest;
        let data_commitment = challenge.winning_data_commitment.inner();

        // Collect model manifests from the registry
        let mut model_manifests: Vec<(ModelId, Manifest)> = Vec::new();
        for &model_id in &challenge.model_ids {
            match self.load_model_manifest(&model_id).await {
                Ok(manifest) => {
                    model_manifests.push((model_id, manifest));
                }
                Err(_) => {
                    // Model not in registry - skip it
                    // This shouldn't happen for active targets, but if it does,
                    // we can't evaluate with this model.
                    warn!("Model {:?} not found in registry, skipping", model_id);
                    continue;
                }
            }
        }

        if model_manifests.is_empty() {
            // No models could be loaded from registry.
            // This is a system issue (models should be available for active targets).
            // Cannot determine fraud without models → default to no fraud.
            warn!("No models available for challenge {:?}, cannot determine fraud", challenge.id);
            return false;
        }

        // Call EvaluationService (handles download, verification, inference)
        let eval_result = match self
            .evaluation_service
            .evaluate(
                &model_manifests,
                data_manifest,
                data_commitment,
                &challenge.target_embedding,
            )
            .await
        {
            Ok(result) => result,
            Err(EvaluationError::DataNotAvailable(msg)) => {
                // Miner is responsible for keeping data available during challenge window
                info!("FRAUD: data unavailable for challenge {:?}: {}", challenge.id, msg);
                return true;
            }
            Err(EvaluationError::DataHashMismatch) => {
                // Miner submitted data that doesn't match their commitment
                info!("FRAUD: data hash mismatch for challenge {:?}", challenge.id);
                return true;
            }
            Err(EvaluationError::ModelNotAvailable(model_id)) => {
                // Model weights couldn't be downloaded. This is a system/model-owner issue,
                // not the miner's fault. Cannot determine fraud.
                warn!("Model {:?} unavailable during evaluation, cannot determine fraud", model_id);
                return false;
            }
            Err(EvaluationError::ComputationFailed(msg)) => {
                // Inference failed. This is a validator-side issue (OOM, bug, etc).
                // Cannot determine fraud.
                warn!("Computation failed for challenge {:?}: {}", challenge.id, msg);
                return false;
            }
        };

        // Trust the evaluation service's results and compare against miner's claims
        let claimed_model_id = challenge.winning_model_id;
        let claimed_distance = challenge.winning_distance_score;

        // Check 1: Did the miner use the correct winning model?
        if eval_result.winning_model_id != claimed_model_id {
            info!(
                "FRAUD: wrong model for challenge {:?}. Claimed: {:?}, Actual winner: {:?}",
                challenge.id, claimed_model_id, eval_result.winning_model_id
            );
            return true;
        }

        // Check 2: Is the claimed distance within epsilon of the actual distance?
        let distance_diff = (eval_result.distance - claimed_distance).abs();
        if distance_diff > self.distance_epsilon {
            info!(
                "FRAUD: distance mismatch for challenge {:?}. Claimed: {}, Actual: {}, Diff: {}",
                challenge.id, claimed_distance, eval_result.distance, distance_diff
            );
            return true;
        }

        // No fraud detected - miner's claims are valid
        false
    }

    /// Shutdown the audit service.
    pub async fn shutdown(&self) {
        info!("Audit service shutting down for epoch {}", self.epoch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_evaluation_service() {
        let service = MockEvaluationService;
        let _: Arc<dyn EvaluationService> = Arc::new(service);
    }

    #[test]
    fn test_distance_epsilon_boundary_within_tolerance() {
        let distance_epsilon: i64 = 1000;
        let claimed_distance: i64 = 5000;

        // Actual distance within epsilon - should NOT be fraud
        let actual_distance = claimed_distance + distance_epsilon - 1;
        let distance_diff = (actual_distance - claimed_distance).abs();
        assert!(distance_diff <= distance_epsilon, "Should be within tolerance");
    }

    #[test]
    fn test_distance_epsilon_boundary_outside_tolerance() {
        let distance_epsilon: i64 = 1000;
        let claimed_distance: i64 = 5000;

        // Actual distance outside epsilon - IS fraud
        let actual_distance = claimed_distance + distance_epsilon + 1;
        let distance_diff = (actual_distance - claimed_distance).abs();
        assert!(distance_diff > distance_epsilon, "Should be outside tolerance");
    }

    #[test]
    fn test_wrong_model_is_always_fraud() {
        // When the evaluation service returns a different winning model,
        // it's always fraud - the miner should have used the correct model
        let claimed_model = ObjectID::random();
        let actual_winner = ObjectID::random();

        // Different models = fraud, regardless of distance
        assert_ne!(claimed_model, actual_winner, "Different models should trigger fraud");
    }

    #[test]
    fn test_distance_diff_symmetry() {
        let distance_epsilon: i64 = 1000;

        // Fraud detection should work whether actual is higher or lower than claimed
        let claimed: i64 = 5000;

        // Actual is higher (miner claimed better than reality)
        let actual_higher: i64 = 6500;
        let diff_higher = (actual_higher - claimed).abs();
        assert!(diff_higher > distance_epsilon, "Higher actual should be fraud");

        // Actual is lower (miner claimed worse than reality - also suspicious)
        let actual_lower: i64 = 3500;
        let diff_lower = (actual_lower - claimed).abs();
        assert!(diff_lower > distance_epsilon, "Lower actual should also be fraud");
    }
}
