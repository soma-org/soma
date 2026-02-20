//! Validator audit service for challenge resolution.
//!
//! When a Challenge object is created, validators:
//! 1. Call CompetitionAPI to download data, verify hash, and run inference
//! 2. Compare results against miner's claims using Burn's tolerance checks
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

use burn::tensor::{TensorData, Tolerance};
use tokio::sync::mpsc;
use tracing::{info, warn};

use blobs::BlobPath;
use runtime::{CompetitionInput, CompetitionOutput, ManifestCompetitionInput, RuntimeAPI};
use types::{
    base::{AuthorityName, SomaAddress},
    challenge::{ChallengeV1, ChallengeId},
    committee::EpochId,
    consensus::ConsensusTransaction,
    crypto::{Signature, SomaKeyPair},
    error::RuntimeResult,
    intent::{Intent, IntentMessage},
    metadata::Manifest,
    model::ModelId,
    object::ObjectID,
    target::TargetId,
    tensor::SomaTensor,
};

use crate::{
    authority::AuthorityState, authority_per_epoch_store::AuthorityPerEpochStore,
    consensus_adapter::ConsensusAdapter,
};

// ===========================================================================
// Mock CompetitionAPI for Testing
// ===========================================================================

/// Mock competition API for testing - returns matching results (no fraud).
///
/// This service returns a result that matches the miner's claimed values,
/// so no fraud will be detected. Use for testing the "challenger loses" path.
pub struct MockRuntimeAPI;

#[async_trait::async_trait]
impl RuntimeAPI for MockRuntimeAPI {
    async fn competition(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        Ok(CompetitionOutput::new(
            0,
            TensorData::zeros::<f32, _>([768]),
            TensorData::zeros::<f32, _>([768]),
            TensorData::from([0.0f32]),
        ))
    }
    async fn download_manifest(&self, manifest: &Manifest, path: &BlobPath) -> RuntimeResult<()> {
        Ok(())
    }
    async fn manifest_competition(
        &self,
        input: ManifestCompetitionInput,
    ) -> RuntimeResult<CompetitionOutput> {
        Ok(CompetitionOutput::new(
            0,
            TensorData::zeros::<f32, _>([768]),
            TensorData::zeros::<f32, _>([768]),
            TensorData::from([0.0f32]),
        ))
    }
}

/// Mock competition API that always returns an error (fraud detection).
///
/// Use this to test the "miner loses" path where data cannot be downloaded.
pub struct MockFraudCompetitionAPI;

#[async_trait::async_trait]
impl RuntimeAPI for MockFraudCompetitionAPI {
    async fn competition(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        Err(types::error::RuntimeError::DataNotAvailable(
            "Mock: data unavailable for testing fraud detection".to_string(),
        ))
    }
    async fn download_manifest(&self, manifest: &Manifest, path: &BlobPath) -> RuntimeResult<()> {
        Ok(())
    }
    async fn manifest_competition(
        &self,
        input: ManifestCompetitionInput,
    ) -> RuntimeResult<CompetitionOutput> {
        // Simulate data unavailable - miner is responsible for data availability
        Err(types::error::RuntimeError::DataNotAvailable(
            "Mock: data unavailable for testing fraud detection".to_string(),
        ))
    }
}

// ===========================================================================
// Tolerance Checking
// ===========================================================================

/// Check if two TensorData values are approximately equal using Burn's permissive tolerance.
///
/// Tolerance::permissive() uses:
/// - relative tolerance: 1% (0.01)
/// - absolute tolerance: 0.01
///
/// This is appropriate for comparing results from different GPUs which may have
/// small numerical differences due to floating point variance.
///
/// Returns true if values are within tolerance, false otherwise.
fn is_within_tolerance(computed: &TensorData, claimed: &TensorData) -> bool {
    // assert_approx_eq panics on mismatch, so we catch the panic
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        computed.assert_approx_eq::<f32>(claimed, Tolerance::permissive());
    }))
    .is_ok()
}

// ===========================================================================
// Audit Service
// ===========================================================================

/// Validator audit service for challenge resolution.
///
/// This service runs on validators and:
/// - Listens for new Challenge objects (via channel from CheckpointExecutor)
/// - Calls CompetitionAPI to download data, verify hash, and run inference
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

    /// Runtime API for running inference (handles all downloading).
    runtime_api: Arc<dyn RuntimeAPI>,

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
        runtime_api: Arc<dyn RuntimeAPI>,
        epoch: EpochId,
        consensus_adapter: Arc<ConsensusAdapter>,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) -> Arc<Self> {
        Arc::new(Self {
            authority_name,
            validator_address,
            account_keypair,
            state,
            runtime_api,
            epoch,
            consensus_adapter,
            epoch_store,
        })
    }

    /// Spawn the audit service task.
    ///
    /// - `challenge_rx`: Receives new Challenge objects from CheckpointExecutor
    pub async fn spawn(self: Arc<Self>, mut challenge_rx: mpsc::Receiver<ChallengeV1>) {
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
    /// All challenges are fraud challenges - call CompetitionAPI, compare results.
    /// Based on the result, submit appropriate report transactions.
    ///
    /// **Report semantics:**
    /// - `ReportSubmission`: "This submission is fraudulent" → miner loses bond
    /// - `ReportChallenge`: "This challenge is invalid" → challenger loses bond
    async fn handle_new_challenge(&self, challenge: ChallengeV1) {
        info!("Auditing challenge {:?} for target {:?}", challenge.id, challenge.target_id);

        // Run fraud audit - returns true if fraud was found
        let fraud_found = self.audit_fraud(&challenge).await;

        if fraud_found {
            // Fraud confirmed: report submission with challenger attribution
            // Don't report challenge (challenger was right)
            self.submit_report_submission(challenge.target_id, Some(challenge.challenger)).await;
        } else {
            // No fraud: report challenge (challenger was wrong)
            self.submit_report_challenge(challenge.id).await;
        }
    }

    /// Submit a ReportSubmission transaction via consensus.
    async fn submit_report_submission(&self, target_id: TargetId, challenger: Option<SomaAddress>) {
        info!(
            "Submitting ReportSubmission for target {:?} with challenger {:?}",
            target_id, challenger
        );

        let kind = types::transaction::TransactionKind::ReportSubmission { target_id, challenger };

        self.submit_validator_transaction(kind).await;
    }

    /// Submit a ReportChallenge transaction via consensus.
    async fn submit_report_challenge(&self, challenge_id: ChallengeId) {
        info!("Submitting ReportChallenge for challenge {:?} (challenger was wrong)", challenge_id);

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
        let consensus_tx =
            ConsensusTransaction::new_user_transaction_message(&self.authority_name, tx);

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

        let model = state.model_registry().active_models.get(model_id).ok_or(())?;

        let weights_manifest = model.weights_manifest.as_ref().ok_or(())?;

        Ok(weights_manifest.manifest.clone())
    }

    /// Verify fraud by calling CompetitionAPI and checking results against claims.
    ///
    /// Returns true if fraud was found (challenger wins), false otherwise.
    ///
    /// # Fraud Detection Logic
    ///
    /// The CompetitionAPI is trusted to determine the correct winning model and distance.
    /// We simply compare the service's results against the miner's claims:
    ///
    /// 1. **Data unavailable**: Miner is responsible for data availability → FRAUD
    /// 2. **Data hash mismatch**: Miner submitted wrong data → FRAUD
    /// 3. **Wrong model**: Miner didn't use the winning model → FRAUD
    /// 4. **Distance mismatch**: Distance differs beyond tolerance → FRAUD
    /// 5. **Model unavailable**: System issue, not miner's fault → NO FRAUD
    /// 6. **Computation failed**: Validator issue, not miner's fault → NO FRAUD
    async fn audit_fraud(&self, challenge: &ChallengeV1) -> bool {
        let data_manifest = &challenge.winning_data_manifest.manifest;

        // Collect model manifests from the registry
        let mut model_manifests: Vec<Manifest> = Vec::new();
        for &model_id in &challenge.model_ids {
            match self.load_model_manifest(&model_id).await {
                Ok(manifest) => {
                    model_manifests.push(manifest);
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

        // Build CompetitionInput
        let input = ManifestCompetitionInput::new(
            data_manifest.clone(),
            model_manifests,
            challenge.target_embedding.clone().into_tensor_data(),
            0, // TODO: use a real seed if needed for stochastic models
        );

        // Call CompetitionAPI (handles download, verification, inference)
        let output = match self.runtime_api.manifest_competition(input).await {
            Ok(result) => result,
            Err(types::error::RuntimeError::DataNotAvailable(msg)) => {
                // Miner is responsible for keeping data available during challenge window
                info!("FRAUD: data unavailable for challenge {:?}: {}", challenge.id, msg);
                return true;
            }
            Err(types::error::RuntimeError::DataHashMismatch) => {
                // Miner submitted data that doesn't match their commitment
                info!("FRAUD: data hash mismatch for challenge {:?}", challenge.id);
                return true;
            }
            Err(types::error::RuntimeError::ModelNotAvailable(model_id)) => {
                // Model weights couldn't be downloaded. This is a system/model-owner issue,
                // not the miner's fault. Cannot determine fraud.
                warn!("Model {:?} unavailable during evaluation, cannot determine fraud", model_id);
                return false;
            }
            Err(e) => {
                // Other errors (computation failed, etc). This is a validator-side issue.
                // Cannot determine fraud.
                warn!("Computation failed for challenge {:?}: {:?}", challenge.id, e);
                return false;
            }
        };

        // Trust the CompetitionAPI's results and compare against miner's claims
        let claimed_model_id = challenge.winning_model_id;
        let claimed_distance = &challenge.winning_distance_score;

        let winner = challenge.model_ids[output.winner()];
        // Check 1: Did the miner use the correct winning model?
        if winner != claimed_model_id {
            info!(
                "FRAUD: wrong model for challenge {:?}. Claimed: {:?}, Actual winner: {:?}",
                challenge.id, claimed_model_id, winner
            );
            return true;
        }

        // Check 2: Is the claimed distance within tolerance of the actual distance?
        // Using Burn's Tolerance::permissive() (1% relative, 0.01 absolute)
        if !is_within_tolerance(output.distance(), claimed_distance.as_tensor_data()) {
            info!(
                "FRAUD: distance mismatch for challenge {:?}. Claimed: {:?}, Actual: {:?}",
                challenge.id,
                claimed_distance.to_vec(),
                output.distance().to_vec::<f32>()
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
    fn test_mock_competition_api() {
        let api = MockRuntimeAPI;
        let _: Arc<dyn RuntimeAPI> = Arc::new(api);
    }

    #[test]
    fn test_tolerance_within_bounds() {
        // Create two TensorData values that are very close (within 1% tolerance)
        let computed = TensorData::from([0.5f32]);
        let claimed = TensorData::from([0.5f32]);

        assert!(
            is_within_tolerance(&computed, &claimed),
            "Identical values should be within tolerance"
        );
    }

    #[test]
    fn test_tolerance_slightly_different() {
        // Values that differ by less than 1% should be within tolerance
        let computed = TensorData::from([0.5f32]);
        let claimed = TensorData::from([0.504f32]); // 0.8% difference

        assert!(
            is_within_tolerance(&computed, &claimed),
            "Values within 1% should be within tolerance"
        );
    }

    #[test]
    fn test_tolerance_outside_bounds() {
        // Values that differ by more than 1% and more than 0.01 absolute
        let computed = TensorData::from([0.5f32]);
        let claimed = TensorData::from([0.52f32]); // 4% difference

        assert!(
            !is_within_tolerance(&computed, &claimed),
            "Values outside 1% should NOT be within tolerance"
        );
    }

    #[test]
    fn test_tolerance_small_absolute_difference() {
        // Even small values should pass absolute tolerance of 0.01
        let computed = TensorData::from([0.001f32]);
        let claimed = TensorData::from([0.008f32]);

        // Difference is 0.007 which is less than 0.01 absolute
        assert!(
            is_within_tolerance(&computed, &claimed),
            "Small absolute differences should be within tolerance"
        );
    }

    #[test]
    fn test_wrong_model_is_always_fraud() {
        // When the CompetitionAPI returns a different winning model,
        // it's always fraud - the miner should have used the correct model
        let claimed_model = ObjectID::random();
        let actual_winner = ObjectID::random();

        // Different models = fraud, regardless of distance
        assert_ne!(claimed_model, actual_winner, "Different models should trigger fraud");
    }
}
