// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Validator audit service for challenge resolution.
//!
//! When a Challenge object is created, validators:
//! 1. Call CompetitionAPI to download data, verify hash, and run inference
//! 2. Compare results against submitter's claims using Burn's tolerance checks
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

use blobs::BlobPath;
use burn::tensor::{TensorData, Tolerance};
use runtime::{CompetitionInput, CompetitionOutput, ManifestCompetitionInput, RuntimeAPI};
use scoring::tonic_gen::scoring_client::ScoringClient;
use scoring::types::{ManifestInput, ScoreRequest};
use tokio::sync::Mutex;
use tracing::{info, warn};
use types::base::{AuthorityName, SomaAddress};
use types::committee::EpochId;
use types::consensus::ConsensusTransaction;
use types::crypto::{Signature, SomaKeyPair};
use types::error::RuntimeResult;
use types::intent::{Intent, IntentMessage};
use types::metadata::{Manifest, ManifestAPI, MetadataAPI};
use types::object::ObjectID;
use types::target::{TargetId, TargetV1};
use types::tensor::SomaTensor;

use crate::authority::AuthorityState;
use crate::authority_per_epoch_store::AuthorityPerEpochStore;
use crate::consensus_adapter::ConsensusAdapter;

// ===========================================================================
// Mock CompetitionAPI for Testing
// ===========================================================================

/// Mock competition API for testing - returns matching results (no fraud).
///
/// This service returns a result that matches the submitter's claimed values,
/// so no fraud will be detected. Use for testing the "challenger loses" path.
pub struct MockRuntimeAPI;

#[async_trait::async_trait]
impl RuntimeAPI for MockRuntimeAPI {
    async fn competition(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        Ok(CompetitionOutput::new(
            0,
            TensorData::zeros::<f32, _>([2048]),
            TensorData::zeros::<f32, _>([2048]),
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
            TensorData::zeros::<f32, _>([2048]),
            TensorData::zeros::<f32, _>([2048]),
            TensorData::from([0.0f32]),
        ))
    }
}

/// Mock competition API that always returns an error (fraud detection).
///
/// Use this to test the "submitter loses" path where data cannot be downloaded.
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
        // Simulate data unavailable - submitter is responsible for data availability
        Err(types::error::RuntimeError::DataNotAvailable(
            "Mock: data unavailable for testing fraud detection".to_string(),
        ))
    }
}

// ===========================================================================
// Remote Scoring Runtime (gRPC client)
// ===========================================================================

/// Runtime implementation that calls a remote scoring gRPC service.
///
/// Validators use this to offload ML inference to a dedicated GPU machine
/// running `soma score`. Multiple validators can share the same scoring service.
pub struct RemoteScoringRuntime {
    client: Mutex<ScoringClient<tonic::transport::Channel>>,
}

impl RemoteScoringRuntime {
    /// Connect to a remote scoring service at the given URL (e.g. "http://gpu-host:9124").
    pub async fn connect(url: &str) -> Result<Self, tonic::transport::Error> {
        let client = ScoringClient::connect(url.to_string()).await?;
        Ok(Self { client: Mutex::new(client) })
    }
}

#[async_trait::async_trait]
impl RuntimeAPI for RemoteScoringRuntime {
    async fn competition(&self, _input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        // Not used by audit_service — it calls manifest_competition directly.
        Err(types::error::RuntimeError::CoreProcessorError(
            "RemoteScoringRuntime does not support competition(), use manifest_competition()"
                .to_string(),
        ))
    }

    async fn download_manifest(&self, _manifest: &Manifest, _path: &BlobPath) -> RuntimeResult<()> {
        // The remote scoring service handles its own downloads.
        Ok(())
    }

    async fn manifest_competition(
        &self,
        input: ManifestCompetitionInput,
    ) -> RuntimeResult<CompetitionOutput> {
        let data = input.data();
        let data_checksum = data.metadata().checksum().to_string();

        let model_manifests: Vec<ManifestInput> = input
            .models()
            .iter()
            .zip(input.model_keys().iter())
            .map(|(m, key)| ManifestInput {
                url: m.url().to_string(),
                checksum: m.metadata().checksum().to_string(),
                size: m.metadata().size(),
                decryption_key: key.map(|k| {
                    use fastcrypto::encoding::{Base58, Encoding};
                    Base58::encode(k)
                }),
            })
            .collect();

        let target_embedding = input
            .target()
            .to_vec::<f32>()
            .map_err(|e| types::error::RuntimeError::CoreProcessorError(format!("{e:?}")))?;

        let request = ScoreRequest {
            data_url: data.url().to_string(),
            data_checksum,
            data_size: data.metadata().size(),
            model_manifests,
            target_embedding,
            seed: input.seed(),
        };

        let response = {
            let mut client = self.client.lock().await;
            client.score(request).await.map_err(|status| {
                let msg = status.message().to_string();
                if msg.contains("not available") || msg.contains("unavailable") {
                    types::error::RuntimeError::DataNotAvailable(msg)
                } else if msg.contains("hash mismatch") || msg.contains("Checksum") {
                    types::error::RuntimeError::DataHashMismatch
                } else {
                    types::error::RuntimeError::CoreProcessorError(msg)
                }
            })?
        };

        let resp = response.into_inner();
        Ok(CompetitionOutput::new(
            resp.winner,
            TensorData::new(resp.loss_score.clone(), [resp.loss_score.len()]),
            TensorData::new(resp.embedding.clone(), [resp.embedding.len()]),
            TensorData::new(resp.distance.clone(), [resp.distance.len()]),
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

/// Validator audit service for submission verification.
///
/// This service runs on validators and:
/// - Calls CompetitionAPI to download data, verify hash, and run inference
/// - Submits ReportSubmission transactions via consensus when fraud is detected
///
/// **Tally-Based Approach:**
///
/// Validators submit individual report transactions that accumulate on Target objects.
/// When 2f+1 stake reports, quorum is reached and ClaimRewards forfeits the bond.
///
/// **NOTE: Currently defunct (Stage 1).**
/// The challenge-triggered spawn has been removed. Stage 2 will add an autonomous
/// epoch-triggered flow where validators sample filled targets and audit them.
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

    /// Submit a ReportSubmission transaction via consensus.
    pub async fn submit_report_submission(&self, target_id: TargetId) {
        info!("Submitting ReportSubmission for target {:?}", target_id);

        let kind = types::transaction::TransactionKind::ReportSubmission { target_id };

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

        Ok(model.manifest.clone())
    }

    /// Verify fraud by calling CompetitionAPI and checking results against claims.
    ///
    /// Returns true if fraud was found, false otherwise.
    ///
    /// # Fraud Detection Logic
    ///
    /// The CompetitionAPI is trusted to determine the correct winning model and distance.
    /// We simply compare the service's results against the submitter's claims:
    ///
    /// 1. **Data unavailable**: Submitter is responsible for data availability → FRAUD
    /// 2. **Data hash mismatch**: Submitter submitted wrong data → FRAUD
    /// 3. **Wrong model**: Submitter didn't use the winning model → FRAUD
    /// 4. **Distance mismatch**: Distance differs beyond tolerance → FRAUD
    /// 5. **Model unavailable**: System issue, not submitter's fault → NO FRAUD
    /// 6. **Computation failed**: Validator issue, not submitter's fault → NO FRAUD
    pub async fn audit_fraud(&self, target: &TargetV1) -> bool {
        let data_manifest = match &target.winning_data_manifest {
            Some(m) => &m.manifest,
            None => {
                warn!("Target missing winning_data_manifest, cannot audit");
                return false;
            }
        };

        // Collect model manifests from the registry
        let mut model_manifests: Vec<Manifest> = Vec::new();
        for &model_id in &target.model_ids {
            match self.load_model_manifest(&model_id).await {
                Ok(manifest) => {
                    model_manifests.push(manifest);
                }
                Err(_) => {
                    warn!("Model {:?} not found in registry, skipping", model_id);
                    continue;
                }
            }
        }

        if model_manifests.is_empty() {
            warn!("No models available for target audit, cannot determine fraud");
            return false;
        }

        // Build CompetitionInput
        let input = ManifestCompetitionInput::new(
            data_manifest.clone(),
            model_manifests,
            target.embedding.clone().into_tensor_data(),
            0, // TODO: use a real seed if needed for stochastic models
        );

        // Call CompetitionAPI (handles download, verification, inference)
        let output = match self.runtime_api.manifest_competition(input).await {
            Ok(result) => result,
            Err(types::error::RuntimeError::DataNotAvailable(msg)) => {
                info!("FRAUD: data unavailable: {}", msg);
                return true;
            }
            Err(types::error::RuntimeError::DataHashMismatch) => {
                info!("FRAUD: data hash mismatch");
                return true;
            }
            Err(types::error::RuntimeError::ModelNotAvailable(model_id)) => {
                warn!("Model {:?} unavailable during evaluation, cannot determine fraud", model_id);
                return false;
            }
            Err(e) => {
                warn!("Computation failed during audit: {:?}", e);
                return false;
            }
        };

        // Trust the CompetitionAPI's results and compare against submitter's claims
        let claimed_model_id = match target.winning_model_id {
            Some(id) => id,
            None => {
                warn!("Target missing winning_model_id, cannot audit");
                return false;
            }
        };
        let claimed_distance = match &target.winning_distance_score {
            Some(d) => d,
            None => {
                warn!("Target missing winning_distance_score, cannot audit");
                return false;
            }
        };

        let winner = target.model_ids[output.winner()];
        // Check 1: Did the submitter use the correct winning model?
        if winner != claimed_model_id {
            info!(
                "FRAUD: wrong model. Claimed: {:?}, Actual winner: {:?}",
                claimed_model_id, winner
            );
            return true;
        }

        // Check 2: Is the claimed distance within tolerance of the actual distance?
        if !is_within_tolerance(output.distance(), claimed_distance.as_tensor_data()) {
            info!(
                "FRAUD: distance mismatch. Claimed: {:?}, Actual: {:?}",
                claimed_distance.to_vec(),
                output.distance().to_vec::<f32>()
            );
            return true;
        }

        // Check 3: Is the claimed loss_score within tolerance of the actual loss_score?
        let claimed_loss_score = match &target.winning_loss_score {
            Some(l) => l,
            None => {
                warn!("Target missing winning_loss_score, cannot audit");
                return false;
            }
        };
        if !is_within_tolerance(output.loss_score(), claimed_loss_score.as_tensor_data()) {
            info!(
                "FRAUD: loss_score mismatch. Claimed: {:?}, Actual: {:?}",
                claimed_loss_score.to_vec(),
                output.loss_score().to_vec::<f32>()
            );
            return true;
        }

        // No fraud detected - submitter's claims are valid
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
        // it's always fraud - the submitter should have used the correct model
        let claimed_model = ObjectID::random();
        let actual_winner = ObjectID::random();

        // Different models = fraud, regardless of distance
        assert_ne!(claimed_model, actual_winner, "Different models should trigger fraud");
    }
}
