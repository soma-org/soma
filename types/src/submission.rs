//! Data submission types for the Soma mining competition.
//!
//! Submissions are the mechanism by which miners submit data that fills targets.
//! Each submission contains a commitment to the data, a manifest for downloading
//! the data, and the embedding/scores reported by the miner.
//!
//! Key design decisions:
//! - Single transaction (no commit-reveal) - front-running deferred to future versions
//! - Both distance and reconstruction scores are submitter-reported
//! - Bond scales with data size (submission_bond_per_byte * data_size)
//! - Submission is embedded in the Target object when filled

use serde::{Deserialize, Serialize};

use crate::{
    base::SomaAddress,
    committee::EpochId,
    digests::DataCommitment,
    metadata::{Manifest, ManifestAPI, MetadataAPI},
    model::ModelId,
    object::ObjectID,
    target::Embedding,
};

/// Type alias: submissions are identified by their ObjectID.
pub type SubmissionId = ObjectID;

/// A data submission manifest, following the ModelWeightsManifest pattern.
/// Contains the URL where submitted data can be downloaded for challenge verification.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct SubmissionManifest {
    /// Existing type: URL + Metadata(checksum, size)
    pub manifest: Manifest,
}

impl SubmissionManifest {
    /// Creates a new SubmissionManifest.
    pub fn new(manifest: Manifest) -> Self {
        Self { manifest }
    }

    /// Returns the size of the submitted data in bytes.
    pub fn size(&self) -> usize {
        use crate::metadata::MetadataAPI;
        self.manifest.metadata().size()
    }
}

/// A data submission to a target in the Soma mining competition.
///
/// Submissions represent a miner's claim to have found data that embeds within
/// a target's radius. The submission includes the data commitment, manifest for
/// downloading, the chosen model, and the reported scores.
///
/// Both distance_score and reconstruction_score are submitter-reported and
/// verified only during challenge audit by re-running inference.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Submission {
    /// The miner who submitted
    pub miner: SomaAddress,

    /// Commitment to the raw data: hash(data_bytes)
    pub data_commitment: DataCommitment,

    /// Manifest for the submitted data (URL + checksum + size)
    pub data_manifest: SubmissionManifest,

    /// Which model the miner chose from the target's model_ids
    pub model_id: ModelId,

    /// Embedding vector as provided by the submitter (fixed-point i64).
    /// Verified only during challenge audit.
    pub embedding: Embedding,

    /// Distance score reported by submitter (fixed-point, scale DISTANCE_SCALE).
    /// Must be <= target.distance_threshold (lower is better). Verified during challenge audit.
    pub distance_score: i64,

    /// Reconstruction error (MSE) reported by submitter (fixed-point).
    /// Must be <= target.reconstruction_threshold (lower is better). Verified during challenge audit.
    pub reconstruction_score: u64,

    /// Bond amount locked by the miner (calculated as submission_bond_per_byte * data_size)
    pub bond_amount: u64,

    /// Epoch in which the submission was made
    pub submit_epoch: EpochId,
}

impl Submission {
    /// Creates a new Submission with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        miner: SomaAddress,
        data_commitment: DataCommitment,
        data_manifest: SubmissionManifest,
        model_id: ModelId,
        embedding: Embedding,
        distance_score: i64,
        reconstruction_score: u64,
        bond_amount: u64,
        submit_epoch: EpochId,
    ) -> Self {
        Self {
            miner,
            data_commitment,
            data_manifest,
            model_id,
            embedding,
            distance_score,
            reconstruction_score,
            bond_amount,
            submit_epoch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::Checksum;
    use crate::crypto::DIGEST_LENGTH;
    use crate::metadata::{ManifestV1, Metadata, MetadataV1};
    use ndarray::Array1;
    use url::Url;

    fn test_manifest() -> SubmissionManifest {
        let url = Url::parse("https://example.com/data").unwrap();
        let metadata = Metadata::V1(MetadataV1::new(
            Checksum::new_from_hash([0u8; DIGEST_LENGTH]),
            1024,
        ));
        let manifest = crate::metadata::Manifest::V1(ManifestV1::new(url, metadata));
        SubmissionManifest::new(manifest)
    }

    #[test]
    fn test_submission_manifest_size() {
        let manifest = test_manifest();
        assert_eq!(manifest.size(), 1024);
    }

    #[test]
    fn test_submission_creation() {
        let miner = SomaAddress::default();
        let data_commitment = DataCommitment::random();
        let data_manifest = test_manifest();
        let model_id = ObjectID::random();
        let embedding = Array1::zeros(10);

        let submission = Submission::new(
            miner,
            data_commitment,
            data_manifest,
            model_id,
            embedding.clone(),
            1000,  // distance_score
            500,   // reconstruction_score
            10240, // bond_amount
            0,     // submit_epoch
        );

        assert_eq!(submission.miner, miner);
        assert_eq!(submission.model_id, model_id);
        assert_eq!(submission.distance_score, 1000);
        assert_eq!(submission.reconstruction_score, 500);
        assert_eq!(submission.bond_amount, 10240);
        assert_eq!(submission.embedding, embedding);
    }
}
