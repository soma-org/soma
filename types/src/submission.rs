//! Data submission types for the Soma mining competition.
//!
//! Contains `SubmissionManifest` — the manifest describing submitted data
//! (URL, checksum, size). Used on Target and Challenge objects for challenge
//! verification.

use serde::{Deserialize, Serialize};

use crate::metadata::{Manifest, ManifestAPI, MetadataAPI};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::Checksum;
    use crate::crypto::DIGEST_LENGTH;
    use crate::metadata::{ManifestV1, Metadata, MetadataV1};
    use url::Url;

    fn test_manifest() -> SubmissionManifest {
        let url = Url::parse("https://example.com/data").unwrap();
        let metadata =
            Metadata::V1(MetadataV1::new(Checksum::new_from_hash([0u8; DIGEST_LENGTH]), 1024));
        let manifest = crate::metadata::Manifest::V1(ManifestV1::new(url, metadata));
        SubmissionManifest::new(manifest)
    }

    #[test]
    fn test_submission_manifest_size() {
        let manifest = test_manifest();
        assert_eq!(manifest.size(), 1024);
    }
}
