use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use types::{
    crypto::{NetworkKeyPair, NetworkPublicKey, NetworkSignature},
    error::{ObjectError, ObjectResult, SharedError, SharedResult},
    metadata::{
        DownloadMetadata, Metadata, MtlsDownloadMetadata, MtlsDownloadMetadataV1, ObjectPath,
    },
};
use url::Url;

use crate::stores::memory::DownloadMetadataGenerator;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SignedParams {
    pub prefix: String,
    pub expires: u64,
    pub signature: NetworkSignature,
}

#[derive(Serialize, Deserialize)]
struct SignedParamInner {
    prefix: String,
    expires: u64,
}

impl SignedParams {
    pub fn new(prefix: String, timeout: Duration, signer: &NetworkKeyPair) -> SignedParams {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let expires = current_time.saturating_add(timeout.as_secs());
        let inner = SignedParamInner {
            prefix: prefix.clone(),
            expires,
        };
        let msg = bcs::to_bytes(&[0; 8]).unwrap();
        // TODO: let msg = bcs::to_bytes(&ScopedMessage::new(Scope::SignedUrl, inner)).unwrap();
        let signature = signer.sign(&msg);
        Self {
            prefix,
            expires,
            signature,
        }
    }

    pub fn verify(&self, verifier: &NetworkPublicKey) -> SharedResult<()> {
        let inner = SignedParamInner {
            prefix: self.prefix.clone(),
            expires: self.expires,
        };
        let msg = bcs::to_bytes(&[0; 8]).unwrap();

        // TODO: let msg = bcs::to_bytes(&ScopedMessage::new(Scope::SignedUrl, inner))
        //     .map_err(|e| SharedError::FastCrypto(e.to_string()))?;
        verifier
            .verify(&msg, &self.signature)
            .map_err(|e| SharedError::FastCrypto(e.to_string()))
    }
}

pub struct ObjectServiceUrlGenerator {
    base: Url,
    signer: NetworkKeyPair,
    timeout: Duration,
}

impl ObjectServiceUrlGenerator {
    pub fn new(base: Url, signer: NetworkKeyPair, timeout: Duration) -> Self {
        Self {
            base,
            signer,
            timeout,
        }
    }
}

#[async_trait]
impl DownloadMetadataGenerator for ObjectServiceUrlGenerator {
    async fn download_metadata(
        &self,
        path: ObjectPath,
        metadata: Metadata,
    ) -> ObjectResult<DownloadMetadata> {
        let prefix = path.path().to_string();

        let params = SignedParams::new(prefix.to_string(), self.timeout, &self.signer);
        let query = serde_urlencoded::to_string(&params)
            .map_err(|e| ObjectError::UrlError(e.to_string()))?;

        let mut url = self
            .base
            .join(&prefix)
            .map_err(|e| ObjectError::UrlError(format!("Failed to join path: {e}")))?;

        url.set_query(Some(&query));

        Ok(DownloadMetadata::Mtls(MtlsDownloadMetadata::V1(
            MtlsDownloadMetadataV1::new(self.signer.public(), url, metadata),
        )))
    }
}
