use std::marker::PhantomData;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    crypto::keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
    error::{SharedError, SharedResult},
    scope::{Scope, ScopedMessage},
};

use super::digest::Digest;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Signed<T: Serialize> {
    inner: T,
    signature: Signature<T>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Signature<T: Serialize> {
    bytes: Bytes,
    marker: PhantomData<T>,
}

impl<T: Serialize> Signed<T> {
    pub fn new(inner: T, scope: Scope, keypair: &ProtocolKeyPair) -> SharedResult<Self> {
        let inner_digest: Digest<T> = Digest::new(&inner)?;
        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(SharedError::SerializationFailure)?;
        let signature = keypair.sign(&message);
        Ok(Self {
            inner,
            signature: Signature {
                bytes: bytes::Bytes::copy_from_slice(signature.to_bytes()),
                marker: PhantomData,
            },
        })
    }

    pub fn verify_signature(
        &self,
        scope: Scope,
        public_key: &ProtocolPublicKey,
    ) -> SharedResult<()> {
        let inner_digest: Digest<T> = Digest::new(&self.inner)?;

        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(SharedError::SerializationFailure)?;

        let sig = ProtocolKeySignature::from_bytes(&self.signature.bytes)
            .map_err(SharedError::MalformedSignature)?;

        public_key
            .verify(&message, &sig)
            .map_err(SharedError::SignatureVerificationFailure)?;

        Ok(())
    }

    pub fn signature(&self) -> Signature<T> {
        Signature {
            bytes: self.signature.bytes.clone(),
            marker: PhantomData,
        }
    }
}

impl<T: Serialize> std::ops::Deref for Signed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
