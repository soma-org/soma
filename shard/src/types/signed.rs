use std::marker::PhantomData;

use bytes::Bytes;

use crate::{
    error::{ShardError, ShardResult},
    ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey, Scope, ScopedMessage,
};

use super::digest::Digest;

pub struct Signed<T> {
    inner: T,
    signature: Signature<T>,
}

pub struct Signature<T> {
    bytes: Bytes,
    marker: PhantomData<T>,
}

impl<T> Signed<T> {
    pub fn new(inner: T, scope: Scope, keypair: &ProtocolKeyPair) -> ShardResult<Self> {
        let inner_digest: Digest<T> = Digest::new(&inner)?;
        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(ShardError::SerializationFailure)?;
        let signature = keypair.sign(&message)?;
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
    ) -> ShardResult<()> {
        let inner_digest: Digest<T> = Digest::new(&self.inner)?;

        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(ShardError::SerializationFailure)?;

        let sig = ProtocolKeySignature::from_bytes(&self.signature.bytes)
            .map_err(ShardError::MalformedSignature)?;

        public_key
            .verify(&message, &sig)
            .map_err(ShardError::SignatureVerificationFailure)?;

        Ok(())
    }

    pub fn signature(&self) -> Signature<T> {
        self.signature
    }
}
