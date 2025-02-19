use bytes::Bytes;
use fastcrypto::traits::{Authenticator, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::{
    digest::Digest,
    error::{SharedError, SharedResult},
    scope::{Scope, ScopedMessage},
    serialized::Serialized,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Signed<T: Serialize, S: Authenticator> {
    inner: T,
    signature: Bytes,
    #[serde(skip)]
    phantom: PhantomData<S>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Signature<T: Serialize, S: Authenticator> {
    signature: Bytes,
    #[serde(skip)]
    phantom: PhantomData<(T, S)>,
}
impl<T, S> Signed<T, S>
where
    T: Serialize,
    S: Authenticator,
{
    pub fn new<K>(inner: T, scope: Scope, signer: &K) -> SharedResult<Self>
    where
        K: SigningKey<Sig = S> + Signer<S>,
    {
        let inner_digest = Digest::new(&inner)?;
        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(SharedError::SerializationFailure)?;
        let signature = signer.sign(&message);
        Ok(Self {
            inner,
            signature: Bytes::copy_from_slice(signature.as_bytes()),
            phantom: PhantomData,
        })
    }

    pub fn verify(&self, scope: Scope, key: &S::PubKey) -> SharedResult<()>
    where
        S::PubKey: VerifyingKey<Sig = S>,
    {
        let inner_digest = Digest::new(&self.inner)?;
        let message = bcs::to_bytes(&ScopedMessage::new(scope, inner_digest))
            .map_err(SharedError::SerializationFailure)?;
        let sig =
            S::from_bytes(&self.signature).map_err(SharedError::SignatureVerificationFailure)?;
        key.verify(&message, &sig)
            .map_err(SharedError::SignatureVerificationFailure)?;
        Ok(())
    }
    pub fn into_inner(self) -> T {
        self.inner
    }
    pub fn signature(self) -> Signature<T, S> {
        Signature {
            signature: self.signature,
            phantom: PhantomData,
        }
    }
    pub fn serialized(&self) -> Serialized<Signature<T, S>> {
        Serialized::new(self.signature.clone())
    }
}

impl<T, S> std::ops::Deref for Signed<T, S>
where
    T: Serialize,
    S: Authenticator,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
