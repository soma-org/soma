use crate::{
    error::{SharedError, SharedResult},
    shard_crypto::{
        digest::Digest,
        scope::{Scope, ScopedMessage},
        serialized::Serialized,
    },
};
use bytes::Bytes;
use fastcrypto::traits::{Authenticator, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Clone, Deserialize, Serialize, Hash)]
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

    pub fn verify_signature(&self, scope: Scope, key: &S::PubKey) -> SharedResult<()>
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
    pub fn raw_signature(&self) -> Bytes {
        self.signature.clone()
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

impl<T: Serialize, S: Authenticator> PartialEq for Signed<T, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.signature != other.signature {
            return false;
        }
        let self_inner_bytes = match bcs::to_bytes(&self.inner) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        let other_inner_bytes = match bcs::to_bytes(&other.inner) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        self_inner_bytes == other_inner_bytes
    }
}

impl<T: Serialize, S: Authenticator> Eq for Signed<T, S> {}
