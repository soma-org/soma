//! sh
use crate::shared::{get_type_name, TypeInput};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

/// sh
pub(crate) fn generate_signed_type(input: TokenStream) -> TokenStream {
    let TypeInput { ty } = parse_macro_input!(input as TypeInput);

    let name = get_type_name(&ty);

    let signed_name = format_ident!("Signed{}", name);
    let inner_digest_name = format_ident!("Inner{}Digest", name);

    let expanded = quote! {




    /// `#signed_name` contains a `#name` instance and its corresponding signature.
    ///
    /// The signature is computed by serializing the `MacroTest` instance and hashing it
    /// to produce an inner digest used solely for signing. This inner digest is then
    /// wrapped in a `ScopedMessage` with a `Scope` that matches the struct name.
    /// Scopes help prevent malicious signature reuse across different domains.
    /// The resulting scoped message is then signed.
    ///
    /// Note: The recommended way to refer to a signed type is by computing a digest that
    /// includes the signature. This ensures that different valid signatures for the same
    /// content result in different digests.
    #[derive(Debug, Deserialize, Serialize)]
    pub struct #signed_name {
        /// The underlying `MacroTest` instance
        inner: MacroTest,
        /// The byte representation of the signature
        signature: bytes::Bytes,
    }

    /// A private type representing the digest of `MacroTest`.
    #[derive(Serialize, Deserialize)]
    struct #inner_digest_name([u8; DIGEST_LENGTH]);

    impl std::ops::Deref for #signed_name {
        type Target = MacroTest;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl MacroTest {
        /// Signs the `MacroTest` instance using the provided keypair.
        ///
        /// This method internally calculates an inner digest, scopes the message,
        /// and derives the signature.
        ///
        /// # Arguments
        ///
        /// * `keypair` - The `ProtocolKeyPair` used for signing.
        ///
        /// # Returns
        ///
        /// A `ShardResult` containing the `#signed_name` if successful.
        pub fn sign(self, keypair: &ProtocolKeyPair) -> ShardResult<#signed_name> {
            let signature = self.compute_signature(keypair)?;
            Ok(#signed_name {
                inner: self,
                signature: bytes::Bytes::copy_from_slice(signature.to_bytes()),
            })
        }

        /// Computes the inner digest of the `MacroTest` instance.
        ///
        /// This method serializes the `MacroTest` using `bcs`, then hashes the result.
        fn inner_digest(&self) -> ShardResult<InnerMacroTestDigest> {
            let mut hasher = DefaultHashFunction::new();
            hasher.update(bcs::to_bytes(self).map_err(ShardError::SerializationFailure)?);
            Ok(InnerMacroTestDigest(hasher.finalize().into()))
        }

        /// Creates a `ScopedMessage` with `Scope::MacroTest` for the given digest.
        const fn scoped_message(digest: InnerMacroTestDigest) -> ScopedMessage<InnerMacroTestDigest> {
            ScopedMessage::new(Scope::MacroTest, digest)
        }

        /// Computes the signature for the `MacroTest` instance.
        ///
        /// This method calls `inner_digest`, `scoped_message`, and then signs the resulting message.
        fn compute_signature(&self, keypair: &ProtocolKeyPair) -> ShardResult<ProtocolKeySignature> {
            let digest = self.inner_digest()?;
            let message = bcs::to_bytes(&Self::scoped_message(digest))
                .map_err(ShardError::SerializationFailure)?;
            Ok(keypair.sign(&message))
        }
    }

    impl #signed_name {
        /// Verifies the signature of the `#signed_name` instance.
        ///
        /// This method computes a hash digest of the inner `MacroTest`, converts it to a
        /// `ScopedMessage` with `Scope::MacroTest`, and verifies the signature against
        /// the provided public key.
        ///
        /// # Arguments
        ///
        /// * `public_key` - The `ProtocolPublicKey` used for verification.
        ///
        /// # Returns
        ///
        /// A `ShardResult` indicating success or failure of the verification.
        pub fn verify_signature(&self, public_key: &ProtocolPublicKey) -> ShardResult<()> {
            let inner = &self.inner;
            let digest = inner.inner_digest()?;
            let message = bcs::to_bytes(&MacroTest::scoped_message(digest))
                .map_err(ShardError::SerializationFailure)?;
            let sig = ProtocolKeySignature::from_bytes(&self.signature)
                .map_err(ShardError::MalformedSignature)?;
            public_key
                .verify(&message, &sig)
                .map_err(ShardError::SignatureVerificationFailure)?;

            Ok(())
        }
    }


        };

    TokenStream::from(expanded)
}
