//! sh
use crate::shared::{get_type_name, TypeInput};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

/// sh
pub(crate) fn generate_digest_type(input: TokenStream) -> TokenStream {
    let TypeInput { ty } = parse_macro_input!(input as TypeInput);

    let name = get_type_name(&ty);

    let digest_name = format_ident!("{}Digest", name);

    let expanded = quote! {

    /// Represents a hash digest for `#name`.
    #[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
    pub struct #digest_name([u8; DIGEST_LENGTH]);

    impl #digest_name {
        /// Lexicographically minimal digest
        const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
        /// Lexicographically maximal digest
        const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
    }
    impl std::hash::Hash for #digest_name {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write(&self.0[..8]);
        }
    }

    impl From<#digest_name > for fastcrypto::hash::Digest<{ DIGEST_LENGTH }> {
        fn from(hd: #digest_name ) -> Self {
            fastcrypto::hash::Digest::new(hd.0)
        }
    }

    impl std::fmt::Display for #digest_name {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                    .get(0..4)
                    .ok_or(std::fmt::Error)?
            )
        }
    }

    impl std::fmt::Debug for #digest_name  {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                        f,
                "{}",
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
            )
        }
    }

    impl AsRef<[u8]> for #digest_name {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    impl #name {
        /// Serializes the `#name` instance using `bcs`.
        ///
        /// # Returns
        ///
        /// A `ShardResult` containing the serialized `bytes::Bytes` if successful.
        pub fn serialize(&self) -> ShardResult<bytes::Bytes> {
            let bytes = bcs::to_bytes(self).map_err(ShardError::SerializationFailure)?;
            Ok(bytes.into())
        }

        /// Computes the digest of the serialized `#name` instance.
        ///
        /// # Arguments
        ///
        /// * `serialized` - The serialized `bytes::Bytes` of the `#name`.
        ///
        /// # Returns
        ///
        /// The computed `#digest_name`.
        fn compute_digest_from_serialized(&self, serialized: &bytes::Bytes) -> #digest_name {
            let mut hasher = DefaultHashFunction::new();
            hasher.update(serialized);
            #digest_name(hasher.finalize().into())
        }
    }};

    TokenStream::from(expanded)
}
