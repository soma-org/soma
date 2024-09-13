//! sh
use crate::shared::{get_type_name, TypeInput};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

/// sh
pub(crate) fn generate_verified_type(input: TokenStream) -> TokenStream {
    let TypeInput { ty } = parse_macro_input!(input as TypeInput);

    let name = get_type_name(&ty);
    let verified_name = format_ident!("Verified{}", name);
    let digest_name = format_ident!("{}Digest", name);

    let expanded = quote! {
            /// Represents a verified `#name` instance.
    ///
    /// This struct holds a verified `#name` along with its cached digest
    /// and serialized form for efficiency. The underlying data is refcounted,
    /// making `clone()` operations relatively inexpensive.
    #[derive(Clone)]
    pub struct #verified_name {
        /// The verified `#name` instance
        inner: std::sync::Arc<#name>,
        /// The cached digest to avoid recomputation
        digest: #digest_name,
        /// The cached serialized bytes to avoid recomputation
        serialized: bytes::Bytes,
    }

    impl #verified_name {
        /// Returns the cached digest of the verified `#name`.
        pub(crate) const fn digest(&self) -> #digest_name {
            self.digest
        }

        /// Returns the cached serialized bytes of the verified `#name`.
        pub(crate) const fn serialized(&self) -> &bytes::Bytes {
            &self.serialized
        }
    }

    impl std::ops::Deref for #verified_name {
        type Target = #name;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl PartialEq for #verified_name {
        fn eq(&self, other: &Self) -> bool {
            self.digest() == other.digest()
        }
    }

    impl #name {
        /// Applies a custom check to the `#name` instance.
        ///
        /// This method allows for arbitrary verification logic to be applied
        /// to a `#name`.
        ///
        /// # Arguments
        ///
        /// * `closure` - A closure that performs the custom check.
        ///
        /// # Returns
        ///
        /// A `ShardResult` indicating success or failure of the check.
        pub fn check<F>(&self, closure: F) -> ShardResult<()>
        where
            F: FnOnce(&Self) -> ShardResult<()>,
        {
            closure(self)
        }

        /// Verifies the `#name` instance and creates a `#verified_name`.
        ///
        /// This method applies a custom verification closure to the `#name`.
        /// If the verification succeeds, it constructs and returns a `#verified_name`.
        ///
        /// # Arguments
        ///
        /// * `closure` - A closure that performs the custom verification.
        ///
        /// # Returns
        ///
        /// A `ShardResult` containing the `#verified_name` if verification succeeds.
        pub fn verify<F>(self, closure: F) -> ShardResult<#verified_name>
        where
            F: FnOnce(&Self) -> ShardResult<()>,
        {
            self.check(closure)?;

            let serialized = self.serialize()?;
            let digest = self.compute_digest_from_serialized(&serialized);

            Ok(#verified_name {
                inner: std::sync::Arc::new(self),
                digest,
                serialized,
            })
        }
    }

        };
    TokenStream::from(expanded)
}
