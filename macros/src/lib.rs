//! sh
mod generate_digest_type;
mod generate_signed_type;
mod generate_verified_type;
mod shared;
use proc_macro::TokenStream;

#[proc_macro]
pub fn generate_digest_type(input: TokenStream) -> TokenStream {
    generate_digest_type::generate_digest_type(input)
}

#[proc_macro]
pub fn generate_signed_type(input: TokenStream) -> TokenStream {
    generate_signed_type::generate_signed_type(input)
}

#[proc_macro]
pub fn generate_verified_type(input: TokenStream) -> TokenStream {
    generate_verified_type::generate_verified_type(input)
}
