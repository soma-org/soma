// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use proc_macro::TokenStream;

use quote::format_ident;
use quote::quote;
use syn::Attribute;
use syn::Data;
use syn::DataStruct;
use syn::DeriveInput;
use syn::Fields;
use syn::FieldsNamed;
use syn::Meta;
use syn::parse_macro_input;

/// Attribute macro for config-based structs: derives serde, renames fields to kebab-case, and
/// gives each field a `#[serde(default = ...)]` that pulls from the struct's `Default` impl.
#[allow(non_snake_case)]
#[proc_macro_attribute]
pub fn DefaultConfig(_attr: TokenStream, input: TokenStream) -> TokenStream {
    let DeriveInput { attrs, vis, ident, generics, data } =
        parse_macro_input!(input as DeriveInput);

    let Data::Struct(DataStruct { struct_token, fields, semi_token }) = data else {
        panic!("Default configs must be structs.");
    };

    let Fields::Named(FieldsNamed { brace_token: _, named }) = fields else {
        panic!("Default configs must have named fields.");
    };

    let fields_with_names: Vec<_> = named
        .iter()
        .map(|field| {
            let Some(ident) = &field.ident else {
                panic!("All fields must have an identifier.");
            };

            (ident, field)
        })
        .collect();

    let fields = fields_with_names.iter().map(|(name, field)| {
        let default = format!("{ident}::__default_{name}");
        quote! { #[serde(default = #default)] #field }
    });

    let defaults = fields_with_names.iter().map(|(name, field)| {
        let ty = &field.ty;
        let fn_name = format_ident!("__default_{}", name);
        let cfg = extract_cfg(&field.attrs);

        quote! {
            #[doc(hidden)] #cfg
            fn #fn_name() -> #ty {
                <Self as std::default::Default>::default().#name
            }
        }
    });

    // Detect an existing #[serde(rename_all = "...")] on the struct. syn 2 stores the parsed Meta
    // on the attribute directly; for `Meta::List` we walk the nested meta with `parse_nested_meta`.
    let has_rename_all = attrs.iter().any(|attr| {
        if !attr.path().is_ident("serde") {
            return false;
        }
        let Meta::List(list) = &attr.meta else {
            return false;
        };
        let mut found = false;
        // parse_nested_meta returns Err on the first failed step; the closure runs once per
        // top-level item inside the list. Setting `found` short-circuits via Err.
        let _ = list.parse_nested_meta(|nested| {
            if nested.path.is_ident("rename_all") {
                found = true;
                Err(nested.error("rename_all found"))
            } else {
                // Skip any value to keep the parser moving past `= "..."` etc.
                if nested.input.peek(syn::Token![=]) {
                    let _: syn::Token![=] = nested.input.parse()?;
                    let _: syn::Lit = nested.input.parse()?;
                }
                Ok(())
            }
        });
        found
    });

    let rename_all = if !has_rename_all {
        quote! { #[serde(rename_all = "kebab-case")] }
    } else {
        quote! {}
    };

    TokenStream::from(quote! {
        #[derive(serde::Serialize, serde::Deserialize)]
        #rename_all
        #(#attrs)* #vis #struct_token #ident #generics {
            #(#fields),*
        } #semi_token

        impl #ident {
            #(#defaults)*
        }
    })
}

fn extract_cfg(attrs: &[Attribute]) -> Option<&Attribute> {
    attrs.iter().find(|attr| attr.path().is_ident("cfg"))
}
