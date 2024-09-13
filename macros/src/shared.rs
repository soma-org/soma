//! sh
use syn::{parse::Parse, parse::ParseStream, Result, Type};

/// sh
pub struct TypeInput {
    pub ty: Type,
}

impl Parse for TypeInput {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self { ty: input.parse()? })
    }
}

/// sh
pub fn get_type_name(ty: &Type) -> &syn::Ident {
    match ty {
        Type::Path(type_path) if !type_path.path.segments.is_empty() => {
            &type_path.path.segments.last().unwrap().ident
        }
        _ => panic!("Unsupported type"),
    }
}
