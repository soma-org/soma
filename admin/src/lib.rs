pub mod admin_types;

// Re-export tonic so downstream crates can use Channel
// without a direct tonic dependency.
pub use tonic;

// Tonic generated RPC stubs (client + server).
pub mod admin_gen {
    include!("proto/admin.Admin.rs");
}
