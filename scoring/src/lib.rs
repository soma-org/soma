// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

pub mod types;

// Re-export tonic so downstream crates (python-sdk) can use Channel
// without a direct tonic dependency.
pub use tonic;

// Tonic generated RPC stubs (client + server).
pub mod tonic_gen {
    include!("proto/scoring.Scoring.rs");
}

#[cfg(feature = "server")]
pub mod scoring;
#[cfg(feature = "server")]
pub mod server;

/// A small model config for testing (embedding_dim=16, num_layers=2).
#[cfg(feature = "server")]
pub fn model_config_small() -> runtime::ModelConfig {
    runtime::ModelConfig {
        embedding_dim: 16,
        pwff_hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        vocab_size: 264,
        ..runtime::ModelConfig::new()
    }
}
