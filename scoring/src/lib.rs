pub mod scoring;
pub mod server;
pub mod types;

/// A small model config for testing (embedding_dim=16, num_layers=2).
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
