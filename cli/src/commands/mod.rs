// Command modules for top-level CLI commands
pub mod balance;
pub mod challenge;
pub mod claim;
pub mod data;
pub mod env;
pub mod faucet;
pub mod model;
pub mod objects;
pub mod pay;
pub mod send;
pub mod stake;
pub mod submit;
pub mod target;
pub mod transfer;
pub mod tx;
pub mod validator;
pub mod wallet;

// Shared parsing helpers used by model and submit commands
pub(crate) mod parse_helpers;

// Re-export subcommand enums for use in soma_commands.rs
pub use challenge::ChallengeCommand;
pub use env::EnvCommand;
pub use model::ModelCommand;
pub use objects::ObjectsCommand;
pub use target::TargetCommand;
pub use validator::SomaValidatorCommand;
pub use wallet::WalletCommand;
