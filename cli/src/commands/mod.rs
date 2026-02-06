// Command modules for top-level CLI commands
pub mod balance;
pub mod claim;
pub mod env;
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

// Re-export subcommand enums for use in soma_commands.rs
pub use claim::ClaimCommand;
pub use env::EnvCommand;
pub use model::ModelCommand;
pub use objects::ObjectsCommand;
pub use submit::SubmitCommand; // Now a struct, not an enum
pub use target::TargetCommand;
pub use validator::SomaValidatorCommand;
pub use wallet::WalletCommand;
