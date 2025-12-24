use std::fmt::{self, Display, Formatter};

use colored::Colorize;
use serde::Serialize;
use types::{
    balance_change::BalanceChange,
    base::SomaAddress,
    digests::{ObjectDigest, TransactionDigest},
    effects::{ExecutionStatus, TransactionEffects, TransactionEffectsAPI},
    object::{ObjectID, ObjectRef, Owner, Version},
    tx_fee::TransactionFee,
};

/// Response from validator command execution
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ValidatorCommandResponse {
    /// Validator info files were created
    MakeValidatorInfo,

    /// Metadata was displayed (no transaction)
    DisplayMetadata,

    /// Transaction was executed
    Transaction(TransactionResponse),

    /// Unsigned transaction was serialized (for offline signing)
    SerializedTransaction {
        serialized_unsigned_transaction: String,
    },
}

/// A successful transaction response with effects
#[derive(Debug, Clone, Serialize)]
pub struct TransactionResponse {
    pub digest: TransactionDigest,
    pub status: TransactionStatus,
    pub executed_epoch: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<u64>,
    pub fee: TransactionFee,

    // Object changes
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub created: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mutated: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deleted: Vec<ObjectRefDisplay>,

    // Gas object
    pub gas_object: OwnedObjectRef,

    // Balance changes
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub balance_changes: Vec<BalanceChange>,
}

/// Object reference with owner information for display
#[derive(Debug, Clone, Serialize)]
pub struct OwnedObjectRef {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub owner: OwnerDisplay,
}

/// Simplified owner display
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OwnerDisplay {
    AddressOwner { address: SomaAddress },
    Shared { initial_shared_version: Version },
    Immutable,
}

/// Object reference for deleted objects (no owner)
#[derive(Debug, Clone, Serialize)]
pub struct ObjectRefDisplay {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
}

#[derive(Debug, Clone, Serialize)]
pub enum TransactionStatus {
    Success,
    Failure { error: String },
}

impl TransactionStatus {
    pub fn is_success(&self) -> bool {
        matches!(self, TransactionStatus::Success)
    }
}

impl From<Owner> for OwnerDisplay {
    fn from(owner: Owner) -> Self {
        match owner {
            Owner::AddressOwner(address) => OwnerDisplay::AddressOwner { address },
            Owner::Shared {
                initial_shared_version,
            } => OwnerDisplay::Shared {
                initial_shared_version,
            },
            Owner::Immutable => OwnerDisplay::Immutable,
        }
    }
}

impl From<(ObjectRef, Owner)> for OwnedObjectRef {
    fn from((obj_ref, owner): (ObjectRef, Owner)) -> Self {
        Self {
            object_id: obj_ref.0,
            version: obj_ref.1,
            digest: obj_ref.2,
            owner: owner.into(),
        }
    }
}

impl From<ObjectRef> for ObjectRefDisplay {
    fn from(obj_ref: ObjectRef) -> Self {
        Self {
            object_id: obj_ref.0,
            version: obj_ref.1,
            digest: obj_ref.2,
        }
    }
}

impl TransactionResponse {
    /// Create a TransactionResponse from effects and optional checkpoint
    pub fn from_effects(effects: &TransactionEffects, checkpoint: Option<u64>) -> Self {
        let status = match effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => TransactionStatus::Failure {
                error: format!("{}", error),
            },
        };

        Self {
            digest: effects.transaction_digest_owned(),
            status,
            executed_epoch: effects.executed_epoch(),
            checkpoint,
            fee: effects.transaction_fee().clone(),
            created: effects.created().into_iter().map(Into::into).collect(),
            mutated: effects
                .mutated_excluding_gas()
                .into_iter()
                .map(Into::into)
                .collect(),
            deleted: effects.deleted().into_iter().map(Into::into).collect(),
            gas_object: effects.gas_object().into(),
            balance_changes: Vec::new(), // Populated separately if available
        }
    }

    /// Create a TransactionResponse with balance changes
    pub fn from_effects_with_balance_changes(
        effects: &TransactionEffects,
        checkpoint: Option<u64>,
        balance_changes: Vec<BalanceChange>,
    ) -> Self {
        let mut response = Self::from_effects(effects, checkpoint);
        response.balance_changes = balance_changes;
        response
    }

    /// Add balance changes to an existing response
    pub fn with_balance_changes(mut self, balance_changes: Vec<BalanceChange>) -> Self {
        self.balance_changes = balance_changes;
        self
    }
}

impl Display for ValidatorCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ValidatorCommandResponse::MakeValidatorInfo => {
                writeln!(
                    f,
                    "{}",
                    "Validator info files created successfully.".green()
                )?;
                writeln!(f, "Generated files:")?;
                writeln!(f, "  - protocol.key")?;
                writeln!(f, "  - account.key")?;
                writeln!(f, "  - network.key")?;
                writeln!(f, "  - worker.key")?;
                writeln!(f, "  - validator.info")?;
            }
            ValidatorCommandResponse::DisplayMetadata => {
                // Metadata is printed separately via ValidatorSummary
            }
            ValidatorCommandResponse::Transaction(tx_response) => {
                write!(f, "{}", tx_response)?;
            }
            ValidatorCommandResponse::SerializedTransaction {
                serialized_unsigned_transaction,
            } => {
                writeln!(f, "{}", "Serialized unsigned transaction:".cyan())?;
                writeln!(f, "{}", serialized_unsigned_transaction)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Use 'soma client execute-signed-tx' to submit after signing.".yellow()
                )?;
            }
        }
        Ok(())
    }
}
impl Display for TransactionResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let status_line = match &self.status {
            TransactionStatus::Success => "Transaction succeeded".green().to_string(),
            TransactionStatus::Failure { error } => {
                format!("{}: {}", "Transaction failed".red(), error)
            }
        };

        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(f, "{}", status_line)?;
        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(f)?;

        // Transaction info
        writeln!(f, "{}: {}", "Digest".bold(), self.digest)?;
        writeln!(f, "{}: {}", "Executed Epoch".bold(), self.executed_epoch)?;
        if let Some(checkpoint) = self.checkpoint {
            writeln!(f, "{}: {}", "Checkpoint".bold(), checkpoint)?;
        }

        // Object changes
        if !self.created.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Created Objects:".bold())?;
            for obj in &self.created {
                write_owned_object_ref(f, obj)?;
            }
        }

        if !self.mutated.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Mutated Objects:".bold())?;
            for obj in &self.mutated {
                write_owned_object_ref(f, obj)?;
            }
        }

        if !self.deleted.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Deleted Objects:".bold())?;
            for obj in &self.deleted {
                write_object_ref(f, obj)?;
            }
        }

        // Gas object
        writeln!(f)?;
        writeln!(f, "{}", "Gas Object:".bold())?;
        write_owned_object_ref(f, &self.gas_object)?;

        // Transaction fee
        writeln!(f)?;
        writeln!(f, "{}", "Transaction Fee:".bold())?;
        writeln!(f, "  Base Fee:      {} shannons", self.fee.base_fee)?;
        writeln!(f, "  Operation Fee: {} shannons", self.fee.operation_fee)?;
        writeln!(f, "  Value Fee:     {} shannons", self.fee.value_fee)?;
        writeln!(f, "  {}: {} shannons", "Total".bold(), self.fee.total_fee)?;

        // Balance changes
        if !self.balance_changes.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Balance Changes:".bold())?;
            for change in &self.balance_changes {
                let amount_str = if change.amount >= 0 {
                    format!("+{}", change.amount).green().to_string()
                } else {
                    format!("{}", change.amount).red().to_string()
                };
                writeln!(f, "  {} : {} shannons", change.address, amount_str)?;
            }
        }

        Ok(())
    }
}

fn write_owned_object_ref(f: &mut Formatter<'_>, obj: &OwnedObjectRef) -> fmt::Result {
    writeln!(f, " ┌──")?;
    writeln!(f, " │ ID: {}", obj.object_id)?;
    writeln!(f, " │ Version: {}", obj.version.value())?;
    writeln!(f, " │ Digest: {}", obj.digest)?;
    match &obj.owner {
        OwnerDisplay::AddressOwner { address } => {
            writeln!(f, " │ Owner: {}", address)?;
        }
        OwnerDisplay::Shared {
            initial_shared_version,
        } => {
            writeln!(
                f,
                " │ Owner: Shared (initial version: {})",
                initial_shared_version.value()
            )?;
        }
        OwnerDisplay::Immutable => {
            writeln!(f, " │ Owner: Immutable")?;
        }
    }
    writeln!(f, " └──")
}

fn write_object_ref(f: &mut Formatter<'_>, obj: &ObjectRefDisplay) -> fmt::Result {
    writeln!(f, " ┌──")?;
    writeln!(f, " │ ID: {}", obj.object_id)?;
    writeln!(f, " │ Version: {}", obj.version.value())?;
    writeln!(f, " │ Digest: {}", obj.digest)?;
    writeln!(f, " └──")
}

/// Validator summary for display (extracted from SystemState)
#[derive(Debug, Clone, Serialize)]
pub struct ValidatorSummary {
    pub address: SomaAddress,
    pub status: ValidatorStatus,
    pub voting_power: u64,
    pub commission_rate: u64,
    // Network addresses (as strings for display)
    pub network_address: String,
    pub p2p_address: String,
    pub primary_address: String,
    pub encoder_validator_address: String,
    // Public keys (hex encoded for display)
    pub protocol_pubkey: String,
    pub network_pubkey: String,
    pub worker_pubkey: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum ValidatorStatus {
    /// Participating in consensus
    Consensus,
    /// Networking-only validator (not in consensus)
    Networking,
    /// Pending activation in next epoch
    Pending,
}

impl Display for ValidatorStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ValidatorStatus::Consensus => write!(f, "{}", "Consensus".green()),
            ValidatorStatus::Networking => write!(f, "{}", "Networking".cyan()),
            ValidatorStatus::Pending => write!(f, "{}", "Pending".yellow()),
        }
    }
}

impl Display for ValidatorSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "─".repeat(60))?;
        writeln!(f, "{}", "Validator Information".bold())?;
        writeln!(f, "{}", "─".repeat(60))?;
        writeln!(f)?;
        writeln!(f, "{}: {}", "Address".bold(), self.address)?;
        writeln!(f, "{}: {}", "Status".bold(), self.status)?;
        writeln!(f, "{}: {}", "Voting Power".bold(), self.voting_power)?;
        writeln!(
            f,
            "{}: {:.2}%",
            "Commission Rate".bold(),
            self.commission_rate as f64 / 100.0
        )?;
        writeln!(f)?;
        writeln!(f, "{}:", "Network Addresses".bold())?;
        writeln!(f, "  Network:   {}", self.network_address)?;
        writeln!(f, "  P2P:       {}", self.p2p_address)?;
        writeln!(f, "  Primary:   {}", self.primary_address)?;
        writeln!(f, "  Encoder:   {}", self.encoder_validator_address)?;

        writeln!(f)?;
        writeln!(f, "{}:", "Public Keys".bold())?;
        writeln!(f, "  Protocol: {}", truncate_key(&self.protocol_pubkey))?;
        writeln!(f, "  Network:  {}", truncate_key(&self.network_pubkey))?;
        writeln!(f, "  Worker:   {}", truncate_key(&self.worker_pubkey))?;

        Ok(())
    }
}

/// Truncate a hex key for display (show first 8 and last 8 chars)
fn truncate_key(key: &str) -> String {
    if key.len() <= 20 {
        key.to_string()
    } else {
        format!("{}...{}", &key[..8], &key[key.len() - 8..])
    }
}

impl ValidatorCommandResponse {
    /// Print the response with optional JSON formatting
    pub fn print(&self, json: bool) {
        if json {
            match serde_json::to_string_pretty(self) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Failed to serialize response: {}", e),
            }
        } else {
            print!("{}", self);
        }
    }
}
