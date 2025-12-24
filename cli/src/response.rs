use std::fmt::{self, Display, Formatter};

use colored::Colorize;
use fastcrypto::encoding::Encoding as _;
use rpc::TransactionExecutionResponseWithCheckpoint;
use sdk::client_config::SomaEnv;
use serde::Serialize;
use types::{
    balance_change::BalanceChange,
    base::SomaAddress,
    crypto::SignatureScheme,
    digests::{ObjectDigest, TransactionDigest},
    effects::{ExecutionStatus, TransactionEffects, TransactionEffectsAPI},
    object::{Object, ObjectID, ObjectRef, Owner, Version},
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

    pub fn from_response(response: &TransactionExecutionResponseWithCheckpoint) -> Self {
        let effects = &response.effects;

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
            checkpoint: Some(response.checkpoint_sequence_number),
            fee: effects.transaction_fee().clone(),
            created: effects.created().into_iter().map(Into::into).collect(),
            mutated: effects
                .mutated_excluding_gas()
                .into_iter()
                .map(Into::into)
                .collect(),
            deleted: effects.deleted().into_iter().map(Into::into).collect(),
            gas_object: effects.gas_object().into(),
            balance_changes: response.balance_changes.clone(),
        }
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

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ClientCommandResponse {
    // Address management
    ActiveAddress(Option<SomaAddress>),
    Addresses(AddressesOutput),
    NewAddress(NewAddressOutput),
    RemoveAddress(RemoveAddressOutput),
    Switch(SwitchOutput),

    // Environment management
    ActiveEnv(Option<String>),
    Envs(EnvsOutput),
    NewEnv(NewEnvOutput),

    // Chain info
    ChainInfo(ChainInfoOutput),

    // Object queries
    Object(ObjectOutput),
    Objects(ObjectsOutput),
    Gas(GasCoinsOutput),
    Balance(BalanceOutput),

    // Transaction results
    Transaction(TransactionResponse),
    TransactionDigest(TransactionDigest),
    SerializedUnsignedTransaction(String),
    SerializedSignedTransaction(String),

    // Simulation
    Simulation(SimulationResponse),
}

impl ClientCommandResponse {
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

impl Display for ClientCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            // Address management
            ClientCommandResponse::ActiveAddress(addr) => match addr {
                Some(a) => writeln!(f, "{}", a),
                None => writeln!(f, "{}", "No active address".yellow()),
            },
            ClientCommandResponse::Addresses(output) => write!(f, "{}", output),
            ClientCommandResponse::NewAddress(output) => write!(f, "{}", output),
            ClientCommandResponse::RemoveAddress(output) => write!(f, "{}", output),
            ClientCommandResponse::Switch(output) => write!(f, "{}", output),

            // Environment management
            ClientCommandResponse::ActiveEnv(env) => match env {
                Some(e) => writeln!(f, "{}", e),
                None => writeln!(f, "{}", "No active environment".yellow()),
            },
            ClientCommandResponse::Envs(output) => write!(f, "{}", output),
            ClientCommandResponse::NewEnv(output) => write!(f, "{}", output),

            // Chain info
            ClientCommandResponse::ChainInfo(output) => write!(f, "{}", output),

            // Object queries
            ClientCommandResponse::Object(output) => write!(f, "{}", output),
            ClientCommandResponse::Objects(output) => write!(f, "{}", output),
            ClientCommandResponse::Gas(output) => write!(f, "{}", output),
            ClientCommandResponse::Balance(output) => write!(f, "{}", output),

            // Transaction results
            ClientCommandResponse::Transaction(output) => write!(f, "{}", output),
            ClientCommandResponse::TransactionDigest(digest) => {
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)
            }
            ClientCommandResponse::SerializedUnsignedTransaction(bytes) => {
                writeln!(f, "{}", "Serialized Unsigned Transaction".cyan().bold())?;
                writeln!(f, "{}", bytes)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Use 'soma client execute-signed-tx --tx-bytes <bytes> --signatures <sigs>' to execute."
                        .yellow()
                )
            }
            ClientCommandResponse::SerializedSignedTransaction(bytes) => {
                writeln!(f, "{}", "Serialized Signed Transaction".cyan().bold())?;
                writeln!(f, "{}", bytes)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Use 'soma client execute-combined-signed-tx --signed-tx-bytes <bytes>' to execute."
                        .yellow()
                )
            }
            ClientCommandResponse::Simulation(output) => write!(f, "{}", output),
        }
    }
}

// =============================================================================
// ADDRESS MANAGEMENT OUTPUTS
// =============================================================================

#[derive(Debug, Serialize)]
pub struct AddressesOutput {
    pub active_address: SomaAddress,
    pub addresses: Vec<(String, SomaAddress)>,
}

impl Display for AddressesOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.addresses.is_empty() {
            return writeln!(f, "{}", "No addresses found".yellow());
        }

        writeln!(f, "{}", "Managed Addresses".bold())?;
        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(
            f,
            "{:<20} {:<46} {}",
            "Alias".bold(),
            "Address".bold(),
            "Active".bold()
        )?;
        writeln!(f, "{}", "─".repeat(70))?;

        for (alias, address) in &self.addresses {
            let active = if *address == self.active_address {
                "●".green().to_string()
            } else {
                " ".to_string()
            };
            writeln!(f, "{:<20} {:<46} {}", alias, address, active)?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct NewAddressOutput {
    pub alias: String,
    pub address: SomaAddress,
    pub key_scheme: SignatureScheme,
    pub recovery_phrase: String,
}

impl Display for NewAddressOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "New Address Created".green().bold())?;
        writeln!(f, "{}", "─".repeat(50))?;
        writeln!(f, "{:<18} {}", "Alias:".bold(), self.alias)?;
        writeln!(f, "{:<18} {}", "Address:".bold(), self.address)?;
        writeln!(f, "{:<18} {}", "Key Scheme:".bold(), self.key_scheme)?;
        writeln!(f)?;
        writeln!(f, "{}", "Recovery Phrase:".bold())?;
        writeln!(
            f,
            "{}",
            "┌────────────────────────────────────────────────────────────────┐"
        )?;
        writeln!(f, "│ {} │", self.recovery_phrase)?;
        writeln!(
            f,
            "└────────────────────────────────────────────────────────────────┘"
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "{}",
            "⚠  Store this recovery phrase securely. It cannot be recovered!"
                .yellow()
                .bold()
        )?;
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct RemoveAddressOutput {
    pub alias_or_address: String,
    pub address: SomaAddress,
}

impl Display for RemoveAddressOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} Removed address: {} ({})",
            "✓".green(),
            self.alias_or_address,
            self.address
        )
    }
}

#[derive(Debug, Serialize)]
pub struct SwitchOutput {
    pub address: Option<SomaAddress>,
    pub env: Option<String>,
}

impl Display for SwitchOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(addr) = &self.address {
            writeln!(f, "{} Active address switched to {}", "✓".green(), addr)?;
        }
        if let Some(env) = &self.env {
            writeln!(
                f,
                "{} Active environment switched to [{}]",
                "✓".green(),
                env
            )?;
        }
        Ok(())
    }
}

// =============================================================================
// ENVIRONMENT MANAGEMENT OUTPUTS
// =============================================================================

#[derive(Debug, Serialize)]
pub struct EnvsOutput {
    pub envs: Vec<SomaEnv>,
    pub active: Option<String>,
}

impl Display for EnvsOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.envs.is_empty() {
            return writeln!(f, "{}", "No environments configured".yellow());
        }

        writeln!(f, "{}", "Configured Environments".bold())?;
        writeln!(f, "{}", "─".repeat(70))?;
        writeln!(
            f,
            "{:<15} {:<50} {}",
            "Alias".bold(),
            "RPC URL".bold(),
            "Active".bold()
        )?;
        writeln!(f, "{}", "─".repeat(70))?;

        for env in &self.envs {
            let active = if Some(&env.alias) == self.active.as_ref() {
                "●".green().to_string()
            } else {
                " ".to_string()
            };
            let rpc = if env.rpc.len() > 48 {
                format!("{}...", &env.rpc[..45])
            } else {
                env.rpc.clone()
            };
            writeln!(f, "{:<15} {:<50} {}", env.alias, rpc, active)?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct NewEnvOutput {
    pub alias: String,
    pub chain_id: String,
}

impl Display for NewEnvOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} Added environment [{}]",
            "✓".green(),
            self.alias.cyan()
        )?;
        writeln!(f, "  Chain ID: {}", self.chain_id)?;
        Ok(())
    }
}

// =============================================================================
// CHAIN INFO OUTPUT
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ChainInfoOutput {
    pub chain_id: String,
    pub server_version: Option<String>,
}

impl Display for ChainInfoOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}: {}", "Chain ID".bold(), self.chain_id)?;
        if let Some(version) = &self.server_version {
            writeln!(f, "{}: {}", "Server Version".bold(), version)?;
        }
        Ok(())
    }
}

// =============================================================================
// OBJECT QUERY OUTPUTS
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ObjectOutput {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub object_type: String,
    pub owner: Option<OwnerDisplay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bcs_bytes: Option<String>,
}

impl ObjectOutput {
    // TODO: maybe we can use the fact we can deserialize the specific types to display the values of those types (only 4 object types)
    pub fn from_object(obj: &Object, include_bcs: bool) -> Self {
        Self {
            object_id: obj.id(),
            version: obj.version(),
            digest: obj.digest(),
            object_type: obj.type_().to_string(),
            owner: Some(OwnerDisplay::from(obj.owner.clone())),
            bcs_bytes: if include_bcs {
                Some(fastcrypto::encoding::Base64::encode(&obj.to_bytes()))
            } else {
                None
            },
        }
    }
}

impl Display for ObjectOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "Object".bold())?;
        writeln!(f, "{}", "─".repeat(50))?;
        writeln!(f, "{:<15} {}", "ID:".bold(), self.object_id)?;
        writeln!(f, "{:<15} {}", "Version:".bold(), self.version.value())?;
        writeln!(f, "{:<15} {}", "Digest:".bold(), self.digest)?;
        writeln!(f, "{:<15} {}", "Type:".bold(), self.object_type)?;
        if let Some(owner) = &self.owner {
            writeln!(f, "{:<15} {}", "Owner:".bold(), owner)?;
        }
        if let Some(bcs) = &self.bcs_bytes {
            writeln!(f)?;
            writeln!(f, "{}", "BCS Bytes:".bold())?;
            writeln!(f, "{}", bcs)?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ObjectsOutput {
    pub address: SomaAddress,
    pub objects: Vec<ObjectOutput>,
}

impl Display for ObjectsOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} objects owned by {}",
            self.objects.len(),
            self.address
        )?;
        writeln!(f, "{}", "─".repeat(90))?;

        if self.objects.is_empty() {
            writeln!(f, "{}", "No objects found".yellow())?;
            return Ok(());
        }

        writeln!(
            f,
            "{:<44} {:<10} {:<30}",
            "Object ID".bold(),
            "Version".bold(),
            "Type".bold()
        )?;
        writeln!(f, "{}", "─".repeat(90))?;

        for obj in &self.objects {
            let type_str = if obj.object_type.len() > 28 {
                format!("{}...", &obj.object_type[..25])
            } else {
                obj.object_type.clone()
            };
            writeln!(
                f,
                "{:<44} {:<10} {:<30}",
                obj.object_id,
                obj.version.value(),
                type_str
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct GasCoinsOutput {
    pub address: SomaAddress,
    pub coins: Vec<(ObjectRef, u64)>,
}

impl Display for GasCoinsOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} gas coins owned by {}",
            self.coins.len(),
            self.address
        )?;
        writeln!(f, "{}", "─".repeat(80))?;

        if self.coins.is_empty() {
            writeln!(f, "{}", "No gas coins found".yellow())?;
            return Ok(());
        }

        writeln!(
            f,
            "{:<44} {:<20} {:<15}",
            "Object ID".bold(),
            "Balance (SHNS)".bold(),
            "SOMA".bold()
        )?;
        writeln!(f, "{}", "─".repeat(80))?;

        let mut total: u128 = 0;
        for (obj_ref, balance) in &self.coins {
            total += *balance as u128;
            writeln!(
                f,
                "{:<44} {:<20} {:<15}",
                obj_ref.0,
                balance,
                format_soma(*balance as u128)
            )?;
        }

        writeln!(f, "{}", "─".repeat(80))?;
        writeln!(
            f,
            "{:<44} {:<20} {}",
            "Total".bold(),
            total,
            format_soma(total).bold()
        )?;

        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct BalanceOutput {
    pub address: SomaAddress,
    pub total_balance: u128,
    pub coin_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coins: Option<Vec<(ObjectID, u64)>>,
}

impl Display for BalanceOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "Balance".bold())?;
        writeln!(f, "{}", "─".repeat(50))?;
        writeln!(f, "{:<15} {}", "Address:".bold(), self.address)?;
        writeln!(
            f,
            "{:<15} {} SOMA",
            "Total:".bold(),
            format_soma(self.total_balance).green()
        )?;
        writeln!(f, "{:<15} {} shannons", "Raw:".bold(), self.total_balance)?;
        writeln!(f, "{:<15} {} coin(s)", "Coins:".bold(), self.coin_count)?;

        if let Some(coins) = &self.coins {
            writeln!(f)?;
            writeln!(f, "{}", "Individual Coins:".bold())?;
            for (id, balance) in coins {
                writeln!(f, "  {} : {} shannons", id, balance)?;
            }
        }

        Ok(())
    }
}

// =============================================================================
// SIMULATION RESPONSE
// =============================================================================

#[derive(Debug, Serialize)]
pub struct SimulationResponse {
    pub status: TransactionStatus,
    pub gas_used: u64,
    pub created: Vec<OwnedObjectRef>,
    pub mutated: Vec<OwnedObjectRef>,
    pub deleted: Vec<ObjectRefDisplay>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub balance_changes: Vec<BalanceChange>,
}

impl Display for SimulationResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "Simulation Result".cyan().bold())?;
        writeln!(f, "{}", "─".repeat(50))?;

        let status_str = match &self.status {
            TransactionStatus::Success => "Would Succeed".green().to_string(),
            TransactionStatus::Failure { error } => {
                format!("{}: {}", "Would Fail".red(), error)
            }
        };
        writeln!(f, "{}: {}", "Status".bold(), status_str)?;
        writeln!(
            f,
            "{}: {} shannons ({})",
            "Estimated Gas".bold(),
            self.gas_used,
            format_soma(self.gas_used as u128)
        )?;

        if !self.created.is_empty() {
            writeln!(f, "{}: {}", "Would Create".bold(), self.created.len())?;
        }
        if !self.mutated.is_empty() {
            writeln!(f, "{}: {}", "Would Mutate".bold(), self.mutated.len())?;
        }
        if !self.deleted.is_empty() {
            writeln!(f, "{}: {}", "Would Delete".bold(), self.deleted.len())?;
        }

        Ok(())
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Format a balance in shannons as SOMA with appropriate suffix
fn format_soma(shannons: u128) -> String {
    const SOMA: u128 = 1_000_000_000; // TODO: 10^9 shannons = 1 SOMA

    if shannons == 0 {
        return "0 SOMA".to_string();
    }

    let whole = shannons / SOMA;
    let frac = shannons % SOMA;

    if whole >= 1_000_000_000 {
        format!("{:.2}B SOMA", whole as f64 / 1_000_000_000.0)
    } else if whole >= 1_000_000 {
        format!("{:.2}M SOMA", whole as f64 / 1_000_000.0)
    } else if whole >= 1_000 {
        format!("{:.2}K SOMA", whole as f64 / 1_000.0)
    } else if whole > 0 {
        if frac > 0 {
            format!("{}.{:02} SOMA", whole, frac / 10_000_000)
        } else {
            format!("{} SOMA", whole)
        }
    } else {
        // Less than 1 SOMA
        format!("0.{:09} SOMA", frac)
    }
}
