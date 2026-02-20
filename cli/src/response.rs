use std::fmt::{self, Display, Formatter};

use colored::Colorize;
use fastcrypto::encoding::Encoding as _;
use rpc::TransactionExecutionResponseWithCheckpoint;
use sdk::client_config::SomaEnv;
use serde::Serialize;
use tabled::{
    builder::Builder as TableBuilder,
    settings::{
        Alignment as TableAlignment, Border as TableBorder, Modify as TableModify,
        Panel as TablePanel, Style as TableStyle,
        object::{Columns as TableCols, Rows as TableRows},
        style::HorizontalLine,
    },
};
use types::{
    balance_change::BalanceChange,
    base::SomaAddress,
    crypto::SignatureScheme,
    digests::{ObjectDigest, TransactionDigest},
    effects::{ExecutionStatus, TransactionEffects, TransactionEffectsAPI},
    object::{Object, ObjectID, ObjectRef, ObjectType, Owner, Version},
    system_state::staking::StakedSomaV1,
    tx_fee::TransactionFee,
};

// =============================================================================
// CONSTANTS
// =============================================================================

const SHANNONS_PER_SOMA: u128 = 1_000_000_000;

// =============================================================================
// VALIDATOR COMMAND RESPONSE
// =============================================================================

pub use crate::commands::validator::{MakeValidatorInfoOutput, DisplayMetadataOutput};

#[derive(Debug, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ValidatorCommandResponse {
    Started,
    MakeValidatorInfo(MakeValidatorInfoOutput),
    DisplayMetadata(DisplayMetadataOutput),
    List(ValidatorListOutput),
    Transaction(TransactionResponse),
    SerializedTransaction { serialized_transaction: String },
    TransactionDigest(TransactionDigest),
    Simulation(SimulationResponse),
}

impl Display for ValidatorCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ValidatorCommandResponse::MakeValidatorInfo(output) => {
                writeln!(f, "{}", "Validator info files created successfully.".green().bold())?;
                writeln!(f)?;

                let mut builder = TableBuilder::default();
                builder.push_record(["Generated Files"]);
                for file in &output.files {
                    builder.push_record([file.as_str()]);
                }

                let mut table = builder.build();
                table.with(TableStyle::rounded());
                table.with(TableModify::new(TableRows::first()).with(TableAlignment::center()));
                write!(f, "{}", table)?;
                writeln!(f)?;
                writeln!(f, "  Output directory: {}", output.output_dir)?;
            }
            ValidatorCommandResponse::DisplayMetadata(output) => {
                match (&output.status, &output.summary) {
                    (Some(status), Some(summary)) => {
                        writeln!(f, "{}'s validator status: {}", output.address, status)?;
                        write!(f, "{}", summary)?;
                    }
                    _ => {
                        writeln!(f, "{} is not an active, networking, or pending validator.", output.address)?;
                    }
                }
            }
            ValidatorCommandResponse::Started => {}
            ValidatorCommandResponse::List(output) => {
                write!(f, "{}", output)?;
            }
            ValidatorCommandResponse::Transaction(tx_response) => {
                write!(f, "{}", tx_response)?;
            }
            ValidatorCommandResponse::SerializedTransaction { serialized_transaction } => {
                writeln!(f, "{}", "Serialized Unsigned Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", serialized_transaction)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "→ Use 'soma client execute-signed-tx' to submit after signing.".yellow()
                )?;
            }
            ValidatorCommandResponse::TransactionDigest(digest) => {
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)?;
            }
            ValidatorCommandResponse::Simulation(sim) => write!(f, "{}", sim)?,
        }
        Ok(())
    }
}

impl ValidatorCommandResponse {
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

// =============================================================================
// TRANSACTION RESPONSE
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct TransactionResponse {
    pub digest: TransactionDigest,
    pub status: TransactionStatus,
    pub executed_epoch: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<u64>,
    pub fee: TransactionFee,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub created: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mutated: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deleted: Vec<ObjectRefDisplay>,
    pub gas_object: OwnedObjectRef,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub balance_changes: Vec<BalanceChange>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OwnedObjectRef {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub owner: OwnerDisplay,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OwnerDisplay {
    AddressOwner { address: SomaAddress },
    Shared { initial_shared_version: Version },
    Immutable,
}

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
            Owner::Shared { initial_shared_version } => {
                OwnerDisplay::Shared { initial_shared_version }
            }
            Owner::Immutable => OwnerDisplay::Immutable,
        }
    }
}

impl From<(ObjectRef, Owner)> for OwnedObjectRef {
    fn from((obj_ref, owner): (ObjectRef, Owner)) -> Self {
        Self { object_id: obj_ref.0, version: obj_ref.1, digest: obj_ref.2, owner: owner.into() }
    }
}

impl From<ObjectRef> for ObjectRefDisplay {
    fn from(obj_ref: ObjectRef) -> Self {
        Self { object_id: obj_ref.0, version: obj_ref.1, digest: obj_ref.2 }
    }
}

impl TransactionResponse {
    pub fn from_effects(effects: &TransactionEffects, checkpoint: Option<u64>) -> Self {
        let status = match effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => {
                TransactionStatus::Failure { error: format!("{}", error) }
            }
        };

        Self {
            digest: effects.transaction_digest_owned(),
            status,
            executed_epoch: effects.executed_epoch(),
            checkpoint,
            fee: effects.transaction_fee().clone(),
            created: effects.created().into_iter().map(Into::into).collect(),
            mutated: effects.mutated_excluding_gas().into_iter().map(Into::into).collect(),
            deleted: effects.deleted().into_iter().map(Into::into).collect(),
            gas_object: effects.gas_object().into(),
            balance_changes: Vec::new(),
        }
    }

    pub fn from_effects_with_balance_changes(
        effects: &TransactionEffects,
        checkpoint: Option<u64>,
        balance_changes: Vec<BalanceChange>,
    ) -> Self {
        let mut response = Self::from_effects(effects, checkpoint);
        response.balance_changes = balance_changes;
        response
    }

    pub fn with_balance_changes(mut self, balance_changes: Vec<BalanceChange>) -> Self {
        self.balance_changes = balance_changes;
        self
    }

    pub fn from_response(response: &TransactionExecutionResponseWithCheckpoint) -> Self {
        let effects = &response.effects;
        let status = match effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => {
                TransactionStatus::Failure { error: format!("{}", error) }
            }
        };

        Self {
            digest: effects.transaction_digest_owned(),
            status,
            executed_epoch: effects.executed_epoch(),
            checkpoint: Some(response.checkpoint_sequence_number),
            fee: effects.transaction_fee().clone(),
            created: effects.created().into_iter().map(Into::into).collect(),
            mutated: effects.mutated_excluding_gas().into_iter().map(Into::into).collect(),
            deleted: effects.deleted().into_iter().map(Into::into).collect(),
            gas_object: effects.gas_object().into(),
            balance_changes: response.balance_changes.clone(),
        }
    }
}

impl Display for TransactionResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Status header
        let (status_icon, status_text) = match &self.status {
            TransactionStatus::Success => ("✓".green(), "Transaction Succeeded".green().bold()),
            TransactionStatus::Failure { error } => {
                writeln!(f, "{} {}", "✗".red(), "Transaction Failed".red().bold())?;
                writeln!(f, "  Error: {}", error)?;
                writeln!(f)?;
                return self.fmt_details(f);
            }
        };

        writeln!(f, "{} {}", status_icon, status_text)?;
        writeln!(f)?;
        self.fmt_details(f)
    }
}

impl TransactionResponse {
    fn fmt_details(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Transaction Info Table
        let mut builder = TableBuilder::default();
        builder.push_record(["Transaction Details", ""]);
        builder.push_record(["Digest", &self.digest.to_string()]);
        builder.push_record(["Epoch", &self.executed_epoch.to_string()]);
        if let Some(checkpoint) = self.checkpoint {
            builder.push_record(["Checkpoint", &checkpoint.to_string()]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(TableModify::new(TableRows::first()).with(TableAlignment::center()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        // Object Changes
        if !self.created.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Created Objects ({})", "●".green(), self.created.len())?;
            self.fmt_object_refs(f, &self.created)?;
        }

        if !self.mutated.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Mutated Objects ({})", "●".yellow(), self.mutated.len())?;
            self.fmt_object_refs(f, &self.mutated)?;
        }

        if !self.deleted.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Deleted Objects ({})", "●".red(), self.deleted.len())?;
            self.fmt_deleted_refs(f, &self.deleted)?;
        }

        // Gas Summary
        writeln!(f)?;
        let mut builder = TableBuilder::default();
        builder.push_record(["Gas Summary", ""]);
        builder.push_record(["Base Fee", &format_fee(self.fee.base_fee)]);
        builder.push_record(["Operation Fee", &format_fee(self.fee.operation_fee)]);
        builder.push_record(["Value Fee", &format_fee(self.fee.value_fee)]);
        builder.push_record(["Total", &format_fee(self.fee.total_fee)]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(TableModify::new(TableRows::first()).with(TableAlignment::center()));
        table.with(TableModify::new(TableCols::last()).with(TableAlignment::right()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        // Balance Changes
        if !self.balance_changes.is_empty() {
            writeln!(f)?;
            let mut builder = TableBuilder::default();
            builder.push_record(["Address", "Change"]);

            for change in &self.balance_changes {
                let amount_str = if change.amount >= 0 {
                    format!("+{}", format_soma(change.amount as u128)).green().to_string()
                } else {
                    format!("-{}", format_soma(change.amount.unsigned_abs() as u128))
                        .red()
                        .to_string()
                };
                builder.push_record([change.address.to_string(), amount_str]);
            }

            let mut table = builder.build();
            table.with(TableStyle::rounded());
            table.with(TablePanel::header("Balance Changes"));
            table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
            table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
            table.with(TableModify::new(TableCols::last()).with(TableAlignment::right()));
            table.with(tabled::settings::style::BorderSpanCorrection);
            writeln!(f, "{}", table)?;
        }

        Ok(())
    }

    pub fn fmt_object_refs(&self, f: &mut Formatter<'_>, refs: &[OwnedObjectRef]) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record(["Object ID", "Version", "Owner"]);

        for obj in refs {
            let owner_str = match &obj.owner {
                OwnerDisplay::AddressOwner { address } => truncate_address(&address.to_string()),
                OwnerDisplay::Shared { initial_shared_version } => {
                    format!("Shared (v{})", initial_shared_version.value())
                }
                OwnerDisplay::Immutable => "Immutable".to_string(),
            };
            builder.push_record([
                obj.object_id.to_string(),
                obj.version.value().to_string(),
                owner_str,
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        writeln!(f, "{}", table)
    }

    pub fn fmt_deleted_refs(
        &self,
        f: &mut Formatter<'_>,
        refs: &[ObjectRefDisplay],
    ) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record(["Object ID", "Version"]);

        for obj in refs {
            builder.push_record([obj.object_id.to_string(), obj.version.value().to_string()]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        writeln!(f, "{}", table)
    }
}

// =============================================================================
// CLIENT COMMAND RESPONSE
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ClientCommandResponse {
    ActiveAddress(Option<ActiveAddressOutput>),
    Addresses(AddressesOutput),
    NewAddress(NewAddressOutput),
    RemoveAddress(RemoveAddressOutput),
    Switch(SwitchOutput),
    ActiveEnv(Option<String>),
    Envs(EnvsOutput),
    NewEnv(NewEnvOutput),
    ChainInfo(ChainInfoOutput),
    Object(ObjectOutput),
    Objects(ObjectsOutput),
    Gas(GasCoinsOutput),
    Balance(BalanceOutput),
    Transaction(TransactionResponse),
    TransactionDigest(TransactionDigest),
    SerializedUnsignedTransaction(String),
    SerializedSignedTransaction(String),
    Simulation(SimulationResponse),
    TransactionQuery(TransactionQueryResponse),
    NoOutput,
}

impl ClientCommandResponse {
    pub fn print(&self, json: bool) {
        if matches!(self, ClientCommandResponse::NoOutput) {
            return;
        }
        if json {
            match serde_json::to_string_pretty(self) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Failed to serialize response: {}", e),
            }
        } else {
            print!("{}", self);
        }
    }

    /// Returns true if this response contains a failed on-chain transaction.
    pub fn has_failed_transaction(&self) -> bool {
        match self {
            ClientCommandResponse::Transaction(tx) => !tx.status.is_success(),
            _ => false,
        }
    }
}

impl Display for ClientCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ClientCommandResponse::ActiveAddress(output) => match output {
                Some(out) => match &out.alias {
                    Some(alias) => writeln!(f, "{} ({})", alias, out.address),
                    None => writeln!(f, "{}", out.address),
                },
                None => writeln!(f, "{}", "No active address set".yellow()),
            },
            ClientCommandResponse::Addresses(output) => write!(f, "{}", output),
            ClientCommandResponse::NewAddress(output) => write!(f, "{}", output),
            ClientCommandResponse::RemoveAddress(output) => write!(f, "{}", output),
            ClientCommandResponse::Switch(output) => write!(f, "{}", output),
            ClientCommandResponse::ActiveEnv(env) => match env {
                Some(e) => writeln!(f, "{}", e),
                None => writeln!(f, "{}", "No active environment set".yellow()),
            },
            ClientCommandResponse::Envs(output) => write!(f, "{}", output),
            ClientCommandResponse::NewEnv(output) => write!(f, "{}", output),
            ClientCommandResponse::ChainInfo(output) => write!(f, "{}", output),
            ClientCommandResponse::Object(output) => write!(f, "{}", output),
            ClientCommandResponse::Objects(output) => write!(f, "{}", output),
            ClientCommandResponse::Gas(output) => write!(f, "{}", output),
            ClientCommandResponse::Balance(output) => write!(f, "{}", output),
            ClientCommandResponse::Transaction(output) => write!(f, "{}", output),
            ClientCommandResponse::TransactionDigest(digest) => {
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)
            }
            ClientCommandResponse::SerializedUnsignedTransaction(bytes) => {
                writeln!(f, "{}", "Serialized Unsigned Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", bytes)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "→ Use 'soma client execute-signed-tx --tx-bytes <bytes> --signatures <sigs>' to execute."
                        .yellow()
                )
            }
            ClientCommandResponse::SerializedSignedTransaction(bytes) => {
                writeln!(f, "{}", "Serialized Signed Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", bytes)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "→ Use 'soma client execute-combined-signed-tx --signed-tx-bytes <bytes>' to execute."
                        .yellow()
                )
            }
            ClientCommandResponse::Simulation(output) => write!(f, "{}", output),
            ClientCommandResponse::TransactionQuery(output) => write!(f, "{}", output),
            ClientCommandResponse::NoOutput => Ok(()),
        }
    }
}

// =============================================================================
// ACTIVE ADDRESS OUTPUT
// =============================================================================

#[derive(Debug)]
pub struct ActiveAddressOutput {
    pub address: SomaAddress,
    pub alias: Option<String>,
}

impl Serialize for ActiveAddressOutput {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.address.serialize(serializer)
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
            return writeln!(
                f,
                "{}",
                "No addresses found. Use 'soma wallet new' to create one.".yellow()
            );
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["", "Alias", "Address"]);

        for (alias, address) in &self.addresses {
            let active = if *address == self.active_address {
                "●".green().to_string()
            } else {
                " ".to_string()
            };
            builder.push_record([active, alias.clone(), address.to_string()]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Managed Addresses"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)?;
        writeln!(f)?;
        writeln!(f, "{} = active address", "●".green())
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
        writeln!(f, "{}", "✓ New Address Created".green().bold())?;
        writeln!(f)?;

        let mut builder = TableBuilder::default();
        builder.push_record(["Alias", &self.alias]);
        builder.push_record(["Address", &self.address.to_string()]);
        builder.push_record(["Key Scheme", &self.key_scheme.to_string()]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        writeln!(f, "{}", table)?;

        writeln!(f)?;
        writeln!(f, "{}", "Recovery Phrase".bold())?;
        writeln!(f, "╭{}╮", "─".repeat(66))?;
        writeln!(f, "│ {:<64} │", self.recovery_phrase)?;
        writeln!(f, "╰{}╯", "─".repeat(66))?;
        writeln!(f)?;
        writeln!(
            f,
            "{}",
            "⚠  Store this recovery phrase securely. It cannot be recovered!".yellow().bold()
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  Run {} to make it active.",
            format!("soma wallet switch {}", self.address).bold()
        )
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
            self.alias_or_address.cyan(),
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
            writeln!(f, "{} Active address → {}", "✓".green(), addr)?;
        }
        if let Some(env) = &self.env {
            writeln!(f, "{} Active environment → [{}]", "✓".green(), env.cyan())?;
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
            return writeln!(
                f,
                "{}",
                "No environments configured. Use 'soma env new' to add one.".yellow()
            );
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["", "Alias", "RPC URL"]);

        for env in &self.envs {
            let active = if Some(&env.alias) == self.active.as_ref() {
                "●".green().to_string()
            } else {
                " ".to_string()
            };
            let rpc = truncate_string(&env.rpc, 50);
            builder.push_record([active, env.alias.clone(), rpc]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Configured Environments"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)?;
        writeln!(f)?;
        writeln!(f, "{} = active environment", "●".green())
    }
}

#[derive(Debug, Serialize)]
pub struct NewEnvOutput {
    pub alias: String,
    pub chain_id: String,
}

impl Display for NewEnvOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Added environment [{}]", "✓".green(), self.alias.cyan())?;
        writeln!(f, "  Chain ID: {}", self.chain_id)?;
        writeln!(f)?;
        writeln!(f, "  Run {} to activate it.", format!("soma env switch {}", self.alias).bold())
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
        let mut builder = TableBuilder::default();
        builder.push_record(["Chain ID", &self.chain_id]);
        if let Some(version) = &self.server_version {
            builder.push_record(["Server Version", version]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Chain Information"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)
    }
}

// =============================================================================
// OBJECT QUERY OUTPUTS
// =============================================================================

/// Deserialized object content for display
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectContent {
    Coin { balance: u64 },
    StakedSoma(StakedSomaDisplay),
    SystemState,
    Target,
    Challenge,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct StakedSomaDisplay {
    pub pool_id: ObjectID,
    pub stake_activation_epoch: u64,
    pub principal: u64,
}

#[derive(Debug, Serialize)]
pub struct ObjectOutput {
    pub object_id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub object_type: String,
    pub owner: Option<OwnerDisplay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ObjectContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bcs_bytes: Option<String>,
}

impl ObjectOutput {
    pub fn from_object(obj: &Object, include_bcs: bool) -> Self {
        let content = Self::extract_content(obj);

        Self {
            object_id: obj.id(),
            version: obj.version(),
            digest: obj.digest(),
            object_type: obj.type_().to_string(),
            owner: Some(OwnerDisplay::from(obj.owner.clone())),
            content,
            bcs_bytes: if include_bcs {
                Some(fastcrypto::encoding::Base64::encode(obj.data.contents()))
            } else {
                None
            },
        }
    }

    fn extract_content(obj: &Object) -> Option<ObjectContent> {
        match obj.type_() {
            ObjectType::Coin => obj.as_coin().map(|balance| ObjectContent::Coin { balance }),
            ObjectType::StakedSoma => obj.as_staked_soma().map(|s| {
                ObjectContent::StakedSoma(StakedSomaDisplay {
                    pool_id: s.pool_id,
                    stake_activation_epoch: s.stake_activation_epoch,
                    principal: s.principal,
                })
            }),
            ObjectType::SystemState => Some(ObjectContent::SystemState),
            ObjectType::Target => Some(ObjectContent::Target),
            ObjectType::Challenge => Some(ObjectContent::Challenge),
        }
    }
}

impl Display for ObjectOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record(["Object ID", &self.object_id.to_string()]);
        builder.push_record(["Version", &self.version.value().to_string()]);
        builder.push_record(["Digest", &self.digest.to_string()]);
        builder.push_record(["Type", &self.object_type]);

        if let Some(owner) = &self.owner {
            let owner_str = match owner {
                OwnerDisplay::AddressOwner { address } => address.to_string(),
                OwnerDisplay::Shared { initial_shared_version } => {
                    format!("Shared (initial version: {})", initial_shared_version.value())
                }
                OwnerDisplay::Immutable => "Immutable".to_string(),
            };
            builder.push_record(["Owner", &owner_str]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Object"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        // Content-specific display
        if let Some(content) = &self.content {
            writeln!(f)?;
            match content {
                ObjectContent::Coin { balance } => {
                    let mut builder = TableBuilder::default();
                    builder.push_record(["Balance (shannons)", &balance.to_string()]);
                    builder.push_record(["Balance (SOMA)", &format_soma(*balance as u128)]);

                    let mut table = builder.build();
                    table.with(TableStyle::rounded());
                    table.with(TablePanel::header("Coin Data"));
                    table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
                    table.with(tabled::settings::style::BorderSpanCorrection);
                    writeln!(f, "{}", table)?;
                }
                ObjectContent::StakedSoma(staked) => {
                    let mut builder = TableBuilder::default();
                    builder.push_record(["Pool ID", &staked.pool_id.to_string()]);
                    builder.push_record(["Principal (shannons)", &staked.principal.to_string()]);
                    builder
                        .push_record(["Principal (SOMA)", &format_soma(staked.principal as u128)]);
                    builder.push_record([
                        "Activation Epoch",
                        &staked.stake_activation_epoch.to_string(),
                    ]);

                    let mut table = builder.build();
                    table.with(TableStyle::rounded());
                    table.with(TablePanel::header("Staked SOMA Data"));
                    table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
                    table.with(tabled::settings::style::BorderSpanCorrection);
                    writeln!(f, "{}", table)?;
                }

                ObjectContent::SystemState => {
                    writeln!(
                        f,
                        "{}",
                        "System State object (use specialized queries for details)".dimmed()
                    )?;
                }
                ObjectContent::Target => {
                    writeln!(
                        f,
                        "{}",
                        "Target object (use 'soma target info <id>' for details)".dimmed()
                    )?;
                }
                ObjectContent::Challenge => {
                    writeln!(
                        f,
                        "{}",
                        "Challenge object (use 'soma challenge info <id>' for details)".dimmed()
                    )?;
                }
                ObjectContent::Unknown => {}
            }
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
        if self.objects.is_empty() {
            return writeln!(f, "{}", format!("No objects owned by {}", self.address).yellow());
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["Object ID", "Type", "Version"]);

        for obj in &self.objects {
            let type_str = truncate_string(&obj.object_type, 25);
            builder.push_record([
                obj.object_id.to_string(),
                type_str,
                obj.version.value().to_string(),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header(format!(
            "Objects owned by {} ({} total)",
            truncate_address(&self.address.to_string()),
            self.objects.len()
        )));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)
    }
}

#[derive(Debug, Serialize)]
pub struct GasCoinsOutput {
    pub address: SomaAddress,
    pub coins: Vec<(ObjectRef, u64)>,
}

impl Display for GasCoinsOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.coins.is_empty() {
            return writeln!(f, "{}", format!("No gas coins owned by {}", self.address).yellow());
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["Object ID", "Balance (shannons)", "Balance (SOMA)"]);

        let mut total: u128 = 0;
        for (obj_ref, balance) in &self.coins {
            total += *balance as u128;
            builder.push_record([
                obj_ref.0.to_string(),
                balance.to_string(),
                format_soma(*balance as u128),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header(format!("Gas Coins ({} coins)", self.coins.len())));
        table.with(TablePanel::footer(format!(
            "Total: {} shannons ({})",
            format_with_commas(total),
            format_soma(total)
        )));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(TableModify::new(TableCols::new(1..)).with(TableAlignment::right()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)
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
        let mut builder = TableBuilder::default();
        builder.push_record(["Address", &self.address.to_string()]);
        builder.push_record(["Total (SOMA)", &format_soma(self.total_balance).green().to_string()]);
        builder.push_record(["Total (shannons)", &self.total_balance.to_string()]);
        builder.push_record(["Coin Count", &self.coin_count.to_string()]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Balance"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        if let Some(coins) = &self.coins {
            writeln!(f)?;
            let mut builder = TableBuilder::default();
            builder.push_record(["Object ID", "Balance (shannons)", "Balance (SOMA)"]);

            for (id, balance) in coins {
                builder.push_record([
                    id.to_string(),
                    balance.to_string(),
                    format_soma(*balance as u128),
                ]);
            }

            let mut table = builder.build();
            table.with(TableStyle::rounded());
            table.with(TablePanel::header("Individual Coins"));
            table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
            table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
            table.with(TableModify::new(TableCols::last()).with(TableAlignment::right()));
            table.with(tabled::settings::style::BorderSpanCorrection);
            writeln!(f, "{}", table)?;
        }

        Ok(())
    }
}

impl BalanceOutput {
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

// =============================================================================
// STATUS OUTPUT
// =============================================================================

#[derive(Debug, Serialize)]
pub struct StatusOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<String>,
    pub rpc_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance: Option<u64>,
    pub server_reachable: bool,
}

impl Display for StatusOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record([
            "Network",
            self.network.as_deref().unwrap_or("none"),
        ]);
        builder.push_record(["RPC URL", &self.rpc_url]);
        if let Some(ref ver) = self.server_version {
            builder.push_record(["Server Version", ver]);
        }
        if let Some(ref cid) = self.chain_id {
            builder.push_record(["Chain ID", cid]);
        }
        if let Some(e) = self.epoch {
            builder.push_record(["Current Epoch", &e.to_string()]);
        }
        if let Some(ref addr) = self.active_address {
            builder.push_record(["Active Address", addr]);
        }
        if let Some(bal) = self.balance {
            let soma = format_soma(bal as u128);
            builder.push_record(["Balance", &soma]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Soma Network Status"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        if !self.server_reachable {
            writeln!(
                f,
                "\n  {} Could not connect to RPC server at {}",
                "Warning:".yellow().bold(),
                self.rpc_url,
            )?;
        }
        Ok(())
    }
}

impl StatusOutput {
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
        writeln!(f)?;

        let status_str = match &self.status {
            TransactionStatus::Success => "Would Succeed".green().to_string(),
            TransactionStatus::Failure { error } => format!("{}: {}", "Would Fail".red(), error),
        };

        let mut builder = TableBuilder::default();
        builder.push_record(["Status", &status_str]);
        builder.push_record([
            "Estimated Gas",
            &format!("{} shannons ({})", self.gas_used, format_soma(self.gas_used as u128)),
        ]);

        if !self.created.is_empty() {
            builder.push_record(["Would Create", &format!("{} object(s)", self.created.len())]);
        }
        if !self.mutated.is_empty() {
            builder.push_record(["Would Mutate", &format!("{} object(s)", self.mutated.len())]);
        }
        if !self.deleted.is_empty() {
            builder.push_record(["Would Delete", &format!("{} object(s)", self.deleted.len())]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        writeln!(f, "{}", table)
    }
}

// =============================================================================
// TRANSACTION QUERY RESPONSE
// =============================================================================

#[derive(Debug, Serialize)]
pub struct TransactionQueryResponse {
    pub digest: TransactionDigest,
    pub status: TransactionStatus,
    pub executed_epoch: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    pub fee: TransactionFee,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub created: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mutated: Vec<OwnedObjectRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deleted: Vec<ObjectRefDisplay>,
    pub gas_object: OwnedObjectRef,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub balance_changes: Vec<BalanceChange>,
}

impl TransactionQueryResponse {
    pub fn from_query_result(result: &rpc::api::client::TransactionQueryResult) -> Self {
        let effects = &result.effects;
        let status = match effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => {
                TransactionStatus::Failure { error: format!("{}", error) }
            }
        };

        let timestamp = result.timestamp_ms.map(|ms| {
            chrono::DateTime::from_timestamp_millis(ms as i64)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| format!("{}ms", ms))
        });

        Self {
            digest: result.digest,
            status,
            executed_epoch: effects.executed_epoch(),
            checkpoint: result.checkpoint,
            timestamp,
            fee: effects.transaction_fee().clone(),
            created: effects.created().into_iter().map(Into::into).collect(),
            mutated: effects.mutated_excluding_gas().into_iter().map(Into::into).collect(),
            deleted: effects.deleted().into_iter().map(Into::into).collect(),
            gas_object: effects.gas_object().into(),
            balance_changes: result.balance_changes.clone(),
        }
    }
}

impl Display for TransactionQueryResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (status_icon, status_text) = match &self.status {
            TransactionStatus::Success => ("✓".green(), "Transaction Succeeded".green().bold()),
            TransactionStatus::Failure { error } => {
                writeln!(f, "{} {}", "✗".red(), "Transaction Failed".red().bold())?;
                writeln!(f, "  Error: {}", error)?;
                writeln!(f)?;
                return self.fmt_details(f);
            }
        };

        writeln!(f, "{} {}", status_icon, status_text)?;
        writeln!(f)?;
        self.fmt_details(f)
    }
}

impl TransactionQueryResponse {
    fn fmt_details(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record(["Transaction Details", ""]);
        builder.push_record(["Digest", &self.digest.to_string()]);
        builder.push_record(["Epoch", &self.executed_epoch.to_string()]);
        if let Some(checkpoint) = self.checkpoint {
            builder.push_record(["Checkpoint", &checkpoint.to_string()]);
        }
        if let Some(timestamp) = &self.timestamp {
            builder.push_record(["Timestamp", timestamp]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(TableModify::new(TableRows::first()).with(TableAlignment::center()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        // Create a temporary TransactionResponse for shared formatting
        let tx_response = TransactionResponse {
            digest: self.digest,
            status: self.status.clone(),
            executed_epoch: self.executed_epoch,
            checkpoint: self.checkpoint,
            fee: self.fee.clone(),
            created: self.created.clone(),
            mutated: self.mutated.clone(),
            deleted: self.deleted.clone(),
            gas_object: self.gas_object.clone(),
            balance_changes: self.balance_changes.clone(),
        };

        if !self.created.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Created Objects ({})", "●".green(), self.created.len())?;
            tx_response.fmt_object_refs(f, &self.created)?;
        }

        if !self.mutated.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Mutated Objects ({})", "●".yellow(), self.mutated.len())?;
            tx_response.fmt_object_refs(f, &self.mutated)?;
        }

        if !self.deleted.is_empty() {
            writeln!(f)?;
            writeln!(f, "{} Deleted Objects ({})", "●".red(), self.deleted.len())?;
            tx_response.fmt_deleted_refs(f, &self.deleted)?;
        }

        // Gas Summary
        writeln!(f)?;
        let mut builder = TableBuilder::default();
        builder.push_record(["Gas Summary", ""]);
        builder.push_record(["Base Fee", &format_fee(self.fee.base_fee)]);
        builder.push_record(["Operation Fee", &format_fee(self.fee.operation_fee)]);
        builder.push_record(["Value Fee", &format_fee(self.fee.value_fee)]);
        builder.push_record(["Total", &format_fee(self.fee.total_fee)]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(TableModify::new(TableRows::first()).with(TableAlignment::center()));
        table.with(TableModify::new(TableCols::last()).with(TableAlignment::right()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        if !self.balance_changes.is_empty() {
            writeln!(f)?;
            let mut builder = TableBuilder::default();
            builder.push_record(["Address", "Change"]);

            for change in &self.balance_changes {
                let amount_str = if change.amount >= 0 {
                    format!("+{}", format_soma(change.amount as u128)).green().to_string()
                } else {
                    format!("-{}", format_soma(change.amount.unsigned_abs() as u128))
                        .red()
                        .to_string()
                };
                builder.push_record([change.address.to_string(), amount_str]);
            }

            let mut table = builder.build();
            table.with(TableStyle::rounded());
            table.with(TablePanel::header("Balance Changes"));
            table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
            table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
            table.with(TableModify::new(TableCols::last()).with(TableAlignment::right()));
            table.with(tabled::settings::style::BorderSpanCorrection);
            writeln!(f, "{}", table)?;
        }

        Ok(())
    }
}

// =============================================================================
// VALIDATOR LIST OUTPUT
// =============================================================================

#[derive(Debug, Serialize)]
pub struct ValidatorListOutput {
    pub validators: Vec<ValidatorSummary>,
}

impl Display for ValidatorListOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.validators.is_empty() {
            return writeln!(f, "{}", "No validators found.".yellow());
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["Address", "Status", "Voting Power", "Commission"]);

        for v in &self.validators {
            builder.push_record([
                truncate_id(&v.address.to_string()),
                v.status.to_string(),
                v.voting_power.to_string(),
                format!("{:.2}%", v.commission_rate as f64 / 100.0),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header(format!(
            "Validators ({} total)",
            self.validators.len()
        )));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)
    }
}

// =============================================================================
// VALIDATOR SUMMARY
// =============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ValidatorSummary {
    pub address: SomaAddress,
    pub status: ValidatorStatus,
    pub voting_power: u64,
    pub commission_rate: u64,
    pub network_address: String,
    pub p2p_address: String,
    pub primary_address: String,
    pub protocol_pubkey: String,
    pub network_pubkey: String,
    pub worker_pubkey: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum ValidatorStatus {
    Active,
    Pending,
}

impl Display for ValidatorStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ValidatorStatus::Active => write!(f, "{}", "Active".green()),
            ValidatorStatus::Pending => write!(f, "{}", "Pending".yellow()),
        }
    }
}

impl Display for ValidatorSummary {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut builder = TableBuilder::default();
        builder.push_record(["Address", &self.address.to_string()]);
        builder.push_record(["Status", &self.status.to_string()]);
        builder.push_record(["Voting Power", &self.voting_power.to_string()]);
        builder.push_record([
            "Commission Rate",
            &format!("{:.2}%", self.commission_rate as f64 / 100.0),
        ]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Validator Information"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        writeln!(f)?;
        let mut builder = TableBuilder::default();
        builder.push_record(["Network", &self.network_address]);
        builder.push_record(["P2P", &self.p2p_address]);
        builder.push_record(["Primary", &self.primary_address]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Network Addresses"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)?;

        writeln!(f)?;
        let mut builder = TableBuilder::default();
        builder.push_record(["Protocol", &truncate_key(&self.protocol_pubkey)]);
        builder.push_record(["Network", &truncate_key(&self.network_pubkey)]);
        builder.push_record(["Worker", &truncate_key(&self.worker_pubkey)]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Public Keys"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Format a fee value showing SOMA first with shannons in parentheses.
fn format_fee(shannons: u64) -> String {
    format!(
        "{} ({} shannons)",
        format_soma(shannons as u128),
        format_with_commas(shannons as u128)
    )
}

/// Format a balance in shannons as SOMA with appropriate suffix (public API).
pub fn format_soma_public(shannons: u128) -> String {
    format_soma(shannons)
}

/// Format a u64/u128 with comma separators for readability.
pub fn format_with_commas(n: u128) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

/// Truncate a hex ID for display (public for use by command modules).
pub fn truncate_id(s: &str) -> String {
    if s.len() <= 16 { s.to_string() } else { format!("{}...{}", &s[..10], &s[s.len() - 6..]) }
}

/// Format a balance in shannons as SOMA with appropriate suffix
pub fn format_soma(shannons: u128) -> String {
    if shannons == 0 {
        return "0 SOMA".to_string();
    }

    let whole = shannons / SHANNONS_PER_SOMA;
    let frac = shannons % SHANNONS_PER_SOMA;

    if whole >= 1_000_000_000 {
        format!("{:.2}B SOMA", whole as f64 / 1_000_000_000.0)
    } else if whole >= 1_000_000 {
        format!("{:.2}M SOMA", whole as f64 / 1_000_000.0)
    } else if whole >= 1_000 {
        format!("{:.2}K SOMA", whole as f64 / 1_000.0)
    } else if whole > 0 {
        if frac > 0 {
            let decimal_str = format!("{:09}", frac);
            let trimmed = decimal_str.trim_end_matches('0');
            format!("{}.{} SOMA", whole, trimmed)
        } else {
            format!("{} SOMA", whole)
        }
    } else {
        let decimal_str = format!("{:09}", frac);
        let trimmed = decimal_str.trim_end_matches('0');
        if trimmed.is_empty() { "0 SOMA".to_string() } else { format!("0.{} SOMA", trimmed) }
    }
}

/// Truncate a hex key for display
fn truncate_key(key: &str) -> String {
    if key.len() <= 20 {
        key.to_string()
    } else {
        format!("{}...{}", &key[..8], &key[key.len() - 8..])
    }
}

/// Truncate an address for display
fn truncate_address(addr: &str) -> String {
    if addr.len() <= 16 {
        addr.to_string()
    } else {
        format!("{}...{}", &addr[..10], &addr[addr.len() - 6..])
    }
}

/// Truncate a string to a maximum length
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() } else { format!("{}...", &s[..max_len - 3]) }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_soma() {
        assert_eq!(format_soma(0), "0 SOMA");
        assert_eq!(format_soma(1_000_000_000), "1 SOMA");
        assert_eq!(format_soma(1_500_000_000), "1.5 SOMA");
        assert_eq!(format_soma(1_001_000_000), "1.001 SOMA");
        assert_eq!(format_soma(1_999_999_999), "1.999999999 SOMA");
        assert_eq!(format_soma(1_000_000_000_000), "1.00K SOMA");
        assert_eq!(format_soma(1_500_000_000_000), "1.50K SOMA");
        assert_eq!(format_soma(1_000_000_000_000_000), "1.00M SOMA");
        assert_eq!(format_soma(500_000_000), "0.5 SOMA");
        assert_eq!(format_soma(100_000), "0.0001 SOMA");
    }

    #[test]
    fn test_truncate_address() {
        let addr = "0x1234567890abcdef1234567890abcdef12345678";
        assert_eq!(truncate_address(addr), "0x12345678...345678");

        let short = "0x1234";
        assert_eq!(truncate_address(short), "0x1234");
    }

    #[test]
    fn test_active_address_output_json_serialization() {
        // JSON output should be just the address string (backward compatible)
        let output = ActiveAddressOutput {
            address: SomaAddress::ZERO,
            alias: Some("my-alias".to_string()),
        };
        let json = serde_json::to_string(&output).unwrap();
        // Should serialize as the address, not as {"address":"...", "alias":"..."}
        assert!(!json.contains("alias"), "JSON should not contain alias field: {json}");
        assert!(json.contains(&SomaAddress::ZERO.to_string()));
    }

    #[test]
    fn test_active_address_display_with_alias() {
        let output = ActiveAddressOutput {
            address: SomaAddress::ZERO,
            alias: Some("my-alias".to_string()),
        };
        let response = ClientCommandResponse::ActiveAddress(Some(output));
        let display = format!("{}", response);
        assert!(display.contains("my-alias"), "Display should contain alias: {display}");
        assert!(
            display.contains(&SomaAddress::ZERO.to_string()),
            "Display should contain address: {display}"
        );
    }

    #[test]
    fn test_active_address_display_without_alias() {
        let output = ActiveAddressOutput { address: SomaAddress::ZERO, alias: None };
        let response = ClientCommandResponse::ActiveAddress(Some(output));
        let display = format!("{}", response);
        assert!(
            display.contains(&SomaAddress::ZERO.to_string()),
            "Display should contain address: {display}"
        );
    }

    #[test]
    fn test_active_address_display_none() {
        let response = ClientCommandResponse::ActiveAddress(None);
        let display = format!("{}", response);
        assert!(display.contains("No active address set"));
    }

    #[test]
    fn test_validator_list_output_empty() {
        let output = ValidatorListOutput { validators: vec![] };
        let display = format!("{}", output);
        assert!(display.contains("No validators found"));
    }

    #[test]
    fn test_validator_list_output_with_validators() {
        let output = ValidatorListOutput {
            validators: vec![
                ValidatorSummary {
                    address: SomaAddress::ZERO,
                    status: ValidatorStatus::Active,
                    voting_power: 1000,
                    commission_rate: 200,
                    network_address: "/ip4/127.0.0.1/tcp/8080".to_string(),
                    p2p_address: "/ip4/127.0.0.1/tcp/8084".to_string(),
                    primary_address: "/ip4/127.0.0.1/tcp/8081".to_string(),
                    protocol_pubkey: "abc123".to_string(),
                    network_pubkey: "def456".to_string(),
                    worker_pubkey: "ghi789".to_string(),
                },
            ],
        };
        let display = format!("{}", output);
        assert!(display.contains("Validators (1 total)"), "Should show count: {display}");
        assert!(display.contains("1000"), "Should show voting power: {display}");
        assert!(display.contains("2.00%"), "Should show commission rate: {display}");
    }

    #[test]
    fn test_validator_list_json_serialization() {
        let output = ValidatorListOutput {
            validators: vec![ValidatorSummary {
                address: SomaAddress::ZERO,
                status: ValidatorStatus::Active,
                voting_power: 1000,
                commission_rate: 200,
                network_address: "/ip4/127.0.0.1/tcp/8080".to_string(),
                p2p_address: "/ip4/127.0.0.1/tcp/8084".to_string(),
                primary_address: "/ip4/127.0.0.1/tcp/8081".to_string(),
                protocol_pubkey: "abc123".to_string(),
                network_pubkey: "def456".to_string(),
                worker_pubkey: "ghi789".to_string(),
            }],
        };
        let json = serde_json::to_string_pretty(&output).unwrap();
        assert!(json.contains("\"voting_power\""), "JSON should have voting_power field: {json}");
        assert!(json.contains("\"validators\""), "JSON should have validators array: {json}");
    }
}
