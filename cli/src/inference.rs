use std::{
    fmt::{Debug, Display, Formatter},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use crate::error::{CliError, CliResult};
use bytes::Bytes;
use clap::Subcommand;
use intelligence::inference::{
    json_client::JSONClient, InferenceClient, InferenceInput, InferenceInputV1, InferenceOutputAPI,
};
use json_to_table::{json_to_table, Orientation};
use objects::{
    networking::{proxy::LocalObjectServerManager, ObjectNetworkService},
    storage::{memory::MemoryObjectStore, ObjectPath, ObjectStorage},
};
use reqwest::Url;
use serde::Serialize;
use serde_json::json;
use tokio::fs;
use tracing::info;
use types::{
    checksum::Checksum,
    consensus::committee::get_available_local_address,
    evaluation::ProbeSet,
    metadata::{Metadata, MetadataV1},
    p2p::to_host_port_str,
};

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceCommand {
    MockCall {
        base_url: Url,
        data_path: PathBuf,
        timeout_secs: Option<u64>,
        epoch: Option<u64>,
    },
}

#[derive(PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceOutput {
    metadata: Metadata,
    probe_set: ProbeSet,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum CommandOutput {
    Mock(InferenceOutput),
}

impl InferenceCommand {
    pub async fn execute(self) -> CliResult<CommandOutput> {
        Ok(match self {
            InferenceCommand::MockCall {
                base_url,
                data_path,
                timeout_secs,
                epoch,
            } => {
                let data = fs::read(&data_path)
                    .await
                    .map_err(|e| CliError::InferenceError(e.to_string()))?;

                let address = get_available_local_address();

                let server_object_storage = Arc::new(MemoryObjectStore::new_for_test());
                let checksum = Checksum::new_from_bytes(&data);
                let size = data.len();

                let metadata = Metadata::V1(MetadataV1::new(checksum, size));

                let object_path = ObjectPath::from_checksum(checksum);

                server_object_storage
                    .put_object(&object_path, Bytes::from(data))
                    .await
                    .map_err(|e| CliError::InferenceError(e.to_string()))?;

                let object_network_service: ObjectNetworkService<MemoryObjectStore> =
                    ObjectNetworkService::new(server_object_storage.clone());

                let mut local_obj_server = LocalObjectServerManager::new();
                local_obj_server
                    .start(&address, object_network_service)
                    .await;

                let object_server_url =
                    Url::parse(&format!("http://{}", to_host_port_str(&address).unwrap())).unwrap();

                let client = JSONClient::new(object_server_url, base_url).unwrap();
                let input = InferenceInput::V1(InferenceInputV1::new(epoch.unwrap_or(0), metadata));
                let inference_output = client
                    .call(input, Duration::from_secs(timeout_secs.unwrap_or(30)))
                    .await
                    .map_err(|e| CliError::InferenceError(e.to_string()))?;

                CommandOutput::Mock(InferenceOutput {
                    metadata: inference_output.metadata(),
                    probe_set: inference_output.probe_set(),
                })
            }
        })
    }
}

impl Display for CommandOutput {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            _ => {
                let json_obj = json![self];
                let mut table = json_to_table(&json_obj);
                let style = tabled::settings::Style::rounded().horizontals([]);
                table.with(style);
                table.array_orientation(Orientation::Column);
                write!(formatter, "{}", table)
            }
        }
    }
}

impl CommandOutput {
    pub fn print(&self, pretty: bool) {
        let line = if pretty {
            format!("{self}")
        } else {
            format!("{:?}", self)
        };
        // Log line by line
        for line in line.lines() {
            // Logs write to a file on the side.  Print to stdout and also log to file, for tests to pass.
            println!("{line}");
            info!("{line}")
        }
    }
}

// when --json flag is used, any output result is transformed into a JSON pretty string and sent to std output
impl Debug for CommandOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match serde_json::to_string_pretty(self) {
            Ok(json) => write!(f, "{json}"),
            Err(err) => write!(f, "Error serializing JSON: {err}"),
        }
    }
}
