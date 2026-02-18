use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::Mutex;

use sdk::transaction_builder::TransactionBuilder;
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::checksum::Checksum;
use types::crypto::DecryptionKey;
use types::digests::{DataCommitment, ModelWeightsCommitment, ModelWeightsUrlCommitment};
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use types::model::{ArchitectureVersion, ModelWeightsManifest};
use types::object::{ObjectID, ObjectRef, Version};
use types::submission::SubmissionManifest;
use types::tensor::SomaTensor;
use types::transaction::{
    AddValidatorArgs, ClaimRewardsArgs, CommitModelArgs, CommitModelUpdateArgs,
    InitiateChallengeArgs, RemoveValidatorArgs, RevealModelArgs, RevealModelUpdateArgs,
    SubmitDataArgs, TransactionData, TransactionKind, UpdateValidatorMetadataArgs,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn to_py_val_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_address(s: &str) -> PyResult<SomaAddress> {
    s.parse::<SomaAddress>().map_err(to_py_val_err)
}

fn parse_object_id(s: &str) -> PyResult<ObjectID> {
    s.parse::<ObjectID>().map_err(to_py_val_err)
}

fn parse_object_ref(dict: &Bound<'_, PyDict>) -> PyResult<ObjectRef> {
    let id_str: String = dict
        .get_item("id")?
        .ok_or_else(|| PyValueError::new_err("missing 'id'"))?
        .extract()?;
    let version: u64 = dict
        .get_item("version")?
        .ok_or_else(|| PyValueError::new_err("missing 'version'"))?
        .extract()?;
    let digest_str: String = dict
        .get_item("digest")?
        .ok_or_else(|| PyValueError::new_err("missing 'digest'"))?
        .extract()?;

    let id = parse_object_id(&id_str)?;
    let digest = digest_str.parse().map_err(to_py_val_err)?;
    Ok((id, Version::from_u64(version), digest))
}

fn parse_hex_32(hex_str: &str, field_name: &str) -> PyResult<[u8; 32]> {
    let stripped = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let bytes = hex::decode(stripped)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex for {}: {}", field_name, e)))?;
    let arr: [u8; 32] = bytes.try_into().map_err(|_| {
        PyValueError::new_err(format!("{} must be exactly 32 bytes", field_name))
    })?;
    Ok(arr)
}

fn parse_embedding_vec(embedding: Vec<f32>) -> PyResult<SomaTensor> {
    if embedding.is_empty() {
        return Err(PyValueError::new_err("Embedding cannot be empty"));
    }
    let dim = embedding.len();
    Ok(SomaTensor::new(embedding, vec![dim]))
}

fn build_submission_manifest(
    url: &str,
    checksum_hex: &str,
    size: usize,
) -> PyResult<SubmissionManifest> {
    let parsed_url: url::Url =
        url.parse().map_err(|e| PyValueError::new_err(format!("Invalid URL: {}", e)))?;
    let checksum_bytes = parse_hex_32(checksum_hex, "data-checksum")?;
    let metadata = Metadata::V1(MetadataV1::new(Checksum(checksum_bytes), size));
    let manifest = Manifest::V1(ManifestV1::new(parsed_url, metadata));
    Ok(SubmissionManifest::new(manifest))
}

fn build_weights_manifest(
    url: &str,
    checksum_hex: &str,
    size: usize,
    decryption_key_hex: &str,
) -> PyResult<ModelWeightsManifest> {
    let parsed_url: url::Url =
        url.parse().map_err(|e| PyValueError::new_err(format!("Invalid URL: {}", e)))?;
    let checksum_bytes = parse_hex_32(checksum_hex, "weights-checksum")?;
    let key_bytes = parse_hex_32(decryption_key_hex, "decryption-key")?;
    let metadata = Metadata::V1(MetadataV1::new(Checksum(checksum_bytes), size));
    let manifest = Manifest::V1(ManifestV1::new(parsed_url, metadata));
    Ok(ModelWeightsManifest { manifest, decryption_key: DecryptionKey::new(key_bytes) })
}

// ---------------------------------------------------------------------------
// PySomaClient
// ---------------------------------------------------------------------------

/// A client for interacting with the Soma network via gRPC.
#[pyclass(name = "SomaClient")]
struct PySomaClient {
    inner: sdk::SomaClient,
}

#[pymethods]
impl PySomaClient {
    /// Create a new SomaClient connected to the given gRPC URL.
    #[new]
    fn new(py: Python<'_>, rpc_url: String) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let client = sdk::SomaClientBuilder::default()
                .build(&rpc_url)
                .await
                .map_err(to_py_err)?;
            Ok(PySomaClient { inner: client })
        })
    }

    /// Get the human-readable chain name (e.g. "mainnet", "testnet", "localnet").
    fn get_chain_identifier<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chain_name = client.get_chain_name().await.map_err(to_py_err)?;
            Ok(chain_name)
        })
    }

    /// Get the server version string.
    fn get_server_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let version = client.get_server_version().await.map_err(to_py_err)?;
            Ok(version)
        })
    }

    /// Get the balance (in shannons) for the given address.
    fn get_balance<'py>(&self, py: Python<'py>, address: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let addr = parse_address(&address)?;
            let balance = client.get_balance(&addr).await.map_err(to_py_err)?;
            Ok(balance)
        })
    }

    /// Get an object by its hex ID. Returns JSON string.
    fn get_object<'py>(&self, py: Python<'py>, object_id: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = parse_object_id(&object_id)?;
            let obj = client.get_object(id).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&obj).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get the current protocol version.
    fn get_protocol_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let version = client.get_protocol_version().await.map_err(to_py_err)?;
            Ok(version)
        })
    }

    /// Get epoch info as JSON. Pass None for latest epoch.
    fn get_epoch<'py>(
        &self,
        py: Python<'py>,
        epoch: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = client.get_epoch(epoch).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get the latest system state as a JSON string.
    fn get_latest_system_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state =
                client.get_latest_system_state().await.map_err(to_py_err)?;
            let json = serde_json::to_string(&state).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Execute a signed transaction (BCS bytes). Returns effects as JSON.
    fn execute_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx: types::transaction::Transaction =
                bcs::from_bytes(&tx_bytes).map_err(to_py_val_err)?;
            let response = client.execute_transaction(&tx).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response.effects).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get a transaction by its digest string. Returns effects as JSON.
    fn get_transaction<'py>(
        &self,
        py: Python<'py>,
        digest: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let d = digest.parse().map_err(to_py_val_err)?;
            let result = client.get_transaction(d).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&result.effects).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Simulate a transaction (unsigned BCS TransactionData bytes). Returns effects as JSON.
    fn simulate_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_data_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx_data: TransactionData =
                bcs::from_bytes(&tx_data_bytes).map_err(to_py_val_err)?;
            let result =
                client.simulate_transaction(&tx_data).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&result.effects).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get the latest checkpoint summary as JSON.
    fn get_latest_checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ckpt = client.get_latest_checkpoint().await.map_err(to_py_err)?;
            let json = serde_json::to_string(&ckpt).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get a checkpoint summary by sequence number. Returns JSON.
    fn get_checkpoint_summary<'py>(
        &self,
        py: Python<'py>,
        sequence_number: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ckpt = client
                .get_checkpoint_summary(sequence_number)
                .await
                .map_err(to_py_err)?;
            let json = serde_json::to_string(&ckpt).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get an object by its hex ID and version. Returns JSON string.
    fn get_object_with_version<'py>(
        &self,
        py: Python<'py>,
        object_id: String,
        version: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = parse_object_id(&object_id)?;
            let obj = client
                .get_object_with_version(id, Version::from_u64(version))
                .await
                .map_err(to_py_err)?;
            let json = serde_json::to_string(&obj).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// List objects owned by an address. Returns list of JSON strings.
    #[pyo3(signature = (owner, object_type=None, limit=None))]
    fn list_owned_objects<'py>(
        &self,
        py: Python<'py>,
        owner: String,
        object_type: Option<String>,
        limit: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            use futures::TryStreamExt as _;

            let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
            request.owner = Some(owner);
            if let Some(limit) = limit {
                request.page_size = Some(limit);
            }
            if let Some(ot) = object_type {
                let parsed = match ot.to_lowercase().as_str() {
                    "coin" => rpc::types::ObjectType::Coin,
                    "stakedsoma" | "staked_soma" => {
                        rpc::types::ObjectType::StakedSoma
                    }
                    "target" => rpc::types::ObjectType::Target,
                    "submission" => rpc::types::ObjectType::Submission,
                    "challenge" => rpc::types::ObjectType::Challenge,
                    "systemstate" | "system_state" => rpc::types::ObjectType::SystemState,
                    _ => {
                        return Err(to_py_val_err(format!(
                            "Unknown object type '{}'. Valid types: coin, staked_soma, target, submission, challenge, system_state",
                            ot
                        )));
                    }
                };
                request.object_type = Some(parsed.into());
            }

            let stream = client.list_owned_objects(request).await;
            tokio::pin!(stream);

            let mut results = Vec::new();
            while let Some(obj) = stream.try_next().await.map_err(to_py_err)? {
                let json = serde_json::to_string(&obj).map_err(to_py_err)?;
                results.push(json);
                if let Some(limit) = limit {
                    if results.len() >= limit as usize {
                        break;
                    }
                }
            }
            Ok(results)
        })
    }

    /// List targets with optional filtering. Returns JSON string.
    #[pyo3(signature = (status=None, epoch=None, limit=None))]
    fn list_targets<'py>(
        &self,
        py: Python<'py>,
        status: Option<String>,
        epoch: Option<u64>,
        limit: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut request = rpc::proto::soma::ListTargetsRequest::default();
            request.status_filter = status;
            request.epoch_filter = epoch;
            if let Some(limit) = limit {
                request.page_size = Some(limit);
            }
            let response = client.list_targets(request).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Get a challenge by its ID. Returns JSON string.
    fn get_challenge<'py>(
        &self,
        py: Python<'py>,
        challenge_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut request = rpc::proto::soma::GetChallengeRequest::default();
            request.challenge_id = Some(challenge_id);
            let response = client.get_challenge(request).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// List challenges with optional filtering. Returns JSON string.
    #[pyo3(signature = (target_id=None, status=None, epoch=None, limit=None))]
    fn list_challenges<'py>(
        &self,
        py: Python<'py>,
        target_id: Option<String>,
        status: Option<String>,
        epoch: Option<u64>,
        limit: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut request = rpc::proto::soma::ListChallengesRequest::default();
            request.target_id = target_id;
            request.status_filter = status;
            request.epoch_filter = epoch;
            if let Some(limit) = limit {
                request.page_size = Some(limit);
            }
            let response =
                client.list_challenges(request).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Check if the client API version matches the server version.
    /// Raises an error if versions don't match.
    fn check_api_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.check_api_version().await.map_err(to_py_err)?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// PyWalletContext
// ---------------------------------------------------------------------------

/// Python wrapper around the Soma wallet, providing key management,
/// transaction building, signing, and execution.
///
/// Transaction builders use the SDK's TransactionBuilder internally.
/// When `gas` is None, gas is auto-selected from the sender's owned coins.
#[pyclass(name = "WalletContext")]
struct PyWalletContext {
    inner: Arc<Mutex<WalletContext>>,
}

/// Helper: use TransactionBuilder to build TransactionData from a TransactionKind.
async fn build_tx_data(
    wallet: &WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    gas: Option<ObjectRef>,
) -> PyResult<TransactionData> {
    let builder = TransactionBuilder::new(wallet);
    builder.build_transaction_data(sender, kind, gas).await.map_err(to_py_err)
}

#[pymethods]
impl PyWalletContext {
    /// Open a wallet from a config file path (e.g. ~/.soma/client.yaml).
    #[new]
    fn new(config_path: String) -> PyResult<Self> {
        let path = PathBuf::from(&config_path);
        let ctx = WalletContext::new(path.as_path()).map_err(to_py_err)?;
        Ok(Self { inner: Arc::new(Mutex::new(ctx)) })
    }

    /// Return all addresses managed by this wallet as hex strings.
    fn get_addresses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let addrs: Vec<String> = wallet.get_addresses().iter().map(|a| a.to_string()).collect();
            Ok(addrs)
        })
    }

    /// Return the active address as a hex string.
    fn active_address<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut wallet = inner.lock().await;
            let addr = wallet.active_address().map_err(to_py_err)?;
            Ok(addr.to_string())
        })
    }

    /// Check if the wallet has any addresses.
    fn has_addresses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            Ok(wallet.has_addresses())
        })
    }

    /// Get gas objects owned by an address. Returns list of JSON strings.
    fn get_gas_objects<'py>(
        &self,
        py: Python<'py>,
        address: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let addr = parse_address(&address)?;
            let wallet = inner.lock().await;
            let refs = wallet
                .get_all_gas_objects_owned_by_address(addr)
                .await
                .map_err(to_py_err)?;
            let result: Vec<String> = refs
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.0.to_string(),
                        "version": r.1.value(),
                        "digest": format!("{}", r.2),
                    })
                    .to_string()
                })
                .collect();
            Ok(result)
        })
    }

    /// Sign BCS-encoded TransactionData, returning BCS-encoded signed Transaction bytes.
    fn sign_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_data_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx_data: TransactionData =
                bcs::from_bytes(&tx_data_bytes).map_err(to_py_val_err)?;
            let wallet = inner.lock().await;
            let signed = wallet.sign_transaction(&tx_data).await;
            let bytes = bcs::to_bytes(&signed).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Sign and execute a transaction, waiting for checkpoint inclusion.
    /// Takes BCS-encoded TransactionData bytes. Returns effects as JSON string.
    fn sign_and_execute_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_data_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx_data: TransactionData =
                bcs::from_bytes(&tx_data_bytes).map_err(to_py_val_err)?;
            let wallet = inner.lock().await;
            let signed = wallet.sign_transaction(&tx_data).await;
            let response = wallet.execute_transaction_must_succeed(signed).await;
            let json = serde_json::to_string(&response.effects).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Sign and execute a transaction, waiting for checkpoint inclusion.
    /// Unlike sign_and_execute_transaction, this does NOT panic on failure —
    /// it returns the effects JSON even if the transaction status is not ok.
    fn sign_and_execute_transaction_may_fail<'py>(
        &self,
        py: Python<'py>,
        tx_data_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx_data: TransactionData =
                bcs::from_bytes(&tx_data_bytes).map_err(to_py_val_err)?;
            let wallet = inner.lock().await;
            let signed = wallet.sign_transaction(&tx_data).await;
            let response = wallet.execute_transaction_may_fail(signed).await.map_err(to_py_err)?;
            let json = serde_json::to_string(&response.effects).map_err(to_py_err)?;
            Ok(json)
        })
    }

    /// Save wallet configuration to disk.
    fn save_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            wallet.save_config().map_err(to_py_err)?;
            Ok(())
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Coin & Object
    // -----------------------------------------------------------------------

    /// Build a TransferCoin transaction. Returns BCS bytes.
    /// If gas is None, auto-selects from sender's coins.
    #[pyo3(signature = (sender, recipient, coin, amount=None, gas=None))]
    fn build_transfer_coin<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        recipient: String,
        coin: Bound<'py, PyDict>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addr = parse_address(&recipient)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::TransferCoin {
                coin: coin_ref,
                amount,
                recipient: recipient_addr,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a TransferObjects transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, recipient, objects, gas=None))]
    fn build_transfer_objects<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        recipient: String,
        objects: Vec<Bound<'py, PyDict>>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addr = parse_address(&recipient)?;
        let obj_refs: Vec<ObjectRef> =
            objects.iter().map(parse_object_ref).collect::<PyResult<_>>()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::TransferObjects {
                objects: obj_refs,
                recipient: recipient_addr,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a PayCoins transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, recipients, amounts, coins, gas=None))]
    fn build_pay_coins<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        recipients: Vec<String>,
        amounts: Vec<u64>,
        coins: Vec<Bound<'py, PyDict>>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addrs: Vec<SomaAddress> = recipients
            .iter()
            .map(|r| parse_address(r))
            .collect::<PyResult<_>>()?;
        let coin_refs: Vec<ObjectRef> =
            coins.iter().map(parse_object_ref).collect::<PyResult<_>>()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::PayCoins {
                coins: coin_refs,
                amounts: Some(amounts),
                recipients: recipient_addrs,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Staking
    // -----------------------------------------------------------------------

    /// Build an AddStake transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, validator, coin, amount=None, gas=None))]
    fn build_add_stake<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        validator: String,
        coin: Bound<'py, PyDict>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let validator_addr = parse_address(&validator)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::AddStake {
                address: validator_addr,
                coin_ref,
                amount,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a WithdrawStake transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, staked_soma, gas=None))]
    fn build_withdraw_stake<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        staked_soma: Bound<'py, PyDict>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let staked_ref = parse_object_ref(&staked_soma)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::WithdrawStake { staked_soma: staked_ref };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an AddStakeToModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, coin, amount=None, gas=None))]
    fn build_add_stake_to_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        coin: Bound<'py, PyDict>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::AddStakeToModel {
                model_id: model,
                coin_ref,
                amount,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Model Management
    // -----------------------------------------------------------------------

    /// Build a CommitModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, weights_url_commitment, weights_commitment, architecture_version, stake_amount, commission_rate, staking_pool_id, gas=None))]
    fn build_commit_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        weights_url_commitment: String,
        weights_commitment: String,
        architecture_version: ArchitectureVersion,
        stake_amount: u64,
        commission_rate: u64,
        staking_pool_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let pool_id = parse_object_id(&staking_pool_id)?;
        let url_commitment = parse_hex_32(&weights_url_commitment, "weights-url-commitment")?;
        let wt_commitment = parse_hex_32(&weights_commitment, "weights-commitment")?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::CommitModel(CommitModelArgs {
                model_id: model,
                weights_url_commitment: ModelWeightsUrlCommitment::new(url_commitment),
                weights_commitment: ModelWeightsCommitment::new(wt_commitment),
                architecture_version,
                stake_amount,
                commission_rate,
                staking_pool_id: pool_id,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a RevealModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, weights_url, weights_checksum, weights_size, decryption_key, embedding, gas=None))]
    fn build_reveal_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        weights_url: String,
        weights_checksum: String,
        weights_size: usize,
        decryption_key: String,
        embedding: Vec<f32>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let manifest = build_weights_manifest(
            &weights_url, &weights_checksum, weights_size, &decryption_key,
        )?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::RevealModel(RevealModelArgs {
                model_id: model,
                weights_manifest: manifest,
                embedding: embedding_tensor,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a CommitModelUpdate transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, weights_url_commitment, weights_commitment, gas=None))]
    fn build_commit_model_update<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        weights_url_commitment: String,
        weights_commitment: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let url_commitment = parse_hex_32(&weights_url_commitment, "weights-url-commitment")?;
        let wt_commitment = parse_hex_32(&weights_commitment, "weights-commitment")?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::CommitModelUpdate(CommitModelUpdateArgs {
                model_id: model,
                weights_url_commitment: ModelWeightsUrlCommitment::new(url_commitment),
                weights_commitment: ModelWeightsCommitment::new(wt_commitment),
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a RevealModelUpdate transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, weights_url, weights_checksum, weights_size, decryption_key, embedding, gas=None))]
    fn build_reveal_model_update<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        weights_url: String,
        weights_checksum: String,
        weights_size: usize,
        decryption_key: String,
        embedding: Vec<f32>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let manifest = build_weights_manifest(
            &weights_url, &weights_checksum, weights_size, &decryption_key,
        )?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::RevealModelUpdate(RevealModelUpdateArgs {
                model_id: model,
                weights_manifest: manifest,
                embedding: embedding_tensor,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a DeactivateModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, gas=None))]
    fn build_deactivate_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::DeactivateModel { model_id: model };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a SetModelCommissionRate transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, new_rate, gas=None))]
    fn build_set_model_commission_rate<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        new_rate: u64,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::SetModelCommissionRate {
                model_id: model,
                new_rate,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ReportModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, gas=None))]
    fn build_report_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ReportModel { model_id: model };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an UndoReportModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, gas=None))]
    fn build_undo_report_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::UndoReportModel { model_id: model };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Submission
    // -----------------------------------------------------------------------

    /// Build a SubmitData transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, data_commitment, data_url, data_checksum, data_size, model_id, embedding, distance_score, bond_coin, gas=None))]
    fn build_submit_data<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        data_commitment: String,
        data_url: String,
        data_checksum: String,
        data_size: usize,
        model_id: String,
        embedding: Vec<f32>,
        distance_score: f32,
        bond_coin: Bound<'py, PyDict>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let model = parse_object_id(&model_id)?;
        let commitment_bytes = parse_hex_32(&data_commitment, "data-commitment")?;
        let manifest = build_submission_manifest(&data_url, &data_checksum, data_size)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let bond_ref = parse_object_ref(&bond_coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::SubmitData(SubmitDataArgs {
                target_id: target,
                data_commitment: DataCommitment::new(commitment_bytes),
                data_manifest: manifest,
                model_id: model,
                embedding: embedding_tensor,
                distance_score: SomaTensor::scalar(distance_score),
                bond_coin: bond_ref,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ClaimRewards transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, gas=None))]
    fn build_claim_rewards<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id: target });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ReportSubmission transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, challenger=None, gas=None))]
    fn build_report_submission<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        challenger: Option<String>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let challenger_addr = challenger.map(|c| parse_address(&c)).transpose()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ReportSubmission {
                target_id: target,
                challenger: challenger_addr,
            };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an UndoReportSubmission transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, gas=None))]
    fn build_undo_report_submission<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::UndoReportSubmission { target_id: target };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Challenge
    // -----------------------------------------------------------------------

    /// Build an InitiateChallenge transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, bond_coin, gas=None))]
    fn build_initiate_challenge<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        bond_coin: Bound<'py, PyDict>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let bond_ref = parse_object_ref(&bond_coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::InitiateChallenge(InitiateChallengeArgs {
                target_id: target,
                bond_coin: bond_ref,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ReportChallenge transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, challenge_id, gas=None))]
    fn build_report_challenge<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        challenge_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ReportChallenge { challenge_id: cid };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an UndoReportChallenge transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, challenge_id, gas=None))]
    fn build_undo_report_challenge<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        challenge_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::UndoReportChallenge { challenge_id: cid };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ClaimChallengeBond transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, challenge_id, gas=None))]
    fn build_claim_challenge_bond<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        challenge_id: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ClaimChallengeBond { challenge_id: cid };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -----------------------------------------------------------------------
    // Transaction builders — Validator Management
    // -----------------------------------------------------------------------

    /// Build an AddValidator transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, pubkey_bytes, network_pubkey_bytes, worker_pubkey_bytes, net_address, p2p_address, primary_address, proxy_address, gas=None))]
    fn build_add_validator<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        proxy_address: Vec<u8>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::AddValidator(AddValidatorArgs {
                pubkey_bytes,
                network_pubkey_bytes,
                worker_pubkey_bytes,
                net_address,
                p2p_address,
                primary_address,
                proxy_address,
            });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a RemoveValidator transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, pubkey_bytes, gas=None))]
    fn build_remove_validator<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        pubkey_bytes: Vec<u8>,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::RemoveValidator(RemoveValidatorArgs { pubkey_bytes });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an UpdateValidatorMetadata transaction. Returns BCS bytes.
    #[pyo3(signature = (
        sender,
        gas=None,
        next_epoch_network_address=None,
        next_epoch_p2p_address=None,
        next_epoch_primary_address=None,
        next_epoch_proxy_address=None,
        next_epoch_protocol_pubkey=None,
        next_epoch_worker_pubkey=None,
        next_epoch_network_pubkey=None,
    ))]
    fn build_update_validator_metadata<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        gas: Option<Bound<'py, PyDict>>,
        next_epoch_network_address: Option<Vec<u8>>,
        next_epoch_p2p_address: Option<Vec<u8>>,
        next_epoch_primary_address: Option<Vec<u8>>,
        next_epoch_proxy_address: Option<Vec<u8>>,
        next_epoch_protocol_pubkey: Option<Vec<u8>>,
        next_epoch_worker_pubkey: Option<Vec<u8>>,
        next_epoch_network_pubkey: Option<Vec<u8>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind =
                TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs {
                    next_epoch_network_address,
                    next_epoch_p2p_address,
                    next_epoch_primary_address,
                    next_epoch_proxy_address,
                    next_epoch_protocol_pubkey,
                    next_epoch_worker_pubkey,
                    next_epoch_network_pubkey,
                });
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a SetCommissionRate transaction (validator). Returns BCS bytes.
    #[pyo3(signature = (sender, new_rate, gas=None))]
    fn build_set_commission_rate<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        new_rate: u64,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::SetCommissionRate { new_rate };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build a ReportValidator transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, reportee, gas=None))]
    fn build_report_validator<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        reportee: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let reportee_addr = parse_address(&reportee)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::ReportValidator { reportee: reportee_addr };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    /// Build an UndoReportValidator transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, reportee, gas=None))]
    fn build_undo_report_validator<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        reportee: String,
        gas: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let reportee_addr = parse_address(&reportee)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wallet = inner.lock().await;
            let kind = TransactionKind::UndoReportValidator { reportee: reportee_addr };
            let tx_data = build_tx_data(&wallet, sender_addr, kind, gas_ref).await?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }
}

// ---------------------------------------------------------------------------
// Faucet
// ---------------------------------------------------------------------------

/// Request test tokens from a faucet server.
///
/// Args:
///     address: The recipient address as a hex string.
///     url: The faucet server URL (default: http://127.0.0.1:9123/v2/gas).
///
/// Returns:
///     JSON string with the faucet response (status, coins_sent).
#[pyfunction]
#[pyo3(signature = (address, url=None))]
fn request_faucet<'py>(
    py: Python<'py>,
    address: String,
    url: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let addr = parse_address(&address)?;
    let faucet_url = url.unwrap_or_else(|| "http://127.0.0.1:9123/v2/gas".to_string());
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let body = serde_json::json!({
            "FixedAmountRequest": { "recipient": addr.to_string() }
        });

        let resp = reqwest::Client::new()
            .post(&faucet_url)
            .json(&body)
            .send()
            .await
            .map_err(to_py_err)?;

        let status_code = resp.status().as_u16();

        if status_code == 429 {
            return Err(to_py_err("Faucet rate limit exceeded (429). Please wait and try again."));
        }
        if status_code == 503 {
            return Err(to_py_err("Faucet service is temporarily unavailable (503)."));
        }

        let text = resp.text().await.map_err(to_py_err)?;

        if status_code >= 400 {
            return Err(to_py_err(format!(
                "Faucet request failed with status {status_code}: {text}"
            )));
        }

        Ok(text)
    })
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python SDK for the Soma network.
#[pymodule]
fn soma_sdk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySomaClient>()?;
    m.add_class::<PyWalletContext>()?;
    m.add_function(wrap_pyfunction!(request_faucet, m)?)?;
    Ok(())
}
