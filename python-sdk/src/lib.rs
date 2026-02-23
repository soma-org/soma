use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::Mutex;

use types::base::SomaAddress;
use types::checksum::Checksum;
use types::crypto::DecryptionKey;
use types::digests::{DataCommitment, ModelWeightsCommitment, ModelWeightsUrlCommitment};
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use types::model::ModelWeightsManifest;
use types::object::{ObjectID, ObjectRef, Version};
use types::submission::SubmissionManifest;
use types::tensor::SomaTensor;
use types::transaction::{
    AddValidatorArgs, ClaimRewardsArgs, CommitModelArgs, CommitModelUpdateArgs,
    InitiateChallengeArgs, RemoveValidatorArgs, RevealModelArgs, RevealModelUpdateArgs,
    SubmitDataArgs, Transaction, TransactionData, TransactionKind, UpdateValidatorMetadataArgs,
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

/// Convert a serde value to a Python object with attribute access.
/// Dicts become `types.SimpleNamespace`, lists are preserved, scalars pass through.
fn to_py_obj(value: &impl serde::Serialize) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let py_val = pythonize::pythonize(py, value).map_err(to_py_err)?;
        dict_to_ns(py, &py_val).map(|v| v.unbind())
    })
}

/// Recursively convert Python dicts to `types.SimpleNamespace` for attribute access.
fn dict_to_ns<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let types_mod = py.import("types")?;
        let ns_cls = types_mod.getattr("SimpleNamespace")?;
        let kwargs = PyDict::new(py);
        for (key, val) in dict {
            kwargs.set_item(&key, dict_to_ns(py, &val)?)?;
        }
        ns_cls.call((), Some(&kwargs))
    } else if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        let items: Vec<Bound<'py, PyAny>> =
            list.iter().map(|item| dict_to_ns(py, &item)).collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, &items)?.into_any())
    } else {
        Ok(obj.clone())
    }
}

fn parse_address(s: &str) -> PyResult<SomaAddress> {
    s.parse::<SomaAddress>().map_err(to_py_val_err)
}

fn parse_object_id(s: &str) -> PyResult<ObjectID> {
    s.parse::<ObjectID>().map_err(to_py_val_err)
}

/// Get a field from a Python object (supports both SimpleNamespace and dict).
fn get_field<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.getattr(name).or_else(|_| {
        obj.get_item(name).map_err(|_| PyValueError::new_err(format!("missing '{name}'")))
    })
}

fn parse_object_ref(obj: &Bound<'_, PyAny>) -> PyResult<ObjectRef> {
    let id_str: String = get_field(obj, "id")?.extract()?;
    let version: u64 = get_field(obj, "version")?.extract()?;
    let digest_str: String = get_field(obj, "digest")?.extract()?;

    let id = parse_object_id(&id_str)?;
    let digest = digest_str.parse().map_err(to_py_val_err)?;
    Ok((id, Version::from_u64(version), digest))
}

fn parse_hex_32(hex_str: &str, field_name: &str) -> PyResult<[u8; 32]> {
    let stripped = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let bytes = hex::decode(stripped)
        .map_err(|e| PyValueError::new_err(format!("Invalid hex for {}: {}", field_name, e)))?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{} must be exactly 32 bytes", field_name)))?;
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
// Serializable structs for SimpleNamespace conversion
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct ManifestObj {
    url: String,
    checksum: String,
    size: usize,
    decryption_key: Option<String>,
}

#[derive(serde::Serialize)]
struct TargetObj {
    id: String,
    status: String,
    embedding: Vec<f32>,
    model_ids: Vec<String>,
    distance_threshold: f32,
    reward_pool: u64,
    generation_epoch: u64,
    bond_amount: u64,
    submitter: Option<String>,
    winning_model_id: Option<String>,
}

/// Extract a ManifestInput from a Python object (SimpleNamespace or dict).
fn extract_manifest(obj: &Bound<'_, PyAny>) -> PyResult<sdk::scoring_types::ManifestInput> {
    let url: String = get_field(obj, "url")?.extract()?;
    let decryption_key: Option<String> = obj
        .getattr("decryption_key")
        .or_else(|_| obj.get_item("decryption_key"))
        .ok()
        .and_then(|v| v.extract().ok());

    let encrypted_weights: Option<Vec<u8>> = obj
        .getattr("encrypted_weights")
        .or_else(|_| obj.get_item("encrypted_weights"))
        .ok()
        .and_then(|v| v.extract().ok());

    let (checksum, size) = if let Some(data) = &encrypted_weights {
        (sdk::crypto_utils::commitment_hex(data), data.len())
    } else {
        let c: String = get_field(obj, "checksum")?.extract()?;
        let s: usize = get_field(obj, "size")?.extract()?;
        (c, s)
    };

    Ok(sdk::scoring_types::ManifestInput { url, checksum, size, decryption_key })
}

// ---------------------------------------------------------------------------
// PyKeypair
// ---------------------------------------------------------------------------

/// Ed25519 keypair for signing Soma transactions.
#[pyclass(name = "Keypair")]
struct PyKeypair {
    inner: sdk::keypair::Keypair,
}

#[pymethods]
impl PyKeypair {
    /// Generate a random Ed25519 keypair.
    #[staticmethod]
    fn generate() -> Self {
        Self { inner: sdk::keypair::Keypair::generate() }
    }

    /// Create a keypair from a 32-byte secret key (raw bytes or hex string).
    #[staticmethod]
    fn from_secret_key(secret: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let secret_bytes: Vec<u8> = if let Ok(b) = secret.extract::<Vec<u8>>() {
            b
        } else if let Ok(s) = secret.extract::<String>() {
            let stripped = s.strip_prefix("0x").unwrap_or(&s);
            hex::decode(stripped)
                .map_err(|e| PyValueError::new_err(format!("Invalid hex string: {}", e)))?
        } else {
            return Err(PyValueError::new_err("secret_key must be bytes or a hex string"));
        };

        if secret_bytes.len() != 32 {
            return Err(PyValueError::new_err(format!(
                "Secret key must be exactly 32 bytes, got {}",
                secret_bytes.len()
            )));
        }

        let kp = sdk::keypair::Keypair::from_secret_key(&secret_bytes).map_err(to_py_val_err)?;
        Ok(Self { inner: kp })
    }

    /// Derive a keypair from a BIP39 mnemonic phrase.
    #[staticmethod]
    fn from_mnemonic(mnemonic: &str) -> PyResult<Self> {
        let kp = sdk::keypair::Keypair::from_mnemonic(mnemonic).map_err(to_py_val_err)?;
        Ok(Self { inner: kp })
    }

    /// Return the Soma address (0x-prefixed hex) for this keypair.
    fn address(&self) -> String {
        self.inner.address().to_string()
    }

    /// Sign BCS-encoded TransactionData bytes.
    /// Returns BCS-encoded signed Transaction bytes.
    fn sign<'py>(&self, py: Python<'py>, tx_data_bytes: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let tx_data: TransactionData = bcs::from_bytes(tx_data_bytes).map_err(to_py_val_err)?;
        let tx = self.inner.sign_transaction(tx_data);
        let bytes = bcs::to_bytes(&tx).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Export the secret key as a hex string (64 hex chars = 32 bytes).
    fn to_secret_key(&self) -> String {
        hex::encode(self.inner.to_secret_key())
    }
}

// ---------------------------------------------------------------------------
// PySomaClient
// ---------------------------------------------------------------------------

type FaucetGrpcClient =
    faucet::faucet_gen::faucet_client::FaucetClient<faucet::tonic::transport::Channel>;

/// A client for interacting with the Soma network via gRPC.
#[pyclass(name = "SomaClient")]
struct PySomaClient {
    inner: sdk::SomaClient,
    faucet_client: Option<Arc<Mutex<FaucetGrpcClient>>>,
    proxy_client: sdk::proxy_client::ProxyClient,
}

#[pymethods]
impl PySomaClient {
    /// Create a new SomaClient connected to the given gRPC URL.
    ///
    /// Optional service URLs connect gRPC clients for scoring, admin, and faucet.
    #[new]
    #[pyo3(signature = (rpc_url, scoring_url=None, admin_url=None, faucet_url=None))]
    fn new(
        py: Python<'_>,
        rpc_url: String,
        scoring_url: Option<String>,
        admin_url: Option<String>,
        faucet_url: Option<String>,
    ) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut builder = sdk::SomaClient::builder();
            if let Some(url) = &scoring_url {
                builder = builder.scoring_url(url);
            }
            if let Some(url) = &admin_url {
                builder = builder.admin_url(url);
            }
            let inner = builder.build(&rpc_url).await.map_err(to_py_err)?;

            let faucet_client = match faucet_url {
                Some(url) => {
                    let fc = FaucetGrpcClient::connect(url).await.map_err(to_py_err)?;
                    Some(Arc::new(Mutex::new(fc)))
                }
                None => None,
            };

            let proxy_client =
                sdk::proxy_client::ProxyClient::from_url(&rpc_url).map_err(to_py_err)?;

            Ok(PySomaClient { inner, faucet_client, proxy_client })
        })
    }

    // -------------------------------------------------------------------
    // Static crypto utility methods (delegate to sdk::crypto_utils)
    // -------------------------------------------------------------------

    /// Encrypt model weights with AES-256-CTR (zero IV).
    /// Returns (encrypted_bytes, key_hex).
    #[staticmethod]
    #[pyo3(signature = (data, key=None))]
    fn encrypt_weights<'py>(
        py: Python<'py>,
        data: &[u8],
        key: Option<&[u8]>,
    ) -> PyResult<(Bound<'py, PyBytes>, String)> {
        let key_arr: Option<[u8; 32]> = key
            .map(|k| {
                k.try_into().map_err(|_| PyValueError::new_err("key must be exactly 32 bytes"))
            })
            .transpose()?;
        let (encrypted, key_bytes) = sdk::crypto_utils::encrypt_weights(data, key_arr.as_ref());
        Ok((PyBytes::new(py, &encrypted), hex::encode(key_bytes)))
    }

    /// Decrypt model weights with AES-256-CTR (zero IV).
    /// Key can be raw 32 bytes or a hex string (64 hex chars).
    #[staticmethod]
    fn decrypt_weights<'py>(
        py: Python<'py>,
        data: &[u8],
        key: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let key_bytes: [u8; 32] = if let Ok(b) = key.extract::<Vec<u8>>() {
            b.try_into().map_err(|_| PyValueError::new_err("key must be exactly 32 bytes"))?
        } else if let Ok(s) = key.extract::<String>() {
            let stripped = s.strip_prefix("0x").unwrap_or(&s);
            let decoded = hex::decode(stripped)
                .map_err(|e| PyValueError::new_err(format!("Invalid hex key: {}", e)))?;
            decoded.try_into().map_err(|_| PyValueError::new_err("key must be exactly 32 bytes"))?
        } else {
            return Err(PyValueError::new_err("key must be bytes or a hex string"));
        };
        let decrypted = sdk::crypto_utils::decrypt_weights(data, &key_bytes);
        Ok(PyBytes::new(py, &decrypted))
    }

    /// Compute the Blake2b-256 hash of data. Returns a 64-character hex string.
    #[staticmethod]
    fn commitment(data: &[u8]) -> String {
        sdk::crypto_utils::commitment_hex(data)
    }

    /// Convert SOMA to shannons (the smallest on-chain unit).
    #[staticmethod]
    fn to_shannons(soma: f64) -> u64 {
        sdk::crypto_utils::to_shannons(soma)
    }

    /// Convert shannons to SOMA.
    #[staticmethod]
    fn to_soma(shannons: u64) -> f64 {
        sdk::crypto_utils::to_soma(shannons)
    }

    // -------------------------------------------------------------------
    // Read-only RPCs
    // -------------------------------------------------------------------

    /// Get the human-readable chain name (e.g. "testnet", "localnet").
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

    /// Get an object by its hex ID. Returns a Python object with attribute access.
    fn get_object<'py>(&self, py: Python<'py>, object_id: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let id = parse_object_id(&object_id)?;
            let obj = client.get_object(id).await.map_err(to_py_err)?;
            to_py_obj(&obj)
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

    /// Get the current model architecture version from the network.
    fn get_architecture_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let version = client.get_architecture_version().await.map_err(to_py_err)?;
            Ok(version)
        })
    }

    /// Get epoch info as a Python object. Pass None for latest epoch.
    fn get_epoch<'py>(&self, py: Python<'py>, epoch: Option<u64>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let response = client.get_epoch(epoch).await.map_err(to_py_err)?;
            to_py_obj(&response)
        })
    }

    /// Get the latest system state as a Python object.
    fn get_latest_system_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = client.get_latest_system_state().await.map_err(to_py_err)?;
            to_py_obj(&state)
        })
    }

    /// Execute a signed transaction (BCS bytes). Returns effects as a Python object.
    fn execute_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx: Transaction = bcs::from_bytes(&tx_bytes).map_err(to_py_val_err)?;
            let response = client.execute_transaction(&tx).await.map_err(to_py_err)?;
            to_py_obj(&response.effects)
        })
    }

    /// Get a transaction by its digest string. Returns effects as a Python object.
    fn get_transaction<'py>(&self, py: Python<'py>, digest: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let d = digest.parse().map_err(to_py_val_err)?;
            let result = client.get_transaction(d).await.map_err(to_py_err)?;
            to_py_obj(&result.effects)
        })
    }

    /// Simulate a transaction (unsigned BCS TransactionData bytes).
    fn simulate_transaction<'py>(
        &self,
        py: Python<'py>,
        tx_data_bytes: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tx_data: TransactionData =
                bcs::from_bytes(&tx_data_bytes).map_err(to_py_val_err)?;
            let result = client.simulate_transaction(&tx_data).await.map_err(to_py_err)?;
            to_py_obj(&result.effects)
        })
    }

    /// Get the latest checkpoint summary as a Python object.
    fn get_latest_checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ckpt = client.get_latest_checkpoint().await.map_err(to_py_err)?;
            to_py_obj(&ckpt)
        })
    }

    /// Get a checkpoint summary by sequence number.
    fn get_checkpoint_summary<'py>(
        &self,
        py: Python<'py>,
        sequence_number: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ckpt = client.get_checkpoint_summary(sequence_number).await.map_err(to_py_err)?;
            to_py_obj(&ckpt)
        })
    }

    /// Get an object by its hex ID and version.
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
            to_py_obj(&obj)
        })
    }

    /// List objects owned by an address.
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
                    "stakedsoma" | "staked_soma" => rpc::types::ObjectType::StakedSoma,
                    "target" => rpc::types::ObjectType::Target,
                    "challenge" => rpc::types::ObjectType::Challenge,
                    "systemstate" | "system_state" => rpc::types::ObjectType::SystemState,
                    _ => {
                        return Err(to_py_val_err(format!(
                            "Unknown object type '{}'. Valid types: coin, staked_soma, target, challenge, system_state",
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
                results.push(to_py_obj(&obj)?);
                if let Some(limit) = limit {
                    if results.len() >= limit as usize {
                        break;
                    }
                }
            }
            Ok(results)
        })
    }

    /// List targets with optional filtering.
    #[pyo3(signature = (status=None, epoch=None, limit=None, read_mask=None))]
    fn list_targets<'py>(
        &self,
        py: Python<'py>,
        status: Option<String>,
        epoch: Option<u64>,
        limit: Option<u32>,
        read_mask: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut request = rpc::proto::soma::ListTargetsRequest::default();
            request.status_filter = status;
            request.epoch_filter = epoch;
            if let Some(limit) = limit {
                request.page_size = Some(limit);
            }
            let effective_mask = read_mask.unwrap_or_else(|| {
                "id,status,generation_epoch,reward_pool,embedding,model_ids,\
                 distance_threshold,submitter,winning_model_id,bond_amount"
                    .to_string()
            });
            request.read_mask = Some(rpc::utils::field::FieldMask {
                paths: effective_mask.split(',').map(|s| s.trim().to_string()).collect(),
            });
            let response = client.list_targets(request).await.map_err(to_py_err)?;
            to_py_obj(&response)
        })
    }

    /// Get a challenge by its ID.
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
            to_py_obj(&response)
        })
    }

    /// List challenges with optional filtering.
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
            let response = client.list_challenges(request).await.map_err(to_py_err)?;
            to_py_obj(&response)
        })
    }

    /// Get the target embedding dimension from the current system parameters.
    fn get_embedding_dim<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = client.get_latest_system_state().await.map_err(to_py_err)?;
            Ok(state.parameters().target_embedding_dim)
        })
    }

    /// Get the minimum model stake (in shannons) from the current system parameters.
    fn get_model_min_stake<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = client.get_latest_system_state().await.map_err(to_py_err)?;
            Ok(state.parameters().model_min_stake)
        })
    }

    /// Look up revealed model manifests from the system state model registry.
    fn get_model_manifests<'py>(
        &self,
        py: Python<'py>,
        model_ids_or_target: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model_ids: Vec<String> = if let Ok(ids) = model_ids_or_target.extract::<Vec<String>>() {
            ids
        } else if let Ok(field) = get_field(model_ids_or_target, "model_ids") {
            field.extract::<Vec<String>>()?
        } else {
            return Err(PyValueError::new_err(
                "Expected a list of model ID strings or an object with .model_ids",
            ));
        };
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            use types::metadata::ManifestAPI as _;
            use types::metadata::MetadataAPI as _;

            let state = client.get_latest_system_state().await.map_err(to_py_err)?;
            let registry = &state.model_registry();

            let mut results: Vec<Py<PyAny>> = Vec::new();
            for id_str in &model_ids {
                let id = id_str.parse::<ObjectID>().map_err(to_py_val_err)?;
                let model =
                    registry.active_models.get(&id).or_else(|| registry.inactive_models.get(&id));
                if let Some(model) = model {
                    if let Some(wm) = &model.weights_manifest {
                        let obj = ManifestObj {
                            url: wm.manifest.url().to_string(),
                            checksum: hex::encode(wm.manifest.metadata().checksum().0),
                            size: wm.manifest.metadata().size(),
                            decryption_key: Some(hex::encode(wm.decryption_key.as_bytes())),
                        };
                        results.push(to_py_obj(&obj)?);
                    }
                }
            }
            Ok(results)
        })
    }

    /// Check if the client API version matches the server version.
    fn check_api_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            client.check_api_version().await.map_err(to_py_err)?;
            Ok(())
        })
    }

    // -------------------------------------------------------------------
    // Target helpers
    // -------------------------------------------------------------------

    /// List targets as typed Target objects.
    #[pyo3(signature = (status=None, epoch=None, limit=None))]
    fn get_targets<'py>(
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
            request.read_mask = Some(rpc::utils::field::FieldMask {
                paths: "id,status,generation_epoch,reward_pool,embedding,model_ids,\
                        distance_threshold,submitter,winning_model_id,bond_amount"
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            });
            let response = client.list_targets(request).await.map_err(to_py_err)?;
            let hex_id = |s: Option<String>| -> Option<String> {
                s.map(|v| if v.starts_with("0x") { v } else { format!("0x{v}") })
            };
            let targets: Vec<Py<PyAny>> = response
                .targets
                .into_iter()
                .map(|t| {
                    to_py_obj(&TargetObj {
                        id: hex_id(t.id).unwrap_or_default(),
                        status: t.status.unwrap_or_default(),
                        embedding: t.embedding,
                        model_ids: t
                            .model_ids
                            .into_iter()
                            .map(|m| if m.starts_with("0x") { m } else { format!("0x{m}") })
                            .collect(),
                        distance_threshold: t.distance_threshold.unwrap_or(0.0),
                        reward_pool: t.reward_pool.unwrap_or(0),
                        generation_epoch: t.generation_epoch.unwrap_or(0),
                        bond_amount: t.bond_amount.unwrap_or(0),
                        submitter: hex_id(t.submitter),
                        winning_model_id: hex_id(t.winning_model_id),
                    })
                })
                .collect::<PyResult<_>>()?;
            Ok(targets)
        })
    }

    /// Poll until the epoch changes. Returns the new epoch number.
    #[pyo3(signature = (timeout=120.0))]
    fn wait_for_next_epoch<'py>(
        &self,
        py: Python<'py>,
        timeout: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        use types::system_state::SystemStateTrait as _;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let state = client.get_latest_system_state().await.map_err(to_py_err)?;
            let start_epoch = state.epoch();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
            loop {
                if std::time::Instant::now() > deadline {
                    return Err(PyRuntimeError::new_err(format!(
                        "Epoch did not advance within {timeout}s"
                    )));
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let state = client.get_latest_system_state().await.map_err(to_py_err)?;
                if state.epoch() > start_epoch {
                    return Ok(state.epoch());
                }
            }
        })
    }

    // -------------------------------------------------------------------
    // Admin gRPC methods (delegated to sdk::SomaClient)
    // -------------------------------------------------------------------

    /// Trigger epoch advancement on localnet. Returns new epoch number.
    fn advance_epoch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let epoch = client.advance_epoch().await.map_err(to_py_err)?;
            Ok(epoch)
        })
    }

    // -------------------------------------------------------------------
    // Faucet gRPC methods (direct, not through sdk)
    // -------------------------------------------------------------------

    /// Request funds from the faucet.
    fn request_faucet<'py>(&self, py: Python<'py>, address: String) -> PyResult<Bound<'py, PyAny>> {
        let fc = self
            .faucet_client
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err("No faucet_url was provided when creating SomaClient")
            })?
            .clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut client = fc.lock().await;
            let response = client
                .request_gas(faucet::faucet_types::GasRequest { recipient: address })
                .await
                .map_err(to_py_err)?
                .into_inner();
            to_py_obj(&response)
        })
    }

    // -------------------------------------------------------------------
    // Proxy client methods (fetch model/data via fullnode proxy)
    // -------------------------------------------------------------------

    /// Fetch model weights via the fullnode proxy.
    fn fetch_model<'py>(&self, py: Python<'py>, model_id: String) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.proxy_client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mid = model_id.parse::<types::model::ModelId>().map_err(to_py_val_err)?;
            let data = proxy.fetch_model(&mid).await.map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &data).unbind()))
        })
    }

    /// Fetch submission data via the fullnode proxy.
    fn fetch_submission_data<'py>(
        &self,
        py: Python<'py>,
        target_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.proxy_client.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tid = target_id.parse::<types::target::TargetId>().map_err(to_py_val_err)?;
            let data = proxy.fetch_submission_data(&tid).await.map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &data).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Scoring gRPC methods (delegated to sdk::SomaClient)
    // -------------------------------------------------------------------

    /// Score model manifests against a data submission.
    #[pyo3(signature = (data_url, models, target_embedding, data=None, data_checksum=None, data_size=None, seed=0))]
    fn score<'py>(
        &self,
        py: Python<'py>,
        data_url: String,
        models: Vec<Bound<'py, PyAny>>,
        target_embedding: Vec<f32>,
        data: Option<&[u8]>,
        data_checksum: Option<String>,
        data_size: Option<usize>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (final_checksum, final_size) = match (data, data_checksum, data_size) {
            (Some(d), cs, sz) => {
                let checksum = cs.unwrap_or_else(|| sdk::crypto_utils::commitment_hex(d));
                let size = sz.unwrap_or(d.len());
                (checksum, size)
            }
            (None, Some(c), Some(s)) => (c, s),
            _ => {
                return Err(PyValueError::new_err(
                    "Either data or both data_checksum and data_size are required",
                ));
            }
        };

        let manifests: Vec<sdk::scoring_types::ManifestInput> =
            models.iter().map(|m| extract_manifest(m)).collect::<PyResult<_>>()?;

        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let request = sdk::scoring_types::ScoreRequest {
                data_url,
                data_checksum: final_checksum,
                data_size: final_size,
                model_manifests: manifests,
                target_embedding,
                seed,
            };
            let response = client.score(request).await.map_err(to_py_err)?;
            // Build the Python object directly from struct fields rather than
            // going through serde (which would apply the vec_f32_as_u32_bits
            // conversion, returning u32 bit patterns instead of f32 floats).
            Python::attach(|py| {
                let types_mod = py.import("types")?;
                let ns_cls = types_mod.getattr("SimpleNamespace")?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("winner", response.winner)?;
                kwargs.set_item("loss_score", response.loss_score)?;
                kwargs.set_item("embedding", response.embedding)?;
                kwargs.set_item("distance", response.distance)?;
                ns_cls.call((), Some(&kwargs)).map(|v| v.unbind())
            })
        })
    }

    /// Health check against the scoring service.
    fn scoring_health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let ok = client.scoring_health().await.map_err(to_py_err)?;
            Ok(ok)
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Coin & Object
    // -------------------------------------------------------------------

    /// Build a TransferCoin transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, recipient, coin, amount=None, gas=None))]
    fn build_transfer_coin<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        recipient: String,
        coin: Bound<'py, PyAny>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addr = parse_address(&recipient)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind =
                TransactionKind::TransferCoin { coin: coin_ref, amount, recipient: recipient_addr };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        objects: Vec<Bound<'py, PyAny>>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addr = parse_address(&recipient)?;
        let obj_refs: Vec<ObjectRef> =
            objects.iter().map(parse_object_ref).collect::<PyResult<_>>()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind =
                TransactionKind::TransferObjects { objects: obj_refs, recipient: recipient_addr };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        coins: Vec<Bound<'py, PyAny>>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let recipient_addrs: Vec<SomaAddress> =
            recipients.iter().map(|r| parse_address(r)).collect::<PyResult<_>>()?;
        let coin_refs: Vec<ObjectRef> =
            coins.iter().map(parse_object_ref).collect::<PyResult<_>>()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::PayCoins {
                coins: coin_refs,
                amounts: Some(amounts),
                recipients: recipient_addrs,
            };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Staking
    // -------------------------------------------------------------------

    /// Build an AddStake transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, validator, coin, amount=None, gas=None))]
    fn build_add_stake<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        validator: String,
        coin: Bound<'py, PyAny>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let validator_addr = parse_address(&validator)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::AddStake { address: validator_addr, coin_ref, amount };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        staked_soma: Bound<'py, PyAny>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let staked_ref = parse_object_ref(&staked_soma)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::WithdrawStake { staked_soma: staked_ref };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        coin: Bound<'py, PyAny>,
        amount: Option<u64>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let coin_ref = parse_object_ref(&coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::AddStakeToModel { model_id: model, coin_ref, amount };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Model Management
    // -------------------------------------------------------------------

    /// Build a CommitModel transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, model_id, weights_url_commitment, weights_commitment, stake_amount, commission_rate, staking_pool_id, gas=None))]
    fn build_commit_model<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        model_id: String,
        weights_url_commitment: String,
        weights_commitment: String,
        stake_amount: u64,
        commission_rate: u64,
        staking_pool_id: String,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let pool_id = parse_object_id(&staking_pool_id)?;
        let url_commitment = parse_hex_32(&weights_url_commitment, "weights-url-commitment")?;
        let wt_commitment = parse_hex_32(&weights_commitment, "weights-commitment")?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let architecture_version =
                client.get_architecture_version().await.map_err(to_py_err)?;
            let kind = TransactionKind::CommitModel(CommitModelArgs {
                model_id: model,
                weights_url_commitment: ModelWeightsUrlCommitment::new(url_commitment),
                weights_commitment: ModelWeightsCommitment::new(wt_commitment),
                architecture_version,
                stake_amount,
                commission_rate,
                staking_pool_id: pool_id,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let manifest =
            build_weights_manifest(&weights_url, &weights_checksum, weights_size, &decryption_key)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::RevealModel(RevealModelArgs {
                model_id: model,
                weights_manifest: manifest,
                embedding: embedding_tensor,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let url_commitment = parse_hex_32(&weights_url_commitment, "weights-url-commitment")?;
        let wt_commitment = parse_hex_32(&weights_commitment, "weights-commitment")?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::CommitModelUpdate(CommitModelUpdateArgs {
                model_id: model,
                weights_url_commitment: ModelWeightsUrlCommitment::new(url_commitment),
                weights_commitment: ModelWeightsCommitment::new(wt_commitment),
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let manifest =
            build_weights_manifest(&weights_url, &weights_checksum, weights_size, &decryption_key)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::RevealModelUpdate(RevealModelUpdateArgs {
                model_id: model,
                weights_manifest: manifest,
                embedding: embedding_tensor,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::DeactivateModel { model_id: model };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::SetModelCommissionRate { model_id: model, new_rate };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ReportModel { model_id: model };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let model = parse_object_id(&model_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::UndoReportModel { model_id: model };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Submission
    // -------------------------------------------------------------------

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
        bond_coin: Bound<'py, PyAny>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let model = parse_object_id(&model_id)?;
        let commitment_bytes = parse_hex_32(&data_commitment, "data-commitment")?;
        let manifest = build_submission_manifest(&data_url, &data_checksum, data_size)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;
        let bond_ref = parse_object_ref(&bond_coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::SubmitData(SubmitDataArgs {
                target_id: target,
                data_commitment: DataCommitment::new(commitment_bytes),
                data_manifest: manifest,
                model_id: model,
                embedding: embedding_tensor,
                distance_score: SomaTensor::scalar(distance_score),
                bond_coin: bond_ref,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id: target });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let challenger_addr = challenger.map(|c| parse_address(&c)).transpose()?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ReportSubmission {
                target_id: target,
                challenger: challenger_addr,
            };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::UndoReportSubmission { target_id: target };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Challenge
    // -------------------------------------------------------------------

    /// Build an InitiateChallenge transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, target_id, bond_coin, gas=None))]
    fn build_initiate_challenge<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        target_id: String,
        bond_coin: Bound<'py, PyAny>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let target = parse_object_id(&target_id)?;
        let bond_ref = parse_object_ref(&bond_coin)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::InitiateChallenge(InitiateChallengeArgs {
                target_id: target,
                bond_coin: bond_ref,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ReportChallenge { challenge_id: cid };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::UndoReportChallenge { challenge_id: cid };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let cid = parse_object_id(&challenge_id)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ClaimChallengeBond { challenge_id: cid };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // Transaction builders -- Validator Management
    // -------------------------------------------------------------------

    /// Build an AddValidator transaction. Returns BCS bytes.
    #[pyo3(signature = (sender, pubkey_bytes, network_pubkey_bytes, worker_pubkey_bytes, proof_of_possession, net_address, p2p_address, primary_address, proxy_address, gas=None))]
    fn build_add_validator<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        pubkey_bytes: Vec<u8>,
        network_pubkey_bytes: Vec<u8>,
        worker_pubkey_bytes: Vec<u8>,
        proof_of_possession: Vec<u8>,
        net_address: Vec<u8>,
        p2p_address: Vec<u8>,
        primary_address: Vec<u8>,
        proxy_address: Vec<u8>,
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::AddValidator(AddValidatorArgs {
                pubkey_bytes,
                network_pubkey_bytes,
                worker_pubkey_bytes,
                proof_of_possession,
                net_address,
                p2p_address,
                primary_address,
                proxy_address,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::RemoveValidator(RemoveValidatorArgs { pubkey_bytes });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        next_epoch_proof_of_possession=None,
    ))]
    fn build_update_validator_metadata<'py>(
        &self,
        py: Python<'py>,
        sender: String,
        gas: Option<Bound<'py, PyAny>>,
        next_epoch_network_address: Option<Vec<u8>>,
        next_epoch_p2p_address: Option<Vec<u8>>,
        next_epoch_primary_address: Option<Vec<u8>>,
        next_epoch_proxy_address: Option<Vec<u8>>,
        next_epoch_protocol_pubkey: Option<Vec<u8>>,
        next_epoch_worker_pubkey: Option<Vec<u8>>,
        next_epoch_network_pubkey: Option<Vec<u8>>,
        next_epoch_proof_of_possession: Option<Vec<u8>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs {
                next_epoch_network_address,
                next_epoch_p2p_address,
                next_epoch_primary_address,
                next_epoch_proxy_address,
                next_epoch_protocol_pubkey,
                next_epoch_worker_pubkey,
                next_epoch_network_pubkey,
                next_epoch_proof_of_possession,
            });
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::SetCommissionRate { new_rate };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let reportee_addr = parse_address(&reportee)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::ReportValidator { reportee: reportee_addr };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
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
        gas: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sender_addr = parse_address(&sender)?;
        let reportee_addr = parse_address(&reportee)?;
        let gas_ref = gas.as_ref().map(parse_object_ref).transpose()?;
        let client = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let kind = TransactionKind::UndoReportValidator { reportee: reportee_addr };
            let tx_data = client
                .build_transaction_data(sender_addr, kind, gas_ref)
                .await
                .map_err(to_py_err)?;
            let bytes = bcs::to_bytes(&tx_data).map_err(to_py_err)?;
            Ok(Python::attach(|py| PyBytes::new(py, &bytes).unbind()))
        })
    }

    // -------------------------------------------------------------------
    // High-level convenience methods (sign + execute via sdk)
    // -------------------------------------------------------------------

    /// Commit a model: auto-generates model_id and staking_pool_id,
    /// computes commitments, signs, and executes.
    /// Returns the model_id as a hex string.
    #[pyo3(signature = (signer, weights_url, encrypted_weights, commission_rate, stake_amount=None))]
    fn commit_model<'py>(
        &self,
        py: Python<'py>,
        signer: &PyKeypair,
        weights_url: String,
        encrypted_weights: &[u8],
        commission_rate: u64,
        stake_amount: Option<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        let kp = signer.inner.copy();

        let url_commitment = sdk::crypto_utils::commitment(weights_url.as_bytes());
        let wt_commitment = sdk::crypto_utils::commitment(encrypted_weights);

        let model_id = ObjectID::random();
        let staking_pool_id = ObjectID::random();
        let model_id_str = model_id.to_string();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let sender = kp.address();
            let architecture_version =
                client.get_architecture_version().await.map_err(to_py_err)?;

            let final_stake = match stake_amount {
                Some(s) => s,
                None => {
                    let state = client.get_latest_system_state().await.map_err(to_py_err)?;
                    state.parameters().model_min_stake
                }
            };

            let kind = TransactionKind::CommitModel(CommitModelArgs {
                model_id,
                weights_url_commitment: ModelWeightsUrlCommitment::new(url_commitment),
                weights_commitment: ModelWeightsCommitment::new(wt_commitment),
                architecture_version,
                stake_amount: final_stake,
                commission_rate,
                staking_pool_id,
            });
            let tx_data =
                client.build_transaction_data(sender, kind, None).await.map_err(to_py_err)?;
            client.sign_and_execute(&kp, tx_data, "CommitModel").await.map_err(to_py_err)?;
            Ok(model_id_str)
        })
    }

    /// Reveal a model: computes checksum/size from encrypted_weights,
    /// signs, and executes.
    #[pyo3(signature = (signer, model_id, weights_url, encrypted_weights, decryption_key, embedding))]
    fn reveal_model<'py>(
        &self,
        py: Python<'py>,
        signer: &PyKeypair,
        model_id: String,
        weights_url: String,
        encrypted_weights: &[u8],
        decryption_key: String,
        embedding: Vec<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        let kp = signer.inner.copy();
        let model = parse_object_id(&model_id)?;

        let checksum_hex = sdk::crypto_utils::commitment_hex(encrypted_weights);
        let weights_size = encrypted_weights.len();

        let manifest =
            build_weights_manifest(&weights_url, &checksum_hex, weights_size, &decryption_key)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let sender = kp.address();
            let kind = TransactionKind::RevealModel(RevealModelArgs {
                model_id: model,
                weights_manifest: manifest,
                embedding: embedding_tensor,
            });
            let tx_data =
                client.build_transaction_data(sender, kind, None).await.map_err(to_py_err)?;
            client.sign_and_execute(&kp, tx_data, "RevealModel").await.map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Submit data: computes commitment/checksum/size from data bytes,
    /// auto-selects a bond coin, signs, and executes.
    #[pyo3(signature = (signer, target_id, data, data_url, model_id, embedding, distance_score))]
    fn submit_data<'py>(
        &self,
        py: Python<'py>,
        signer: &PyKeypair,
        target_id: String,
        data: &[u8],
        data_url: String,
        model_id: String,
        embedding: Vec<f32>,
        distance_score: f32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        let kp = signer.inner.copy();
        let target = parse_object_id(&target_id)?;
        let model = parse_object_id(&model_id)?;

        let commitment_bytes = sdk::crypto_utils::commitment(data);
        let hash_hex = sdk::crypto_utils::commitment_hex(data);
        let data_size = data.len();

        let manifest = build_submission_manifest(&data_url, &hash_hex, data_size)?;
        let embedding_tensor = parse_embedding_vec(embedding)?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let sender = kp.address();

            // Auto-fetch bond coin
            let bond_coin = {
                use futures::TryStreamExt as _;
                let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
                request.owner = Some(sender.to_string());
                request.page_size = Some(1);
                request.object_type = Some(rpc::types::ObjectType::Coin.into());
                let stream = client.list_owned_objects(request).await;
                tokio::pin!(stream);
                let obj = stream.try_next().await.map_err(to_py_err)?.ok_or_else(|| {
                    PyRuntimeError::new_err(format!("No bond coin found for address {}", sender))
                })?;
                obj.compute_object_reference()
            };

            let kind = TransactionKind::SubmitData(SubmitDataArgs {
                target_id: target,
                data_commitment: DataCommitment::new(commitment_bytes),
                data_manifest: manifest,
                model_id: model,
                embedding: embedding_tensor,
                distance_score: SomaTensor::scalar(distance_score),
                bond_coin,
            });
            let tx_data =
                client.build_transaction_data(sender, kind, None).await.map_err(to_py_err)?;
            client.sign_and_execute(&kp, tx_data, "SubmitData").await.map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Claim rewards: signs and executes a ClaimRewards transaction.
    fn claim_rewards<'py>(
        &self,
        py: Python<'py>,
        signer: &PyKeypair,
        target_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.clone();
        let kp = signer.inner.copy();
        let target = parse_object_id(&target_id)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let sender = kp.address();
            let kind = TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id: target });
            let tx_data =
                client.build_transaction_data(sender, kind, None).await.map_err(to_py_err)?;
            client.sign_and_execute(&kp, tx_data, "ClaimRewards").await.map_err(to_py_err)?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python SDK for the Soma network.
#[pymodule]
fn soma_sdk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySomaClient>()?;
    m.add_class::<PyKeypair>()?;
    Ok(())
}
