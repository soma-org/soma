// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use blobs::{BlobPath, downloader::BlobDownloader};
use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::store::SafetensorsStore;
use burn::tensor::{Tensor, TensorData};
use models::{ModelAPI, ModelOutput};
use object_store::memory::InMemory;
use object_store::{ObjectStore, PutPayload};
use runtime::v1::RuntimeV1;
use runtime::{CompetitionInput, ManifestCompetitionInput, RuntimeAPI};
use types::checksum::Checksum;
use types::error::{BlobError, BlobResult, ModelResult};
use types::metadata::{Manifest, ManifestAPI, ManifestV1, Metadata, MetadataAPI, MetadataV1};
use url::Url;

type B = NdArray;

// ---------------------------------------------------------------------------
// Mock ModelAPI
// ---------------------------------------------------------------------------

struct MockModel {
    /// For each call, return a fixed embedding + loss keyed by the safetensor bytes content.
    /// If empty, we use a default output derived from call order.
    outputs: Mutex<Vec<ModelOutput<B>>>,
}

impl MockModel {
    fn new(outputs: Vec<ModelOutput<B>>) -> Self {
        Self { outputs: Mutex::new(outputs) }
    }
}

#[async_trait]
impl ModelAPI for MockModel {
    type Data = Arc<[u8]>;
    type Backend = B;

    async fn dataloader(&self, buffer: Arc<[u8]>) -> Self::Data {
        buffer
    }

    async fn eval(
        &self,
        _data: Self::Data,
        _weights: SafetensorsStore,
        _seed: u64,
    ) -> ModelResult<ModelOutput<Self::Backend>> {
        let mut outputs = self.outputs.lock().unwrap();
        assert!(!outputs.is_empty(), "MockModel: more calls than expected outputs");
        Ok(outputs.remove(0))
    }
}

// ---------------------------------------------------------------------------
// Mock ModelAPI that fails
// ---------------------------------------------------------------------------

struct FailingModel;

#[async_trait]
impl ModelAPI for FailingModel {
    type Data = Arc<[u8]>;
    type Backend = B;

    async fn dataloader(&self, buffer: Arc<[u8]>) -> Self::Data {
        buffer
    }

    async fn eval(
        &self,
        _data: Self::Data,
        _weights: SafetensorsStore,
        _seed: u64,
    ) -> ModelResult<ModelOutput<Self::Backend>> {
        Err(types::error::ModelError::SafeTensorsFailure("mock failure".into()))
    }
}

// ---------------------------------------------------------------------------
// Mock BlobDownloader
// ---------------------------------------------------------------------------

/// A mock downloader that writes manifest data directly into the in-memory store.
struct MockDownloader {
    store: Arc<InMemory>,
    /// Pre-registered manifest data: checksum -> bytes
    manifest_data: HashMap<Checksum, Vec<u8>>,
}

impl MockDownloader {
    fn new(store: Arc<InMemory>, manifest_data: HashMap<Checksum, Vec<u8>>) -> Self {
        Self { store, manifest_data }
    }
}

#[async_trait]
impl BlobDownloader for MockDownloader {
    type Store = InMemory;

    async fn download(&self, manifest: &Manifest, blob_path: BlobPath) -> BlobResult<()> {
        let checksum = manifest.metadata().checksum();
        let data = self
            .manifest_data
            .get(&checksum)
            .ok_or_else(|| BlobError::NotFound(format!("no mock data for {checksum}")))?;
        self.store
            .put(&blob_path.path(), PutPayload::from(data.clone()))
            .await
            .map_err(BlobError::ObjectStoreError)?;
        Ok(())
    }
}

/// A downloader that always fails.
struct FailingDownloader;

#[async_trait]
impl BlobDownloader for FailingDownloader {
    type Store = InMemory;

    async fn download(&self, _manifest: &Manifest, _blob_path: BlobPath) -> BlobResult<()> {
        Err(BlobError::NetworkRequest("mock download failure".into()))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_checksum(val: u8) -> Checksum {
    Checksum([val; 32])
}

fn make_manifest(checksum: Checksum, size: usize) -> Manifest {
    Manifest::V1(ManifestV1::new(
        Url::parse("http://example.com/blob").unwrap(),
        Metadata::V1(MetadataV1::new(checksum, size)),
    ))
}

/// Minimal valid safetensors bytes (empty tensors map).
fn empty_safetensors() -> Vec<u8> {
    // safetensors format: 8 bytes LE header length + JSON header + tensor data
    // Minimal valid: header = "{}" (empty map), no tensor data.
    let header = b"{}";
    let header_len = header.len() as u64;
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(header);
    buf
}

fn make_output(embedding: &[f32], loss: f32) -> ModelOutput<B> {
    let device = Default::default();
    ModelOutput {
        embedding: Tensor::<B, 1>::from_floats(embedding, &device),
        loss: Tensor::<B, 1>::from_floats([loss], &device),
    }
}

async fn setup_store_with_data(
    store: &InMemory,
    data_path: &BlobPath,
    data: &[u8],
    model_paths: &[&BlobPath],
) {
    store.put(&data_path.path(), PutPayload::from(data.to_vec())).await.unwrap();
    let safetensor_bytes = empty_safetensors();
    for path in model_paths {
        store.put(&path.path(), PutPayload::from(safetensor_bytes.clone())).await.unwrap();
    }
}

// ===========================================================================
// Tests for RuntimeV1::competition
// ===========================================================================

#[tokio::test]
async fn runtime_single_model_returns_winner_zero() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(MockDownloader::new(store.clone(), HashMap::new()));

    let data_path = BlobPath::Data(1, make_checksum(0));
    let model_path = BlobPath::Weights(1, make_checksum(1));
    setup_store_with_data(&store, &data_path, b"test data", &[&model_path]).await;

    let embedding = [1.0, 0.0, 0.0, 0.0];
    let model = Arc::new(MockModel::new(vec![make_output(&embedding, 0.5)]));

    let device = <B as Backend>::Device::default();
    let target_data = TensorData::from([1.0f32, 0.0, 0.0, 0.0]);

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);
    let input = CompetitionInput::new(data_path, vec![model_path], target_data, 0);
    let output = runtime.competition(input).await.unwrap();

    assert_eq!(output.winner(), 0);
    // Embedding matches target, cosine distance should be ~0
    let dist: Vec<f32> = output.distance().to_vec().unwrap();
    assert!(dist[0].abs() < 1e-5, "expected distance ~0, got {}", dist[0]);
}

#[tokio::test]
async fn runtime_selects_model_with_lowest_loss() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(MockDownloader::new(store.clone(), HashMap::new()));

    let data_path = BlobPath::Data(1, make_checksum(0));
    let model_paths = vec![
        BlobPath::Weights(1, make_checksum(1)),
        BlobPath::Weights(1, make_checksum(2)),
        BlobPath::Weights(1, make_checksum(3)),
    ];
    let path_refs: Vec<&BlobPath> = model_paths.iter().collect();
    setup_store_with_data(&store, &data_path, b"data", &path_refs).await;

    let model = Arc::new(MockModel::new(vec![
        make_output(&[1.0, 0.0], 0.9), // worst
        make_output(&[0.0, 1.0], 0.1), // best
        make_output(&[1.0, 1.0], 0.5), // middle
    ]));

    let device = <B as Backend>::Device::default();
    let target_data = TensorData::from([0.0f32, 1.0]);

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);
    let input = CompetitionInput::new(data_path, model_paths, target_data, 0);
    let output = runtime.competition(input).await.unwrap();

    assert_eq!(output.winner(), 1);
}

#[tokio::test]
async fn runtime_computes_cosine_distance_to_target() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(MockDownloader::new(store.clone(), HashMap::new()));

    let data_path = BlobPath::Data(1, make_checksum(0));
    let model_path = BlobPath::Weights(1, make_checksum(1));
    setup_store_with_data(&store, &data_path, b"data", &[&model_path]).await;

    // Orthogonal vectors → cosine distance = 1.0
    let model = Arc::new(MockModel::new(vec![make_output(&[1.0, 0.0], 0.1)]));

    let device = <B as Backend>::Device::default();
    let target_data = TensorData::from([0.0f32, 1.0]);

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);
    let input = CompetitionInput::new(data_path, vec![model_path], target_data, 0);
    let output = runtime.competition(input).await.unwrap();

    let dist: Vec<f32> = output.distance().to_vec().unwrap();
    assert!(
        (dist[0] - 1.0).abs() < 1e-5,
        "orthogonal vectors should have distance ~1, got {}",
        dist[0]
    );
}

#[tokio::test]
async fn runtime_returns_loss_and_embedding_from_winner() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(MockDownloader::new(store.clone(), HashMap::new()));

    let data_path = BlobPath::Data(1, make_checksum(0));
    let model_path = BlobPath::Weights(1, make_checksum(1));
    setup_store_with_data(&store, &data_path, b"data", &[&model_path]).await;

    let embedding = [0.6, 0.8];
    let loss = 0.42;
    let model = Arc::new(MockModel::new(vec![make_output(&embedding, loss)]));

    let device = <B as Backend>::Device::default();
    let target_data = TensorData::from([1.0f32, 0.0]);

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);
    let input = CompetitionInput::new(data_path, vec![model_path], target_data, 0);
    let output = runtime.competition(input).await.unwrap();

    let loss_vec: Vec<f32> = output.loss_score().to_vec().unwrap();
    assert!((loss_vec[0] - 0.42).abs() < 1e-5);

    let emb_vec: Vec<f32> = output.embedding().to_vec().unwrap();
    assert!((emb_vec[0] - 0.6).abs() < 1e-5);
    assert!((emb_vec[1] - 0.8).abs() < 1e-5);
}

#[tokio::test]
async fn runtime_model_call_failure_propagates() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(MockDownloader::new(store.clone(), HashMap::new()));

    let data_path = BlobPath::Data(1, make_checksum(0));
    let model_path = BlobPath::Weights(1, make_checksum(1));
    setup_store_with_data(&store, &data_path, b"data", &[&model_path]).await;

    let model = Arc::new(FailingModel);
    let device = <B as Backend>::Device::default();
    let target_data = TensorData::from([1.0f32, 0.0]);

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);
    let input = CompetitionInput::new(data_path, vec![model_path], target_data, 0);
    let err = runtime.competition(input).await.unwrap_err();

    assert!(
        err.to_string().contains("mock failure"),
        "expected model error to propagate, got: {err}"
    );
}

// ===========================================================================
// Tests for RuntimeV1::download_manifest
// ===========================================================================

#[tokio::test]
async fn download_manifest_writes_to_store() {
    let store = Arc::new(InMemory::new());
    let checksum = make_checksum(10);
    let data = b"manifest content".to_vec();

    let mut manifest_data = HashMap::new();
    manifest_data.insert(checksum, data.clone());

    let downloader = Arc::new(MockDownloader::new(store.clone(), manifest_data));
    let model = Arc::new(MockModel::new(vec![]));
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store.clone(), downloader, 1, device, model);

    let manifest = make_manifest(checksum, data.len());
    let path = BlobPath::Data(1, checksum);

    runtime.download_manifest(&manifest, &path).await.unwrap();

    // Verify data was written
    let result = store.get(&path.path()).await.unwrap();
    let bytes = result.bytes().await.unwrap();
    assert_eq!(bytes.as_ref(), b"manifest content");
}

#[tokio::test]
async fn download_manifest_propagates_download_error() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(FailingDownloader);
    let model = Arc::new(MockModel::new(vec![]));
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);

    let manifest = make_manifest(make_checksum(10), 100);
    let path = BlobPath::Data(1, make_checksum(10));

    let err = runtime.download_manifest(&manifest, &path).await.unwrap_err();
    assert!(
        err.to_string().contains("mock download failure"),
        "expected download error, got: {err}"
    );
}

// ===========================================================================
// Tests for RuntimeV1::manifest_competition
// ===========================================================================

#[tokio::test]
async fn manifest_runtime_downloads_then_runs() {
    let store = Arc::new(InMemory::new());

    let data_checksum = make_checksum(20);
    let model_checksum_a = make_checksum(21);
    let model_checksum_b = make_checksum(22);
    let epoch = 5u64;

    let data_content = b"input data".to_vec();
    let safetensor_bytes = empty_safetensors();

    let mut manifest_data = HashMap::new();
    manifest_data.insert(data_checksum, data_content.clone());
    manifest_data.insert(model_checksum_a, safetensor_bytes.clone());
    manifest_data.insert(model_checksum_b, safetensor_bytes.clone());

    let downloader = Arc::new(MockDownloader::new(store.clone(), manifest_data));
    let model = Arc::new(MockModel::new(vec![
        make_output(&[1.0, 0.0], 0.8), // model A
        make_output(&[0.0, 1.0], 0.2), // model B (winner)
    ]));
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store, downloader, epoch, device, model);

    let data_manifest = make_manifest(data_checksum, data_content.len());
    let model_manifest_a = make_manifest(model_checksum_a, safetensor_bytes.len());
    let model_manifest_b = make_manifest(model_checksum_b, safetensor_bytes.len());

    let target_data = TensorData::from([0.0f32, 1.0]);
    let input = ManifestCompetitionInput::new(
        data_manifest,
        vec![model_manifest_a, model_manifest_b],
        target_data,
        0,
    );

    let output = runtime.manifest_competition(input).await.unwrap();
    assert_eq!(output.winner(), 1);
}

#[tokio::test]
async fn manifest_runtime_uses_correct_blob_paths() {
    // Verify that manifest_competition constructs BlobPath using epoch + manifest checksum.
    let store = Arc::new(InMemory::new());
    let epoch = 42u64;

    let data_checksum = make_checksum(30);
    let model_checksum = make_checksum(31);

    let data_content = b"some data".to_vec();
    let safetensor_bytes = empty_safetensors();

    // The downloader will write to the store at the computed BlobPath.
    // After manifest_competition, we verify the paths exist in the store.
    let mut manifest_data = HashMap::new();
    manifest_data.insert(data_checksum, data_content.clone());
    manifest_data.insert(model_checksum, safetensor_bytes.clone());

    let downloader = Arc::new(MockDownloader::new(store.clone(), manifest_data));
    let model = Arc::new(MockModel::new(vec![make_output(&[1.0], 0.5)]));
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store.clone(), downloader, epoch, device, model);

    let input = ManifestCompetitionInput::new(
        make_manifest(data_checksum, data_content.len()),
        vec![make_manifest(model_checksum, safetensor_bytes.len())],
        TensorData::from([1.0f32]),
        0,
    );

    runtime.manifest_competition(input).await.unwrap();

    // Verify expected blob paths were written
    let data_path = BlobPath::Data(epoch, data_checksum);
    let weights_path = BlobPath::Weights(epoch, model_checksum);

    assert!(store.get(&data_path.path()).await.is_ok(), "data blob should exist");
    assert!(store.get(&weights_path.path()).await.is_ok(), "weights blob should exist");
}

#[tokio::test]
async fn manifest_runtime_download_failure_propagates() {
    let store = Arc::new(InMemory::new());
    let downloader = Arc::new(FailingDownloader);
    let model = Arc::new(MockModel::new(vec![]));
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);

    let input = ManifestCompetitionInput::new(
        make_manifest(make_checksum(40), 100),
        vec![make_manifest(make_checksum(41), 100)],
        TensorData::from([1.0f32]),
        0,
    );

    let err = runtime.manifest_competition(input).await.unwrap_err();
    assert!(
        err.to_string().contains("mock download failure"),
        "expected download error, got: {err}"
    );
}

#[tokio::test]
async fn manifest_runtime_model_failure_after_download_propagates() {
    let store = Arc::new(InMemory::new());

    let data_checksum = make_checksum(50);
    let model_checksum = make_checksum(51);

    let mut manifest_data = HashMap::new();
    manifest_data.insert(data_checksum, b"data".to_vec());
    manifest_data.insert(model_checksum, empty_safetensors());

    let downloader = Arc::new(MockDownloader::new(store.clone(), manifest_data));
    let model = Arc::new(FailingModel);
    let device = <B as Backend>::Device::default();

    let runtime = RuntimeV1::new(store, downloader, 1, device, model);

    let input = ManifestCompetitionInput::new(
        make_manifest(data_checksum, 4),
        vec![make_manifest(model_checksum, empty_safetensors().len())],
        TensorData::from([1.0f32]),
        0,
    );

    let err = runtime.manifest_competition(input).await.unwrap_err();
    assert!(err.to_string().contains("mock failure"), "expected model error, got: {err}");
}
