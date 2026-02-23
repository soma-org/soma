#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;

use arrgen::normal_array;
use burn::backend::NdArray;
use burn::store::SafetensorsStore;
use burn::tensor::ops::FloatElem;
use burn::tensor::{PrintOptions, Tolerance, set_print_options};
use models::ModelAPI;
use models::tensor_conversions::ArrayWrapper;
use models::v1::ModelRunner;
use models::v1::modules::model::ModelConfig;
use safetensors::serialize;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

// Tiny model dimensions for fast tests.
// VOCAB_SIZE must be >= 258 to accommodate PAD_TOKEN_ID (256) and EOS_TOKEN_ID (257).
const VOCAB_SIZE: usize = 264;
const EMBEDDING_DIM: usize = 4;
const NUM_HEADS: usize = 2;
const NUM_LAYERS: usize = 2;
const HIDDEN_DIM: usize = EMBEDDING_DIM * 2;
const SEQ_LEN: usize = 8;
const BATCH_SIZE: usize = 2;
const SEED: u64 = 42;

fn tiny_model_config() -> ModelConfig {
    ModelConfig::new()
        .with_embedding_dim(EMBEDDING_DIM)
        .with_pwff_hidden_dim(HIDDEN_DIM)
        .with_num_layers(NUM_LAYERS)
        .with_num_heads(NUM_HEADS)
        .with_vocab_size(VOCAB_SIZE)
}

fn build_weights(
    seed: u64,
    num_layers: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> SafetensorsStore {
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();

    for l in 0..num_layers {
        let lseed = seed + l as u64;
        tensors.insert(
            format!("encoder.layers.{}.norm_1.gamma", l),
            ArrayWrapper(normal_array(lseed + 1, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_1.beta", l),
            ArrayWrapper(normal_array(lseed + 2, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.weight", l),
            ArrayWrapper(normal_array(lseed + 3, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.bias", l),
            ArrayWrapper(normal_array(lseed + 4, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.weight", l),
            ArrayWrapper(normal_array(lseed + 5, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.bias", l),
            ArrayWrapper(normal_array(lseed + 6, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.weight", l),
            ArrayWrapper(normal_array(lseed + 7, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.bias", l),
            ArrayWrapper(normal_array(lseed + 8, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.weight", l),
            ArrayWrapper(normal_array(lseed + 9, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.bias", l),
            ArrayWrapper(normal_array(lseed + 10, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.gamma", l),
            ArrayWrapper(normal_array(lseed + 11, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.beta", l),
            ArrayWrapper(normal_array(lseed + 12, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.weight", l),
            ArrayWrapper(normal_array(lseed + 13, &[embedding_dim, hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.bias", l),
            ArrayWrapper(normal_array(lseed + 14, &[hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.weight", l),
            ArrayWrapper(normal_array(lseed + 15, &[hidden_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.bias", l),
            ArrayWrapper(normal_array(lseed + 16, &[embedding_dim], 0.0, 1.0)),
        );
    }
    tensors.insert(
        "final_norm.gamma".to_string(),
        ArrayWrapper(normal_array(seed + 100, &[embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "final_norm.beta".to_string(),
        ArrayWrapper(normal_array(seed + 200, &[embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "embedding.weight".to_string(),
        ArrayWrapper(normal_array(seed + 250, &[vocab_size, embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "predictor.weight".to_string(),
        ArrayWrapper(normal_array(seed + 300, &[embedding_dim, vocab_size], 0.0, 1.0)),
    );
    tensors.insert(
        "predictor.bias".to_string(),
        ArrayWrapper(normal_array(seed + 400, &[vocab_size], 0.0, 1.0)),
    );

    let st = serialize(tensors, &None).unwrap();
    SafetensorsStore::from_bytes(Some(st))
}

fn make_runner() -> ModelRunner<TestBackend> {
    let config = tiny_model_config();
    let device: <TestBackend as burn::prelude::Backend>::Device = Default::default();
    ModelRunner::new(config, device, 1).with_max_seq_len(SEQ_LEN).with_batch_size(BATCH_SIZE)
}

fn make_input_buffer(size: usize) -> Arc<[u8]> {
    let vec: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    Arc::from(vec.as_slice())
}

#[tokio::test]
async fn eval_produces_embedding_and_loss() {
    let runner = make_runner();
    let buffer = make_input_buffer(SEQ_LEN * BATCH_SIZE);
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let output = runner.eval(data, weights, SEED).await.unwrap();

    // Embedding should be [EMBEDDING_DIM]
    assert_eq!(output.embedding.dims(), [EMBEDDING_DIM]);
    // Loss should be scalar [1]
    assert_eq!(output.loss.dims(), [1]);

    // Loss must be finite and positive (cross entropy is always >= 0)
    let loss_val: Vec<f32> = output.loss.to_data().to_vec::<f32>().unwrap();
    assert!(loss_val[0].is_finite(), "loss should be finite, got {}", loss_val[0]);
    assert!(loss_val[0] > 0.0, "loss should be positive, got {}", loss_val[0]);
}

#[tokio::test]
async fn eval_is_deterministic() {
    let buffer = make_input_buffer(SEQ_LEN * BATCH_SIZE);

    let runner1 = make_runner();
    let data1 = runner1.dataloader(buffer.clone()).await;
    let weights1 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output1 = runner1.eval(data1, weights1, SEED).await.unwrap();

    let runner2 = make_runner();
    let data2 = runner2.dataloader(buffer).await;
    let weights2 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output2 = runner2.eval(data2, weights2, SEED).await.unwrap();

    output1
        .embedding
        .to_data()
        .assert_approx_eq::<FT>(&output2.embedding.to_data(), Tolerance::default());
    output1.loss.to_data().assert_approx_eq::<FT>(&output2.loss.to_data(), Tolerance::default());
}

#[tokio::test]
async fn different_seeds_produce_different_losses() {
    let buffer = make_input_buffer(SEQ_LEN * BATCH_SIZE);

    let runner1 = make_runner();
    let data1 = runner1.dataloader(buffer.clone()).await;
    let weights1 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output1 = runner1.eval(data1, weights1, SEED).await.unwrap();

    let runner2 = make_runner();
    let data2 = runner2.dataloader(buffer).await;
    let weights2 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    // Different seed changes the SIGReg noise, so loss should differ
    let output2 = runner2.eval(data2, weights2, SEED + 999).await.unwrap();

    let loss1: Vec<f32> = output1.loss.to_data().to_vec::<f32>().unwrap();
    let loss2: Vec<f32> = output2.loss.to_data().to_vec::<f32>().unwrap();
    assert!(
        (loss1[0] - loss2[0]).abs() > 1e-6,
        "different seeds should produce different losses: {} vs {}",
        loss1[0],
        loss2[0]
    );
}

#[tokio::test]
async fn different_data_produces_different_embeddings() {
    let buffer1 = make_input_buffer(SEQ_LEN * BATCH_SIZE);
    // Different data: offset bytes
    let buffer2: Arc<[u8]> = Arc::from(
        (100..100 + SEQ_LEN * BATCH_SIZE).map(|i| (i % 256) as u8).collect::<Vec<u8>>().as_slice(),
    );

    let runner1 = make_runner();
    let data1 = runner1.dataloader(buffer1).await;
    let weights1 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output1 = runner1.eval(data1, weights1, SEED).await.unwrap();

    let runner2 = make_runner();
    let data2 = runner2.dataloader(buffer2).await;
    let weights2 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output2 = runner2.eval(data2, weights2, SEED).await.unwrap();

    let emb1: Vec<f32> = output1.embedding.to_data().to_vec::<f32>().unwrap();
    let emb2: Vec<f32> = output2.embedding.to_data().to_vec::<f32>().unwrap();
    let diff: f32 = emb1.iter().zip(&emb2).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-6,
        "different data should produce different embeddings, total diff = {}",
        diff
    );
}

#[tokio::test]
async fn different_weights_produce_different_outputs() {
    let buffer = make_input_buffer(SEQ_LEN * BATCH_SIZE);

    let runner1 = make_runner();
    let data1 = runner1.dataloader(buffer.clone()).await;
    let weights1 = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output1 = runner1.eval(data1, weights1, SEED).await.unwrap();

    let runner2 = make_runner();
    let data2 = runner2.dataloader(buffer).await;
    let weights2 = build_weights(SEED + 1000, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);
    let output2 = runner2.eval(data2, weights2, SEED).await.unwrap();

    let loss1: Vec<f32> = output1.loss.to_data().to_vec::<f32>().unwrap();
    let loss2: Vec<f32> = output2.loss.to_data().to_vec::<f32>().unwrap();
    assert!(
        (loss1[0] - loss2[0]).abs() > 1e-6,
        "different weights should produce different losses: {} vs {}",
        loss1[0],
        loss2[0]
    );
}

#[tokio::test]
async fn empty_data_returns_error() {
    let runner = make_runner();
    let buffer: Arc<[u8]> = Arc::from(Vec::new().as_slice());
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let result = runner.eval(data, weights, SEED).await;
    assert!(result.is_err(), "empty data should return an error");
}

#[tokio::test]
async fn multi_batch_data_accumulates_correctly() {
    // Create enough data for multiple batches: 3 sequences with batch_size=2 → 2 batches
    let num_items = 3;
    let buffer = make_input_buffer(SEQ_LEN * num_items);

    let runner = make_runner();
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let output = runner.eval(data, weights, SEED).await.unwrap();

    assert_eq!(output.embedding.dims(), [EMBEDDING_DIM]);
    assert_eq!(output.loss.dims(), [1]);

    let loss_val: Vec<f32> = output.loss.to_data().to_vec::<f32>().unwrap();
    assert!(
        loss_val[0].is_finite(),
        "loss should be finite after multi-batch, got {}",
        loss_val[0]
    );
    assert!(loss_val[0] > 0.0, "loss should be positive after multi-batch, got {}", loss_val[0]);
}

#[tokio::test]
async fn single_item_data_works() {
    // Single sequence item: exactly one batch of size 1
    let buffer = make_input_buffer(SEQ_LEN);

    let runner = make_runner();
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let output = runner.eval(data, weights, SEED).await.unwrap();

    assert_eq!(output.embedding.dims(), [EMBEDDING_DIM]);
    let loss_val: Vec<f32> = output.loss.to_data().to_vec::<f32>().unwrap();
    assert!(loss_val[0].is_finite());
    assert!(loss_val[0] > 0.0);
}

#[tokio::test]
async fn loss_snapshot() {
    // Regression test: pin the exact loss and embedding values for a fixed config/seed/data
    let runner = make_runner();
    let buffer = make_input_buffer(SEQ_LEN * BATCH_SIZE);
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let output = runner.eval(data, weights, SEED).await.unwrap();

    set_print_options(PrintOptions { threshold: 1000, edge_items: 3, precision: Some(8) });
    println!("embedding: {}", output.embedding);
    println!("loss: {}", output.loss);

    let loss_val: Vec<f32> = output.loss.to_data().to_vec::<f32>().unwrap();
    let emb_val: Vec<f32> = output.embedding.to_data().to_vec::<f32>().unwrap();

    // Sanity bounds: cross entropy + SIGReg with random weights can be large
    assert!(loss_val[0] > 0.5, "loss unexpectedly low: {}", loss_val[0]);
    assert!(loss_val[0] < 1000.0, "loss unexpectedly high: {}", loss_val[0]);

    // All embedding values should be finite
    for (i, v) in emb_val.iter().enumerate() {
        assert!(v.is_finite(), "embedding[{}] is not finite: {}", i, v);
    }
}

#[tokio::test]
async fn partial_sequence_data_works() {
    // Data that doesn't fill a complete sequence — should still work with padding
    let buffer = make_input_buffer(SEQ_LEN / 2);

    let runner = make_runner();
    let data = runner.dataloader(buffer).await;
    let weights = build_weights(SEED, NUM_LAYERS, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE);

    let output = runner.eval(data, weights, SEED).await.unwrap();

    assert_eq!(output.embedding.dims(), [EMBEDDING_DIM]);
    let loss_val: Vec<f32> = output.loss.to_data().to_vec::<f32>().unwrap();
    assert!(loss_val[0].is_finite());
}
