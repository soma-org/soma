use std::collections::{HashMap, HashSet};

use burn::tensor::TensorData;
use probes::tensor::IntoTensorData;
use safetensors::{tensor::Metadata, SafeTensors};
use types::error::{EvaluationError, EvaluationResult};

const TENSORS_PER_EMBEDDING: usize = 2;
const MIN_INDEX: u64 = 1;

pub struct IndexedTensors<'data> {
    safetensors: SafeTensors<'data>,
    embedding_details: HashMap<u64, EmbeddingDetails>,
}

#[derive(Clone, Copy)]
pub struct EmbeddingDetails {
    pub num_bytes_represented: u64,
    pub num_bytes_used: u64,
}

impl<'data> IndexedTensors<'data> {
    pub fn new(
        metadata: Metadata,
        safetensors: SafeTensors<'data>,
        data_len: u64,
    ) -> EvaluationResult<Self> {
        let num_tensors = safetensors.len();
        if num_tensors % TENSORS_PER_EMBEDDING != 0 {
            return Err(EvaluationError::SafeTensorsFailure(
                "invalid tensor number".to_string(),
            ));
        }
        let num_embeddings = (num_tensors / TENSORS_PER_EMBEDDING) as u64;
        let mut bytes_added = HashSet::new();
        let mut embedding_details = HashMap::new();
        for index in MIN_INDEX..=num_embeddings {
            let ek = embedding_key(index);
            let bk = byte_key(index);

            // Parse embedding tensor
            let embedding_info = metadata.info(&ek).ok_or_else(|| {
                EvaluationError::SafeTensorsFailure("embedding not present".to_string())
            })?;

            let bytes_used = embedding_info.shape[0] * embedding_info.dtype.size();
            let byte_range_data = &safetensors
                .tensor(&bk)
                .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?
                .to_tensor_data()
                .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;
            let bytes_represented = ranges_to_bytes(byte_range_data)?;

            // Validate no byte overlaps
            for &byte_idx in &bytes_represented {
                if bytes_added.insert(byte_idx) {
                    return Err(EvaluationError::SafeTensorsFailure(format!(
                        "Byte overlap detected at index {}",
                        byte_idx
                    )));
                }
            }

            embedding_details.insert(
                index,
                EmbeddingDetails {
                    num_bytes_represented: bytes_represented.len() as u64,
                    num_bytes_used: bytes_used as u64,
                },
            );
        }

        for byte_idx in 0..data_len {
            if !bytes_added.contains(&byte_idx) {
                return Err(EvaluationError::SafeTensorsFailure(format!(
                    "Missing byte index {} in byte mappings",
                    byte_idx
                )));
            }
        }
        Ok(Self {
            safetensors,
            embedding_details,
        })
    }
    pub fn get_embedding(&self, index: u64) -> EvaluationResult<(TensorData, EmbeddingDetails)> {
        let ek = embedding_key(index);
        let tensor_data = self
            .safetensors
            .tensor(&ek)
            .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?
            .to_tensor_data()
            .map_err(|e| EvaluationError::SafeTensorsFailure(e.to_string()))?;
        let embedding_details = self
            .embedding_details
            .get(&index)
            .ok_or_else(|| {
                EvaluationError::SafeTensorsFailure("embedding not present".to_string())
            })?
            .clone();

        Ok((tensor_data, embedding_details))
    }
}

fn embedding_key(index: u64) -> String {
    format!("{}", index)
}

fn byte_key(index: u64) -> String {
    format!("b{}", index)
}

pub fn ranges_to_bytes(ranges: &TensorData) -> EvaluationResult<HashSet<u64>> {
    // Validate shape: must be [N, 2]
    let shape = ranges.shape.clone();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(EvaluationError::SafeTensorsFailure(format!(
            "Expected shape [N, 2], got {:?}",
            shape
        )));
    }

    let num_ranges = shape[0];
    let values = ranges.as_slice::<i64>().unwrap();

    let mut bytes = HashSet::new();

    // Flat iteration: each pair is start, end
    for i in 0..num_ranges {
        let start = values[2 * i] as u64;
        let end = values[2 * i + 1] as u64;

        if start > end {
            return Err(EvaluationError::SafeTensorsFailure(format!(
                "Invalid range at row {i}: start {start} > end {end}"
            )));
        }

        bytes.extend(start..=end);
    }

    Ok(bytes)
}
