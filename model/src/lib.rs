use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use error::{ModelError, ModelResult};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
pub mod client;
pub mod error;

#[enum_dispatch]
pub(crate) trait ModelInputAPI {
    fn input(&self) -> Bytes;
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelInputV1 {
    input: Bytes,
}

impl ModelInputV1 {
    pub fn new(input: Bytes) -> Self {
        Self { input }
    }
}

impl ModelInputAPI for ModelInputV1 {
    fn input(&self) -> Bytes {
        self.input.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ModelInputAPI)]
pub enum ModelInput {
    V1(ModelInputV1),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

#[enum_dispatch]
pub trait ModelOutputAPI {
    fn embeddings(&self) -> Array2<f32>;
    fn byte_range(&self, index: usize) -> Option<ByteRange>;
    fn byte_ranges(&self) -> Vec<ByteRange>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ModelOutputV1 {
    embeddings: Array2<f32>,
    byte_ranges: Vec<ByteRange>,
}

impl ModelOutputAPI for ModelOutputV1 {
    fn embeddings(&self) -> Array2<f32> {
        self.embeddings.clone()
    }
    fn byte_range(&self, index: usize) -> Option<ByteRange> {
        self.byte_ranges.get(index).cloned()
    }
    fn byte_ranges(&self) -> Vec<ByteRange> {
        self.byte_ranges.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(ModelOutputAPI)]
pub enum ModelOutput {
    V1(ModelOutputV1),
}

impl ModelOutput {
    pub fn validate(
        &self,
        expected_embedding_dim: usize,
        expected_byte_len: u64,
    ) -> ModelResult<()> {
        // 1. Verify embedding dimensions
        let embeddings = self.embeddings();
        let embedding_shape = embeddings.shape();
        if embedding_shape.len() != 2 || embedding_shape[1] != expected_embedding_dim {
            return Err(ModelError::ValidationError(
                "Embeddings must have the expected dimension".to_string(),
            ));
        }

        let byte_ranges = self.byte_ranges();

        // 2. Check that number of byte ranges matches number of embeddings
        if embedding_shape[0] != byte_ranges.len() {
            return Err(ModelError::ValidationError(
                "Number of byte ranges must match number of embeddings".to_string(),
            ));
        }

        // 3. Validate byte ranges
        let mut total_bytes: u64 = 0;
        let mut prev_end: Option<u64> = None;

        for (i, range) in byte_ranges.iter().enumerate() {
            // Check that range is valid (at least one byte)
            if range.start > range.end || range.end.wrapping_sub(range.start) < 1 {
                return Err(ModelError::ValidationError(
                    "Each byte range must represent at least one byte".to_string(),
                ));
            }

            // Check contiguity
            if i == 0 {
                // First range must start at 0
                if range.start != 0 {
                    return Err(ModelError::ValidationError(
                        "First byte range must start at 0".to_string(),
                    ));
                }
            } else {
                // Subsequent ranges must start at prev_end + 1
                if let Some(prev) = prev_end {
                    if range.start != prev + 1 {
                        return Err(ModelError::ValidationError(
                            "Byte ranges must be contiguous".to_string(),
                        ));
                    }
                }
            }

            // Update total bytes and previous end
            total_bytes = total_bytes.checked_add(range.end - range.start).ok_or(
                ModelError::ValidationError("Byte range length overflow".to_string()),
            )?;
            prev_end = Some(range.end);
        }

        // 4. Verify total byte length
        if total_bytes != expected_byte_len {
            return Err(ModelError::ValidationError(
                "Total byte length does not match expected length".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_model_input_serde() {
        // Create a sample ModelInput
        let input = ModelInput::V1(ModelInputV1 {
            input: "test".to_string().into(),
        });

        // Serialize to JSON
        let serialized = serde_json::to_string(&input).expect("Serialization failed");
        println!("{}", serialized);

        // Deserialize back to ModelInput
        let deserialized: ModelInput =
            serde_json::from_str(&serialized).expect("Deserialization failed");
        assert_eq!(deserialized, input);
    }

    #[test]
    fn test_model_output_serde() {
        let output = ModelOutput::V1(ModelOutputV1 {
            embeddings: array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            byte_ranges: vec![
                ByteRange { start: 0, end: 3 },
                ByteRange { start: 4, end: 8 },
                ByteRange { start: 9, end: 10 },
            ],
        });

        // Serialize to JSON
        let serialized = serde_json::to_string(&output).expect("Serialization failed");
        println!("{}", serialized);

        // Deserialize back to ModelInput
        let deserialized: ModelOutput =
            serde_json::from_str(&serialized).expect("Deserialization failed");
        assert_eq!(deserialized, output);
    }
}
