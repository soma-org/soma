//! BCS-compatible wrappers for floating-point types.
//!
//! BCS (Binary Canonical Serialization) does not support floating-point types directly.
//! This module provides wrapper types that serialize f32 values as raw bytes (little-endian
//! IEEE 754), making them BCS-compatible while preserving the full precision.
//!
//! Types:
//! - `BcsF32`: Simple f32 wrapper for single values (used in ProtocolConfig)
//! - `SomaTensor`: TensorData wrapper for multi-dimensional arrays (used in transactions)

use burn::tensor::TensorData;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::{Hash, Hasher};

// ============================================================================
// BcsF32 - Simple f32 wrapper for BCS serialization
// ============================================================================

/// A BCS-compatible f32 wrapper.
///
/// Stores f32 as raw bytes (little-endian IEEE 754) since BCS doesn't support floats.
/// Used in ProtocolConfig for threshold values that need to be BCS-serializable.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, JsonSchema)]
#[schemars(transparent)]
pub struct BcsF32(#[schemars(with = "f32")] pub f32);

impl BcsF32 {
    pub fn new(value: f32) -> Self {
        Self(value)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Serialize for BcsF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            // Human-readable formats (YAML, JSON): serialize as f32
            self.0.serialize(serializer)
        } else {
            // Binary formats (BCS): serialize as raw bytes (4 bytes, little-endian)
            let bytes = self.0.to_le_bytes();
            bytes.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for BcsF32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // Human-readable formats (YAML, JSON): deserialize as f32
            let value = f32::deserialize(deserializer)?;
            Ok(Self(value))
        } else {
            // Binary formats (BCS): deserialize from raw bytes
            let bytes: [u8; 4] = <[u8; 4]>::deserialize(deserializer)?;
            Ok(Self(f32::from_le_bytes(bytes)))
        }
    }
}

impl From<f32> for BcsF32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<BcsF32> for f32 {
    fn from(value: BcsF32) -> Self {
        value.0
    }
}

impl std::fmt::Display for BcsF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for BcsF32 {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<f32>().map(BcsF32)
    }
}

// ============================================================================
// SomaTensor - TensorData wrapper for BCS serialization
// ============================================================================

/// BCS-compatible serialization format for SomaTensor.
/// Stores f32 values as raw bytes since BCS doesn't support floats.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SomaTensorBcs {
    /// Raw bytes of the f32 values (little-endian IEEE 754)
    bytes: Vec<u8>,
    /// Shape of the tensor
    shape: Vec<usize>,
}

/// Wrapper around Burn's TensorData that implements Hash for transaction serialization.
///
/// Used for embeddings and tensor data that needs to be included in transactions.
/// The underlying TensorData stores shape and bytes, making it compatible with
/// Burn's tensor operations and the CompetitionAPI.
///
/// Serialization uses a BCS-compatible format that stores f32 values as raw bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct SomaTensor(pub TensorData);

impl Eq for SomaTensor {}

impl Serialize for SomaTensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let values = self.to_vec();
        // Convert f32 values to raw bytes (little-endian)
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bcs_format = SomaTensorBcs { bytes, shape: self.0.shape.clone() };
        bcs_format.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SomaTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bcs_format = SomaTensorBcs::deserialize(deserializer)?;
        // Convert raw bytes back to f32 values
        let values: Vec<f32> = bcs_format
            .bytes
            .chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().expect("Invalid byte chunk size");
                f32::from_le_bytes(arr)
            })
            .collect();
        Ok(SomaTensor::new(values, bcs_format.shape))
    }
}

impl SomaTensor {
    /// Create a new SomaTensor from f32 values and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self(TensorData::new(data, shape))
    }

    /// Create a SomaTensor from Burn's TensorData.
    pub fn from_tensor_data(td: TensorData) -> Self {
        Self(td)
    }

    /// Consume self and return the inner TensorData.
    pub fn into_tensor_data(self) -> TensorData {
        self.0
    }

    /// Get a reference to the inner TensorData.
    pub fn as_tensor_data(&self) -> &TensorData {
        &self.0
    }

    /// Get the total number of elements (product of shape dimensions).
    pub fn dim(&self) -> usize {
        self.0.shape.iter().product()
    }

    /// Get the total number of elements (alias for dim()).
    pub fn len(&self) -> usize {
        self.dim()
    }

    /// Returns true if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.dim() == 0
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Get the underlying data as f32 values.
    ///
    /// # Panics
    /// Panics if the underlying data is not f32 (should never happen for SomaTensor).
    pub fn to_vec(&self) -> Vec<f32> {
        self.0.to_vec::<f32>().expect("SomaTensor always stores f32")
    }

    /// Create a scalar SomaTensor from a single f32 value.
    pub fn scalar(value: f32) -> Self {
        Self::new(vec![value], vec![1])
    }

    /// Extract a scalar f32 value from a shape-[1] tensor.
    ///
    /// # Panics
    /// Panics if the tensor is not a scalar (shape != [1]).
    pub fn as_scalar(&self) -> f32 {
        assert_eq!(self.0.shape, vec![1], "SomaTensor is not a scalar");
        self.to_vec()[0]
    }

    /// Create a SomaTensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }
}

/// Hash implementation using byte representation.
///
/// This is deterministic for identical f32 values since we hash the raw bytes.
/// The hash includes both shape and data to ensure different-shaped tensors
/// with the same data produce different hashes.
impl Hash for SomaTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.shape.hash(state);
        // Hash the raw bytes of the tensor data
        self.0.as_bytes().hash(state);
    }
}

impl Default for SomaTensor {
    fn default() -> Self {
        Self::zeros(vec![0])
    }
}

impl std::fmt::Display for SomaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values = self.to_vec();
        if values.len() == 1 {
            // Scalar tensor - just display the value
            write!(f, "{}", values[0])
        } else if values.len() <= 8 {
            // Small tensor - display all values
            write!(
                f,
                "[{}]",
                values.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(", ")
            )
        } else {
            // Large tensor - display first few and last few
            let first: Vec<_> = values.iter().take(3).map(|v| format!("{:.6}", v)).collect();
            let last: Vec<_> =
                values.iter().rev().take(3).rev().map(|v| format!("{:.6}", v)).collect();
            write!(f, "[{}, ..., {}] (shape: {:?})", first.join(", "), last.join(", "), self.shape())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_soma_tensor_creation() {
        let tensor = SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(tensor.dim(), 3);
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_soma_tensor_scalar() {
        let tensor = SomaTensor::scalar(42.0);
        assert_eq!(tensor.dim(), 1);
        assert_eq!(tensor.shape(), &[1]);
        assert_eq!(tensor.as_scalar(), 42.0);
    }

    #[test]
    fn test_soma_tensor_hash_equality() {
        use std::collections::hash_map::DefaultHasher;

        let tensor1 = SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor2 = SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor3 = SomaTensor::new(vec![1.0, 2.0, 4.0], vec![3]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        tensor1.hash(&mut hasher1);
        tensor2.hash(&mut hasher2);
        tensor3.hash(&mut hasher3);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();
        let hash3 = hasher3.finish();

        // Same content should produce same hash
        assert_eq!(hash1, hash2);
        // Different content should (almost certainly) produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_soma_tensor_zeros() {
        let tensor = SomaTensor::zeros(vec![2, 3]);
        assert_eq!(tensor.dim(), 6);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn test_soma_tensor_serialization() {
        let tensor = SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let serialized = bcs::to_bytes(&tensor).unwrap();
        let deserialized: SomaTensor = bcs::from_bytes(&serialized).unwrap();
        assert_eq!(tensor, deserialized);
    }

    #[test]
    fn test_soma_tensor_conversion() {
        let tensor = SomaTensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor_data = tensor.clone().into_tensor_data();
        let back = SomaTensor::from_tensor_data(tensor_data);
        assert_eq!(tensor, back);
    }

    #[test]
    fn test_soma_tensor_scalar_serialization() {
        let tensor = SomaTensor::scalar(std::f32::consts::PI);
        let serialized = bcs::to_bytes(&tensor).unwrap();
        let deserialized: SomaTensor = bcs::from_bytes(&serialized).unwrap();
        assert_eq!(tensor.as_scalar(), deserialized.as_scalar());
    }

    #[test]
    fn test_soma_tensor_special_values() {
        // Test that special float values serialize correctly
        let tensor = SomaTensor::new(vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY], vec![4]);
        let serialized = bcs::to_bytes(&tensor).unwrap();
        let deserialized: SomaTensor = bcs::from_bytes(&serialized).unwrap();
        let original = tensor.to_vec();
        let restored = deserialized.to_vec();
        assert_eq!(original[0], restored[0]); // 0.0
        assert_eq!(original[2], restored[2]); // INFINITY
        assert_eq!(original[3], restored[3]); // NEG_INFINITY
    }
}
