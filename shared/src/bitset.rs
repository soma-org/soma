use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSet<T, const N: usize>([u8; N], PhantomData<T>);

impl<T, const N: usize> BitSet<T, N>
where
    T: From<usize> + Into<usize> + Copy, // Added Copy trait bound
{
    pub fn new(indices: &[T]) -> Self {
        let mut bits = [0u8; N];
        for idx in indices {
            let idx_usize: usize = (*idx).into();
            let byte_idx = idx_usize / 8;
            let bit_idx = idx_usize % 8;
            bits[byte_idx] |= 1 << bit_idx;
        }
        Self(bits, PhantomData)
    }

    pub fn get_indices(&self) -> Vec<T> {
        let mut indices = Vec::new();
        for (byte_idx, &byte) in self.0.iter().enumerate() {
            for bit_idx in 0..8 {
                if byte & (1 << bit_idx) != 0 {
                    indices.push(T::from(byte_idx * 8 + bit_idx));
                }
            }
        }
        indices
    }
}

// Custom Serialize implementation
impl<T, const N: usize> Serialize for BitSet<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

// Custom Deserialize implementation
impl<'de, T, const N: usize> Deserialize<'de> for BitSet<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Use Vec<u8> as intermediate type since arrays don't implement Deserialize
        let vec = Vec::<u8>::deserialize(deserializer)?;
        if vec.len() != N {
            return Err(serde::de::Error::custom(format!(
                "expected array of length {}, got {}",
                N,
                vec.len()
            )));
        }
        let mut bits = [0u8; N];
        bits.copy_from_slice(&vec);
        Ok(Self(bits, PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset() {
        let indices = vec![0usize, 7, 8, 254];
        let bitset = BitSet::<usize, 32>::new(&indices);
        // First byte should be 10000001
        assert_eq!(bitset.0[0], 0b10000001);
        // Second byte should be 00000001
        assert_eq!(bitset.0[1], 0b00000001);
        // Last byte should be 01000000
        assert_eq!(bitset.0[31], 0b01000000);
    }

    #[test]
    fn test_bitset_operations() {
        let indices = vec![0usize, 8, 15];
        let bitset = BitSet::<usize, 2>::new(&indices);

        let retrieved_indices = bitset.get_indices();
        assert_eq!(indices, retrieved_indices);
    }

    #[test]
    fn test_serde() {
        let indices = vec![0usize, 8, 15];
        let bitset = BitSet::<usize, 2>::new(&indices);

        // Test serialization
        let serialized = bcs::to_bytes(&bitset).unwrap();

        // Test deserialization
        let deserialized: BitSet<usize, 2> = bcs::from_bytes(&serialized).unwrap();

        assert_eq!(bitset.get_indices(), deserialized.get_indices());
    }
}
