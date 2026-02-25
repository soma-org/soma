// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::data::dataset::Dataset;
use std::sync::Arc;

pub const PAD_TOKEN_ID: u16 = 256;
pub const EOS_TOKEN_ID: u16 = 257;

#[derive(Debug, Clone)]
pub struct ByteSequenceItem {
    pub token_ids: Vec<u16>,
    pub pos_ids: Vec<u32>,
}

pub struct ByteSequenceDataset {
    pub seq_len: usize,
    pub buffer: Arc<[u8]>,
}

impl ByteSequenceDataset {
    pub fn new(seq_len: usize, buffer: Arc<[u8]>) -> Self {
        Self { seq_len, buffer }
    }
}

impl Dataset<ByteSequenceItem> for ByteSequenceDataset {
    fn len(&self) -> usize {
        if self.buffer.is_empty() {
            return 0;
        }
        self.buffer.len().div_ceil(self.seq_len)
    }

    fn get(&self, index: usize) -> Option<ByteSequenceItem> {
        let start = index.checked_mul(self.seq_len)?;
        if start >= self.buffer.len() {
            return None;
        }

        let remaining_bytes = self.buffer.len() - start;
        let data_len = remaining_bytes.min(self.seq_len);

        // Place EOS only if final chunk AND we have space after the real data
        let is_final = index + 1 == self.len();
        let has_room = data_len < self.seq_len;
        let eos_slot = if is_final && has_room { Some(data_len) } else { None };

        // token_ids
        let mut token_ids = Vec::with_capacity(self.seq_len);
        for i in 0..self.seq_len {
            let tok = if i < data_len {
                self.buffer[start + i] as u16
            } else if eos_slot == Some(i) {
                EOS_TOKEN_ID
            } else {
                PAD_TOKEN_ID
            };
            token_ids.push(tok);
        }

        // pos_ids
        let mut pos_ids = Vec::with_capacity(self.seq_len);
        let pos_after_last = (start + data_len) as u32;
        for i in 0..self.seq_len {
            let pos = if i < data_len { (start + i) as u32 } else { pos_after_last };
            pos_ids.push(pos);
        }

        Some(ByteSequenceItem { token_ids, pos_ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_buffer(size: usize) -> Arc<[u8]> {
        let vec: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        Arc::from(vec.as_slice())
    }

    #[test]
    fn test_dataset_length() {
        let cases = vec![(0, 0), (1, 1), (9, 1), (10, 1), (11, 2), (20, 2), (21, 3)];

        for (size, expected_len) in cases {
            let data = ByteSequenceDataset { seq_len: 10, buffer: create_test_buffer(size) };
            assert_eq!(
                data.len(),
                expected_len,
                "size {} → expected {} chunks, got {}",
                size,
                expected_len,
                data.len()
            );
        }
    }

    #[test]
    fn test_full_chunk_no_eos() {
        let seq = 8;
        let data = ByteSequenceDataset { seq_len: seq, buffer: create_test_buffer(16) };

        assert_eq!(data.len(), 2);

        let item = data.get(0).unwrap();
        for i in 0..seq {
            assert_eq!(item.token_ids[i], i as u16);
            assert_eq!(item.pos_ids[i], i as u32);
        }

        let item2 = data.get(1).unwrap();
        for i in 0..seq {
            let global = 8 + i;
            assert_eq!(item2.token_ids[i], global as u16);
            assert_eq!(item2.pos_ids[i], global as u32);
        }
        assert!(!item2.token_ids.contains(&EOS_TOKEN_ID));
    }

    #[test]
    fn test_last_chunk_with_eos_and_padding() {
        let seq = 6;
        let total_bytes = 10;
        let data = ByteSequenceDataset { seq_len: seq, buffer: create_test_buffer(total_bytes) };

        assert_eq!(data.len(), 2);

        let last = data.get(1).unwrap();
        for i in 0..4 {
            let global = 6 + i;
            assert_eq!(last.token_ids[i], global as u16);
            assert_eq!(last.pos_ids[i], global as u32);
        }

        assert_eq!(last.token_ids[4], EOS_TOKEN_ID);
        assert_eq!(last.pos_ids[4], 10);

        assert_eq!(last.token_ids[5], PAD_TOKEN_ID);
        assert_eq!(last.pos_ids[5], 10);
    }

    #[test]
    fn test_exact_multiple_no_eos() {
        let seq = 5;
        let total_bytes = 10;
        let data = ByteSequenceDataset { seq_len: seq, buffer: create_test_buffer(total_bytes) };

        let last = data.get(1).unwrap();
        for i in 0..5 {
            let global = 5 + i;
            assert_eq!(last.token_ids[i], global as u16);
            assert_eq!(last.pos_ids[i], global as u32);
        }
        assert!(!last.token_ids.contains(&EOS_TOKEN_ID));
    }

    #[test]
    fn test_empty_buffer() {
        let data = ByteSequenceDataset { seq_len: 16, buffer: create_test_buffer(0) };
        assert_eq!(data.len(), 0);
        assert!(data.get(0).is_none());
    }

    #[test]
    fn test_single_byte() {
        let seq = 4;
        let data = ByteSequenceDataset { seq_len: seq, buffer: create_test_buffer(1) };

        assert_eq!(data.len(), 1);
        let item = data.get(0).unwrap();
        assert_eq!(item.token_ids, vec![0, EOS_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]);
        assert_eq!(item.pos_ids, vec![0, 1, 1, 1]);
    }
}
