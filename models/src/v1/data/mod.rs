use std::sync::Arc;

use batcher::{ByteSequenceBatch, ByteSequenceBatcher};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    prelude::Backend,
};
use dataset::ByteSequenceDataset;

pub mod batcher;
pub mod dataset;

/// Build a [`DataLoader`] from a raw byte buffer.
pub fn build_data_loader<B: Backend>(
    buffer: Arc<[u8]>,
    max_seq_len: usize,
    batch_size: usize,
    num_workers: usize,
) -> Arc<dyn DataLoader<B, ByteSequenceBatch<B>> + Send + Sync + 'static> {
    let dataset = ByteSequenceDataset::new(max_seq_len, buffer);
    let batcher = ByteSequenceBatcher::new();
    DataLoaderBuilder::new(batcher).batch_size(batch_size).num_workers(num_workers).build(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn build_data_loader_produces_correct_batches() {
        let buffer: Arc<[u8]> = Arc::from(vec![0u8; 20].as_slice());
        let loader = build_data_loader::<B>(buffer, 10, 2, 1);
        let mut iter = loader.iter();
        let batch = iter.next().unwrap();
        // 20 bytes / seq_len 10 = 2 items, batch_size 2 → one batch of shape [2, 10]
        assert_eq!(batch.token_ids.dims(), [2, 10]);
        assert_eq!(batch.pos_ids.dims(), [2, 10]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn build_data_loader_partial_last_batch() {
        let buffer: Arc<[u8]> = Arc::from(vec![0u8; 30].as_slice());
        let loader = build_data_loader::<B>(buffer, 10, 2, 1);
        let mut iter = loader.iter();
        // 30 bytes / seq_len 10 = 3 items, batch_size 2 → two batches: [2,10] then [1,10]
        let batch1 = iter.next().unwrap();
        assert_eq!(batch1.token_ids.dims(), [2, 10]);
        let batch2 = iter.next().unwrap();
        assert_eq!(batch2.token_ids.dims(), [1, 10]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn build_data_loader_empty_buffer() {
        let buffer: Arc<[u8]> = Arc::from(Vec::new().as_slice());
        let loader = build_data_loader::<B>(buffer, 10, 2, 1);
        let mut iter = loader.iter();
        assert!(iter.next().is_none());
    }
}
