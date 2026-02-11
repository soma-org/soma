use burn::{
    Tensor,
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Int, TensorData},
};

use crate::v1::data::dataset::ByteSequenceItem;

#[derive(Clone, Default)]
pub struct ByteSequenceBatcher {}

impl ByteSequenceBatcher {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug)]
pub struct ByteSequenceBatch<B: Backend> {
    pub token_ids: Tensor<B, 2, Int>,
    pub pos_ids: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, ByteSequenceItem, ByteSequenceBatch<B>> for ByteSequenceBatcher {
    fn batch(&self, items: Vec<ByteSequenceItem>, device: &B::Device) -> ByteSequenceBatch<B> {
        let token_tensors = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(TensorData::from(item.token_ids.as_slice()), device)
                    .unsqueeze()
            })
            .collect::<Vec<_>>();

        let pos_tensors = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(TensorData::from(item.pos_ids.as_slice()), device)
                    .unsqueeze()
            })
            .collect::<Vec<_>>();

        let token_ids = Tensor::cat(token_tensors, 0).to_device(device);
        let pos_ids = Tensor::cat(pos_tensors, 0).to_device(device);

        ByteSequenceBatch { token_ids, pos_ids }
    }
}

#[cfg(test)]
mod tests {
    use crate::v1::data::dataset::ByteSequenceDataset;

    use super::*;
    use burn::{backend::NdArray, data::dataset::Dataset, tensor::Device};
    use std::sync::Arc;

    type TestBackend = NdArray<f32>;

    fn create_test_buffer(size: usize) -> Arc<[u8]> {
        let vec: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        Arc::from(vec.as_slice())
    }

    #[test]
    fn test_dataset_length() {
        let seq = 5;
        let total_bytes = 10;
        let data = ByteSequenceDataset { seq_len: seq, buffer: create_test_buffer(total_bytes) };

        let device: Device<TestBackend> = Default::default();
        let byte_sequence_batcher = ByteSequenceBatcher::new();

        let items: Vec<ByteSequenceItem> = (0..data.len()).filter_map(|i| data.get(i)).collect();

        assert_eq!(items.len(), 2, "Should produce exactly 2 items for 10 bytes + seq_len=5");
        assert_eq!(items[0].token_ids.len(), seq, "Each item should have full sequence length");
        assert_eq!(items[1].token_ids.len(), seq, "Each item should have full sequence length");

        let batch: ByteSequenceBatch<TestBackend> = byte_sequence_batcher.batch(items, &device);

        // Basic shape checks
        assert_eq!(batch.token_ids.dims().len(), 2, "token_ids should be 2D tensor");
        assert_eq!(batch.pos_ids.dims().len(), 2, "pos_ids should be 2D tensor");

        let expected_shape = [2usize, seq];
        assert_eq!(
            batch.token_ids.dims(),
            expected_shape,
            "token_ids shape should be [batch=2, seq_len=5]"
        );
        assert_eq!(
            batch.pos_ids.dims(),
            expected_shape,
            "pos_ids shape should be [batch=2, seq_len=5]"
        );

        let token_data: Vec<i64> = batch.token_ids.to_data().to_vec::<i64>().unwrap();
        let pos_data: Vec<i64> = batch.pos_ids.to_data().to_vec::<i64>().unwrap();

        // token_ids should be [[0,1,2,3,4], [5,6,7,8,9]]
        assert_eq!(
            token_data,
            vec![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "token_ids values don't match expected sequence"
        );

        // pos_ids should be exactly the same as token_ids in this test case
        assert_eq!(
            pos_data,
            vec![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "pos_ids values don't match expected positions"
        );
    }
}
