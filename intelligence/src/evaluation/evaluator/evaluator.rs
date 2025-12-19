use crate::{evaluation::evaluator::EvaluatorClient, safetensor_format::IndexedTensors};
use async_trait::async_trait;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::store::ModuleSnapshot;
use burn::store::SafetensorsStore;
use object_store::ObjectStore;
use objects::stores::EphemeralStore;
use probes::v1::probe::{Probe, ProbeConfig};
use std::{marker::PhantomData, time::Duration};
use types::{
    error::EvaluationResult,
    evaluation::{EvaluationInput, EvaluationInputAPI, EvaluationOutput},
    metadata::MetadataAPI,
};

pub struct Evaluator<ES: ObjectStore, E: EphemeralStore<ES>> {
    ephemeral_store: E,
    e_marker: PhantomData<ES>,
}

impl<ES: ObjectStore, E: EphemeralStore<ES>> Evaluator<ES, E> {
    pub fn new(ephemeral_store: E) -> Self {
        Self {
            ephemeral_store,
            e_marker: PhantomData,
        }
    }
}

#[async_trait]
impl<ES: ObjectStore, E: EphemeralStore<ES>> EvaluatorClient for Evaluator<ES, E> {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput> {
        let buffer = self
            .ephemeral_store
            .buffer_object(input.embedding_object_path().clone())
            .await
            .unwrap();

        let (_size, metadata) = safetensors::SafeTensors::read_metadata(buffer.as_ref()).unwrap();
        let st = safetensors::SafeTensors::deserialize(buffer.as_ref()).unwrap();

        let indexed_tensors = IndexedTensors::new(
            metadata,
            st,
            input.input_download_metadata().metadata().size() as u64,
        )
        .unwrap();

        let probe_buffer = self
            .ephemeral_store
            .buffer_object(input.probe_object_path().clone())
            .await
            .unwrap();

        let mut store = SafetensorsStore::from_bytes(Some(probe_buffer.as_ref().to_vec()));
        let device = WgpuDevice::default();
        let mut model: Probe<Wgpu> = ProbeConfig::new().init(&device);
        model.load_from(&mut store).unwrap();

        // Load probe model?
        // create a batch of tensors (x: t mixed latent with noise, y: linear interpolation)
        // forward pass the tensors
        // forward pass the latents with sig reg

        // figure out the number of tensors present
        // sliding window based on probe size context length
        // flow matching on the interpolation

        unimplemented!();
    }
}
