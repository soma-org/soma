use crate::{evaluation::evaluators::EvaluatorAPI, safetensor_format::IndexedTensors};
use async_trait::async_trait;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::store::ModuleSnapshot;
use burn::store::SafetensorsStore;
use object_store::ObjectStore;
use objects::downloader::ObjectDownloader;
use objects::readers::url::ObjectHttpClient;
use objects::stores::EphemeralStore;
use objects::stores::PersistentStore;
use probes::v1::probe::{Probe, ProbeConfig};
use std::{marker::PhantomData, time::Duration};
use types::{
    error::EvaluationResult,
};
use std::sync::Arc;

pub struct Evaluator<ES: ObjectStore, PS: ObjectStore, E: EphemeralStore<ES>, P: PersistentStore<PS>> {
    persistent_store: P,
    ephemeral_store: E,
    downloader: Arc<ObjectDownloader>,
    object_http_client: ObjectHttpClient,
    es_marker: PhantomData<ES>,
    ps_marker: PhantomData<PS>
}

impl<ES: ObjectStore, PS: ObjectStore, E: EphemeralStore<ES>, P: PersistentStore<PS>> Evaluator<ES, PS, E, P> {
    pub fn new(
        persistent_store: P,
        ephemeral_store: E,
        downloader: Arc<ObjectDownloader>,
        object_http_client: ObjectHttpClient,
    ) -> Self {
        Self {
            persistent_store,
            ephemeral_store,
            downloader,
            object_http_client,
            es_marker: PhantomData,
            ps_marker: PhantomData,
        }
    }
}

#[async_trait]
impl<ES: ObjectStore, PS: ObjectStore, E: EphemeralStore<ES>, P: PersistentStore<PS>> EvaluatorAPI for Evaluator<ES, PS, E, P> {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput> {

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
