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
    evaluation::{EvaluationInput, EvaluationInputAPI, EvaluationOutput},
    metadata::MetadataAPI,
};
use std::sync::Arc;

// async fn load_data(
//         &self,
//         object_path: &ObjectPath,
//         download_metadata: &DownloadMetadata,
//         cancellation: CancellationToken,
//     ) -> ShardResult<()> {
//         if self
//             .ephemeral_store
//             .object_store()
//             .head(&object_path.path())
//             .await
//             .is_ok()
//         {
//             return Ok(());
//         }

//         // Check persistent.
//         if self
//             .persistent_store
//             .object_store()
//             .head(&object_path.path())
//             .await
//             .is_ok()
//         {
//             let reader = Arc::new(ObjectStoreReader::new(
//                 self.persistent_store.object_store().clone(),
//                 object_path.clone(),
//             ));
//             self.downloader
//                 .download(
//                     reader,
//                     self.ephemeral_store.object_store().clone(),
//                     object_path.clone(),
//                     download_metadata.metadata().clone(),
//                 )
//                 .await
//                 .map_err(ShardError::ObjectError)?;
//             return Ok(());
//         }

//         // In msim tests, data may be at a different path (e.g., uploads/xxx instead of inputs/xxx).
//         // Check if data exists at the URL path in the store and copy to expected location.
//         #[cfg(msim)]
//         {
//             use object_store::path::Path as ObjPath;
//             use types::error::ObjectError;

//             let url_path = ObjPath::from(download_metadata.url().path().trim_start_matches('/'));
//             if let Ok(result) = self.persistent_store.object_store().get(&url_path).await {
//                 if let Ok(bytes) = result.bytes().await {
//                     self.ephemeral_store
//                         .object_store()
//                         .put(&object_path.path(), bytes.into())
//                         .await
//                         .map_err(|e| ShardError::ObjectError(ObjectError::ObjectStoreError(e)))?;
//                     return Ok(());
//                 }
//             }
//         }

//         let reader = Arc::new(
//             self.object_http_client
//                 .get_reader(download_metadata)
//                 .await
//                 .map_err(ShardError::ObjectError)?,
//         );

//         self.downloader
//             .download(
//                 reader,
//                 self.ephemeral_store.object_store().clone(),
//                 object_path.clone(),
//                 download_metadata.metadata().clone(),
//             )
//             .await
//             .map_err(ShardError::ObjectError)?;

//         Ok(())
//     }
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
