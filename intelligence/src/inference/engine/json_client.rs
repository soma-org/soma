
use async_trait::async_trait;
use object_store::local::LocalFileSystem;
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use types::{
    error::{InferenceError, InferenceResult},
    shard_crypto::keys::EncoderPublicKey,
};
use url::Url;

pub struct JsonModuleClient {
    url: Url,
    client: Client,
    storage: Arc<LocalFileSystem>,
    own_encoder: EncoderPublicKey,
}

impl JsonModuleClient {
    pub fn new(
        url: Url,
        storage: Arc<LocalFileSystem>,
        own_encoder: EncoderPublicKey,
    ) -> InferenceResult<Self> {
        Ok(Self {
            url,
            client: ClientBuilder::new()
                .pool_idle_timeout(None)
                .build()
                .map_err(InferenceError::ReqwestError)?,
            storage,
            own_encoder,
        })
    }
}

#[derive(Serialize)]
struct JsonCallRequest {
    input_path: String,
    output_path: String,
}

#[derive(Deserialize)]
struct JsonCallResponse {
    probe_encoder: Option<EncoderPublicKey>,
}

#[async_trait]
impl ModuleClient<LocalFileSystem> for JsonModuleClient {
    async fn call(&self, input: ModuleInput, timeout: Duration) -> InferenceResult<ModuleOutput> {
        let input_filepath = self
            .storage
            .path_to_filesystem(&input.input_path().path())
            .map_err(InferenceError::ObjectStoreError)?
            .to_string_lossy()
            .into_owned();

        let output_filepath = self
            .storage
            .path_to_filesystem(&input.output_path().path())
            .map_err(InferenceError::ObjectStoreError)?
            .to_string_lossy()
            .into_owned();

        let request_data = JsonCallRequest {
            input_path: input_filepath,
            output_path: output_filepath,
        };
        let response = self
            .client
            .post(self.url.clone())
            .timeout(timeout)
            .json(&request_data)
            .send()
            .await
            .map_err(InferenceError::ReqwestError)?;

        let response_data: JsonCallResponse = response
            .json()
            .await
            .map_err(InferenceError::ReqwestError)?;

        if let Some(probe_encoder) = response_data.probe_encoder {
            return Ok(ModuleOutput::V1(ModuleOutputV1::new(probe_encoder)));
        }

        return Ok(ModuleOutput::V1(ModuleOutputV1::new(
            self.own_encoder.clone(),
        )));
    }
}



use std::{marker::PhantomData, sync::Arc, time::Duration};

use async_trait::async_trait;
use fastcrypto::hash::HashFunction;
use futures::StreamExt;
use object_store::ObjectStore;
use objects::{
    downloader::ObjectDownloader,
    readers::{store::ObjectStoreReader, url::ObjectHttpClient},
    stores::{EphemeralStore, PersistentStore},
};
use tokio_util::sync::CancellationToken;
use types::{
    actors::{ActorMessage, Processor},
    checksum::Checksum,
    crypto::DefaultHash,
    error::{ShardError, ShardResult},
    metadata::{DownloadMetadata, Metadata, MetadataV1, ObjectPath},
};

use crate::inference::{
    InferenceInput, InferenceInputAPI, InferenceOutput, InferenceOutputV1, engine::{InferenceEngineAPI, ModuleClient, ModuleInput, ModuleInputV1, ModuleOutputAPI}
};

pub struct InferenceWorkQueue<
    I: InferenceEngineAPI,
> {
    inference_engine: Arc<I>,
}

impl<

    I: InferenceEngineAPI,
    > InferenceWorkQueue<I>
{
    pub fn new(
    inference_engine: Arc<I>,

    ) -> Self {
        Self {
            inference_engine,
        }
    }

    async fn load_data(
        &self,
        object_path: &ObjectPath,
        download_metadata: &DownloadMetadata,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        if self
            .ephemeral_store
            .object_store()
            .head(&object_path.path())
            .await
            .is_ok()
        {
            return Ok(());
        }

        // Check persistent.
        if self
            .persistent_store
            .object_store()
            .head(&object_path.path())
            .await
            .is_ok()
        {
            let reader = Arc::new(ObjectStoreReader::new(
                self.persistent_store.object_store().clone(),
                object_path.clone(),
            ));
            self.downloader
                .download(
                    reader,
                    self.ephemeral_store.object_store().clone(),
                    object_path.clone(),
                    download_metadata.metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;
            return Ok(());
        }

        // In msim tests, data may be at a different path (e.g., uploads/xxx instead of inputs/xxx).
        // Check if data exists at the URL path in the store and copy to expected location.
        #[cfg(msim)]
        {
            use object_store::path::Path as ObjPath;
            use types::error::ObjectError;

            let url_path = ObjPath::from(download_metadata.url().path().trim_start_matches('/'));
            if let Ok(result) = self.persistent_store.object_store().get(&url_path).await {
                if let Ok(bytes) = result.bytes().await {
                    self.ephemeral_store
                        .object_store()
                        .put(&object_path.path(), bytes.into())
                        .await
                        .map_err(|e| ShardError::ObjectError(ObjectError::ObjectStoreError(e)))?;
                    return Ok(());
                }
            }
        }

        let reader = Arc::new(
            self.object_http_client
                .get_reader(download_metadata)
                .await
                .map_err(ShardError::ObjectError)?,
        );

        self.downloader
            .download(
                reader,
                self.ephemeral_store.object_store().clone(),
                object_path.clone(),
                download_metadata.metadata().clone(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

        Ok(())
    }

    async fn store_to_persistent(
        &self,
        object_path: &ObjectPath,
        metadata: &Metadata,
    ) -> ShardResult<()> {
        let reader = Arc::new(ObjectStoreReader::new(
            self.ephemeral_store.object_store().clone(),
            object_path.clone(),
        ));
        self.downloader
            .download(
                reader,
                self.persistent_store.object_store().clone(),
                object_path.clone(),
                metadata.clone(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

        Ok(())
    }
}

#[async_trait]
impl<
        PS: ObjectStore,
        ES: ObjectStore,
        P: PersistentStore<PS>,
        E: EphemeralStore<ES>,
        M: ModuleClient<ES>,
    > Processor for InferenceWorkQueue<PS, ES, P, E, M>
{
    type Input = InferenceInput;
    type Output = InferenceOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;
            self.load_data(
                input.input_object_path(),
                input.input_download_metadata(),
                msg.cancellation.clone(),
            )
            .await?;

            let tmp_output_path = ObjectPath::new_tmp(input.epoch(), input.shard_digest().clone());

            let module_input = ModuleInput::V1(ModuleInputV1::new(
                input.epoch(),
                input.input_object_path().clone(),
                tmp_output_path.clone(),
            ));

            let timeout = Duration::from_secs(10);

            let module_output = self
                .module_client
                .call(module_input, timeout)
                .await
                .map_err(ShardError::InferenceError)?;

            self.store_to_persistent(
                input.input_object_path(),
                input.input_download_metadata().metadata(),
            )
            .await?;

            let get_output_result = self
                .ephemeral_store
                .object_store()
                .get(&tmp_output_path.path())
                .await
                .map_err(ShardError::ObjectStoreError)?;

            let output_size = get_output_result.meta.size;
            let mut output_hasher = DefaultHash::new();
            let mut output_stream = get_output_result.into_stream();

            while let Some(chunk) = output_stream.next().await {
                let bytes = chunk.map_err(ShardError::ObjectStoreError)?;
                output_hasher.update(&bytes);
            }

            let output_checksum = Checksum::new_from_hash(output_hasher.finalize().into());
            let output_path = ObjectPath::Embeddings(
                input.epoch(),
                input.shard_digest().clone(),
                output_checksum,
            );

            self.ephemeral_store
                .object_store()
                .copy(&tmp_output_path.path(), &output_path.path())
                .await
                .map_err(ShardError::ObjectStoreError)?;

            let output_metadata = Metadata::V1(MetadataV1::new(output_checksum, output_size));
            self.store_to_persistent(&output_path, &output_metadata)
                .await?;

            let input_download_metadata = self
                .persistent_store
                .download_metadata(
                    input.input_object_path().clone(),
                    input.input_download_metadata().metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;

            let output_download_metadata = self
                .persistent_store
                .download_metadata(output_path.clone(), output_metadata)
                .await
                .map_err(ShardError::ObjectError)?;

            Ok(InferenceOutput::V1(InferenceOutputV1::new(
                input_download_metadata,
                input.input_object_path().clone(),
                output_download_metadata,
                output_path,
                module_output.probe_encoder().clone(),
            )))
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
