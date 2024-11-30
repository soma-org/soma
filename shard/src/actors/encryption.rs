use std::{marker::PhantomData, sync::Arc};

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
    error::ShardResult,
    networking::blob::{http_network::BlobHttpClient, BlobNetworkClient, GET_OBJECT_TIMEOUT},
    storage::blob::{BlobCompression, BlobEncryption, BlobPath},
    types::{
        checksum::Checksum,
        manifest::{Batch, BatchAPI},
        network_committee::NetworkingIndex,
    },
};
use async_trait::async_trait;

use super::{ActorMessage, Processor};

pub(crate) struct Encryptor<K, B: BlobEncryption<K>> {
    encryptor: Arc<B>,
    marker: PhantomData<K>,
}

pub(crate) enum EncryptionInput<K> {
    Encrypt(K, Bytes),
    Decrypt(K, Bytes),
}

#[async_trait]
impl<K: Sync + Send + 'static, B: BlobEncryption<K>> Processor for Encryptor<K, B> {
    type Input = EncryptionInput<K>;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let encryptor: Arc<B> = self.encryptor.clone();

        let _ = tokio::task::spawn_blocking(move || match msg.input {
            EncryptionInput::Encrypt(key, contents) => {
                let encrypted = encryptor.encrypt(key, contents);
                let _ = msg.sender.send(Ok(encrypted));
            }
            EncryptionInput::Decrypt(key, contents) => {
                let decrypted = encryptor.decrypt(key, contents);
                let _ = msg.sender.send(Ok(decrypted));
            }
        })
        .await;
        // NOTE: await here causes the program to only operate sequentially because the actor run loop
        // also awaits when calling a processors process fn
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
