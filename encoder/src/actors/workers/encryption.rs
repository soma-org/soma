use std::{marker::PhantomData, sync::Arc};

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
     encryption::Encryptor, error::ShardResult, networking::blob::{http_network::ObjectHttpClient, ObjectNetworkClient, GET_OBJECT_TIMEOUT}, storage::object::{ObjectPath}, types::{checksum::Checksum, network_committee::NetworkingIndex}
};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct EncryptionProcessor<K, E: Encryptor<K>> {
    encryptor: Arc<E>,
    marker: PhantomData<K>,
}

impl<K, E: Encryptor<K>> EncryptionProcessor<K, E> {
    pub(crate) fn new(encryptor: Arc<E>) -> Self {
        Self {
            encryptor,
            marker: PhantomData,
        }
    }
}

pub(crate) enum EncryptionInput<K> {
    Encrypt(K, Bytes),
    Decrypt(K, Bytes),
}

#[async_trait]
impl<K: Sync + Send + 'static, E: Encryptor<K>> Processor for EncryptionProcessor<K, E> {
    type Input = EncryptionInput<K>;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let encryptor: Arc<E> = self.encryptor.clone();

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
