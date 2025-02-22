use std::sync::Arc;

use bytes::Bytes;

use crate::error::ShardError;

use async_trait::async_trait;
use shared::crypto::{EncryptionKey, Encryptor};

use crate::actors::{ActorMessage, Processor};

pub(crate) struct EncryptionProcessor<E: Encryptor> {
    encryptor: Arc<E>,
}

impl<E: Encryptor> EncryptionProcessor<E> {
    pub(crate) fn new(encryptor: Arc<E>) -> Self {
        Self { encryptor }
    }
}

pub(crate) enum EncryptionInput {
    Encrypt(EncryptionKey, Bytes),
    Decrypt(EncryptionKey, Bytes),
}

#[async_trait]
impl<E: Encryptor> Processor for EncryptionProcessor<E> {
    type Input = EncryptionInput;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let encryptor: Arc<E> = self.encryptor.clone();

        let _ = tokio::task::spawn_blocking(move || match msg.input {
            EncryptionInput::Encrypt(key, contents) => {
                let encrypted = encryptor.encrypt(key, contents);
                let _ = msg
                    .sender
                    .send(encrypted.map_err(|_| ShardError::EncryptionFailed));
            }
            EncryptionInput::Decrypt(key, contents) => {
                let decrypted = encryptor.decrypt(key, contents);
                let _ = msg
                    .sender
                    .send(decrypted.map_err(|_| ShardError::EncryptionFailed));
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
