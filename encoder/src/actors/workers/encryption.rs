use std::sync::Arc;

use bytes::Bytes;
use tokio::runtime::Handle;

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

        let _ = Handle::current()
            .spawn_blocking(move || match msg.input {
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
#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use shared::crypto::{Aes256IV, Aes256Key};
    use std::time::Duration;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    use crate::{
        actors::ActorManager, encryption::aes_encryptor::Aes256Ctr64LEEncryptor, error::ShardResult,
    };

    use super::*;

    #[tokio::test]
    async fn test_encryption_actor() -> ShardResult<()> {
        let encryptor = Aes256Ctr64LEEncryptor::new();
        let processor = EncryptionProcessor::new(Arc::new(encryptor));
        let manager = ActorManager::new(1, processor);
        let handle = manager.handle();
        let cancellation = CancellationToken::new();

        // Generate a random 256-bit key
        let key_bytes: [u8; 32] = [2u8; 32];
        let aes_key = Aes256Key::from(key_bytes);
        let aes_iv = Aes256IV {
            iv: [0u8; 16],
            key: aes_key,
        };

        let encryption_key = EncryptionKey::Aes256(aes_iv);

        // Generate random contents
        let contents = Bytes::from([0_u8; 1024].to_vec());

        let encrypted = handle
            .process(
                EncryptionInput::Encrypt(encryption_key.clone(), contents.clone()),
                cancellation.clone(),
            )
            .await?;

        let decrypted = handle
            .process(
                EncryptionInput::Decrypt(encryption_key.clone(), encrypted.clone()),
                cancellation.clone(),
            )
            .await?;

        // Assert that the roundtrip is identical
        assert_eq!(decrypted, contents);

        // Assert that encrypted does not match original contents as long as
        // contents is not an empty set.
        assert_ne!(encrypted, contents);

        manager.shutdown();
        sleep(Duration::from_millis(100)).await;
        Ok(())
    }
}
