use super::{ActorMessage, Processor};
use crate::intelligence::model::{python::PythonModule, Model};
use async_trait::async_trait;
use numpy::ndarray::ArrayD;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub(crate) struct ModelProcessor<M: Model> {
    model: Arc<M>,
    semaphore: Option<Arc<Semaphore>>,
}

impl<M: Model> ModelProcessor<M> {
    pub fn new(model: M, concurrency: Option<usize>) -> Self {
        let semaphore = concurrency.map(|n| Arc::new(Semaphore::new(n)));
        Self {
            model: Arc::new(model),
            semaphore,
        }
    }
}

#[async_trait]
impl Processor for ModelProcessor<PythonModule> {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Some(sem) = &self.semaphore {
            let model = self.model.clone();
            if let Ok(permit) = sem.clone().acquire_owned().await {
                tokio::spawn(async move {
                    if let Ok(embeddings) = model.call(&msg.input).await {
                        let _ = msg.sender.send(embeddings);
                    }
                    drop(permit)
                });
            }
        } else {
            if let Ok(embeddings) = self.model.call(&msg.input).await {
                let _ = msg.sender.send(embeddings);
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
