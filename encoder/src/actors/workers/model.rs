use crate::actors::{ActorMessage, Processor};
use crate::{
    error::ShardResult,
    intelligence::model::{python::PythonModule, Model},
};
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
impl<M: Model> Processor for ModelProcessor<M> {
    type Input = ArrayD<f32>;
    type Output = ArrayD<f32>;

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Some(sem) = &self.semaphore {
            if let Ok(permit) = sem.clone().acquire_owned().await {
                let model = self.model.clone();
                tokio::spawn(async move {
                    if let Ok(embeddings) = model.call(&msg.input).await {
                        let _ = msg.sender.send(Ok(embeddings));
                    }
                    drop(permit)
                });
            }
        } else {
            if let Ok(embeddings) = self.model.call(&msg.input).await {
                let _ = msg.sender.send(Ok(embeddings));
            }
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
