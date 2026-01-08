use std::{ sync::Arc, time::Duration};
use async_trait::async_trait;
use types::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
};

use crate::inference::{
    InferenceInput, InferenceOutput, engine::{InferenceEngineAPI}
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

}

#[async_trait]
impl<
    I: InferenceEngineAPI,

    > Processor for InferenceWorkQueue<I>
{
    type Input = InferenceInput;
    type Output = InferenceOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;
            // TODO: make this adjusted with size and coefficient configured by Parameters
            let inference_timeout = Duration::from_secs(60);
            let inference_output = self
                .inference_engine
                .call(input.clone(), inference_timeout)
                .await
                .map_err(ShardError::InferenceError)?;
            Ok(inference_output)
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
