use std::{sync::Arc, time::Duration};

use async_trait::async_trait;

use types::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{EvaluationInput, EvaluationOutput},
};

use crate::evaluation::evaluators::EvaluatorAPI;

pub struct EvaluationWorkQueue<
    C: EvaluatorAPI,
> {
    evaluator_client: Arc<C>,
}

impl<
C: EvaluatorAPI
    > EvaluationWorkQueue< C>
{
    pub fn new(
        evaluator_client: Arc<C>,
    ) -> Self {
        Self {
            evaluator_client,
        }
    }
}

#[async_trait]
impl<
        C: EvaluatorAPI,
    > Processor for EvaluationWorkQueue< C>
{
    type Input = EvaluationInput;
    type Output = EvaluationOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;
            // TODO: make this adjusted with size and coefficient configured by Parameters
            let evaluation_timeout = Duration::from_secs(60);
            let evaluation_output = self
                .evaluator_client
                .call(input.clone(), evaluation_timeout)
                .await
                .map_err(ShardError::EvaluationError)?;

            Ok(evaluation_output)
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
