use async_trait::async_trait;
use object_store::ObjectStore;
use objects::stores::EphemeralStore;
use std::{marker::PhantomData, time::Duration};
use types::{
    error::EvaluationResult,
    evaluation::{
        EvaluationInput, EvaluationInputAPI, EvaluationOutput, EvaluationOutputAPI,
        EvaluationOutputV1,
    },
};

use crate::evaluation::evaluators::EvaluatorAPI;

pub struct MockEvaluator<ES: ObjectStore, E: EphemeralStore<ES>> {
    output: EvaluationOutput,
    ephemeral_store: E,
    e_marker: PhantomData<ES>,
}

impl<ES: ObjectStore, E: EphemeralStore<ES>> MockEvaluator<ES, E> {
    pub fn new(output: EvaluationOutput, ephemeral_store: E) -> Self {
        Self {
            output,
            ephemeral_store,
            e_marker: PhantomData,
        }
    }
}

#[async_trait]
impl<ES: ObjectStore, E: EphemeralStore<ES>> EvaluatorAPI for MockEvaluator<ES, E> {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput> {
        let EvaluationOutput::V1(v1) = self.output.clone();

        if input.target_embedding().is_none() {
            Ok(EvaluationOutput::V1(EvaluationOutputV1::new(
                v1.evaluation_scores().clone(),
                v1.summary_embedding().clone(),
                v1.sampled_embedding().clone(),
                None,
            )))
        } else {
            Ok(self.output.clone())
        }
    }
}
