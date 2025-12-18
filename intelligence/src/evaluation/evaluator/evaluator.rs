use async_trait::async_trait;
use object_store::ObjectStore;
use objects::stores::EphemeralStore;
use std::{marker::PhantomData, time::Duration};
use types::{
    error::EvaluationResult,
    evaluation::{EvaluationInput, EvaluationInputAPI, EvaluationOutput},
};

use crate::evaluation::evaluator::EvaluatorClient;

pub struct Evaluator<ES: ObjectStore, E: EphemeralStore<ES>> {
    ephemeral_store: E,
    e_marker: PhantomData<ES>,
}

impl<ES: ObjectStore, E: EphemeralStore<ES>> Evaluator<ES, E> {
    pub fn new(ephemeral_store: E) -> Self {
        Self {
            ephemeral_store,
            e_marker: PhantomData,
        }
    }
}

#[async_trait]
impl<ES: ObjectStore, E: EphemeralStore<ES>> EvaluatorClient for Evaluator<ES, E> {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput> {
        let buffer = self
            .ephemeral_store
            .buffer_object(input.embedding_object_path().clone())
            .await
            .unwrap();

        let representations = safetensors::SafeTensors::deserialize(buffer.as_ref()).unwrap();

        unimplemented!();
    }
}
