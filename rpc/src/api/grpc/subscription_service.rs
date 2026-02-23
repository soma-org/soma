// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use crate::api::RpcService;
use crate::proto::soma::Checkpoint;
use crate::proto::soma::SubscribeCheckpointsRequest;
use crate::proto::soma::SubscribeCheckpointsResponse;
use crate::proto::soma::subscription_service_server::SubscriptionService;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

#[tonic::async_trait]
impl SubscriptionService for RpcService {
    /// Server streaming response type for the SubscribeCheckpoints method.
    type SubscribeCheckpointsStream = Pin<
        Box<
            dyn tokio_stream::Stream<Item = Result<SubscribeCheckpointsResponse, tonic::Status>>
                + Send,
        >,
    >;

    async fn subscribe_checkpoints(
        &self,
        request: tonic::Request<SubscribeCheckpointsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeCheckpointsStream>, tonic::Status> {
        let subscription_service_handle = self
            .subscription_service_handle
            .as_ref()
            .ok_or_else(|| tonic::Status::unimplemented("subscription service not enabled"))?;
        let read_mask = request.into_inner().read_mask.unwrap_or_default();
        let read_mask = FieldMaskTree::from(read_mask);

        let Some(mut receiver) = subscription_service_handle.register_subscription().await else {
            return Err(tonic::Status::unavailable("too many existing subscriptions"));
        };

        let store = self.reader.clone();
        let response = Box::pin(async_stream::stream! {
            while let Some(checkpoint) = receiver.recv().await {
                let cursor = checkpoint.summary.sequence_number;

                let mut checkpoint_message = Checkpoint::merge_from(
                    checkpoint.as_ref(),
                    &read_mask
                );

                if read_mask.contains("transactions.balance_changes") {
                    for (txn, txn_digest) in checkpoint_message.transactions_mut().iter_mut().zip(
                        checkpoint
                            .transactions
                            .iter()
                            .map(|t| t.transaction.digest()),
                    ) {
                        if let Some(info) = store.get_transaction_info(&txn_digest)
                        {
                            *txn.balance_changes_mut() = info.balance_changes
                                .into_iter()
                                .map(crate::proto::soma::BalanceChange::from)
                                .collect::<Vec<_>>();
                        }
                    }
                }

                let response = SubscribeCheckpointsResponse {
                    cursor: Some(cursor),
                    checkpoint: Some(checkpoint_message),
                    ..Default::default()
                };

                yield Ok(response);
            }
        });

        Ok(tonic::Response::new(response))
    }
}
