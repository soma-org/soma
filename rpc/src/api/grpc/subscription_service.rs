use std::pin::Pin;

use crate::api::RpcService;
use crate::proto::soma::Commit;
use crate::proto::soma::SubscribeCommitsRequest;
use crate::proto::soma::SubscribeCommitsResponse;
use crate::proto::soma::subscription_service_server::SubscriptionService;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

#[tonic::async_trait]
impl SubscriptionService for RpcService {
    /// Server streaming response type for the SubscribeCommits method.
    type SubscribeCommitsStream = Pin<
        Box<
            dyn tokio_stream::Stream<Item = Result<SubscribeCommitsResponse, tonic::Status>> + Send,
        >,
    >;

    async fn subscribe_commits(
        &self,
        request: tonic::Request<SubscribeCommitsRequest>,
    ) -> Result<tonic::Response<Self::SubscribeCommitsStream>, tonic::Status> {
        let subscription_service_handle = self
            .subscription_service_handle
            .as_ref()
            .ok_or_else(|| tonic::Status::unimplemented("subscription service not enabled"))?;
        let read_mask = request.into_inner().read_mask.unwrap_or_default();
        let read_mask = FieldMaskTree::from(read_mask);

        let Some(mut receiver) = subscription_service_handle.register_subscription().await else {
            return Err(tonic::Status::unavailable(
                "too many existing subscriptions",
            ));
        };

        let store = self.reader.clone();
        let response = Box::pin(async_stream::stream! {
            while let Some(checkpoint) = receiver.recv().await {
                let cursor = checkpoint.commit_index;

                let mut checkpoint_message = Commit::merge_from(
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


                let mut response = SubscribeCommitsResponse::default();
                response.cursor = Some(cursor.into());
                response.commit = Some(checkpoint_message);

                yield Ok(response);
            }
        });

        Ok(tonic::Response::new(response))
    }
}
