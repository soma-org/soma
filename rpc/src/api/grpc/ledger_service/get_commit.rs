use crate::api::RpcService;
use crate::api::error::CommitNotFoundError;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::BalanceChange;
use crate::proto::soma::Commit;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::ExecutedTransaction;
use crate::proto::soma::GetCommitRequest;
use crate::proto::soma::GetCommitResponse;
use crate::proto::soma::get_commit_request::CommitId;
use crate::proto::timestamp_ms_to_proto;
use crate::types::Digest;
// use crate::error::CheckpointNotFoundError;
use crate::utils::field::FieldMaskTree;
use crate::utils::field::FieldMaskUtil;
use crate::utils::merge::Merge;
use prost_types::FieldMask;
use types::consensus::output::ConsensusOutputAPI;

pub const READ_MASK_DEFAULT: &str = "sequence_number,digest";

#[tracing::instrument(skip(service))]
pub fn get_commit(
    service: &RpcService,
    request: GetCommitRequest,
) -> Result<GetCommitResponse, RpcError> {
    let read_mask = {
        let read_mask = request
            .read_mask
            .unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Commit>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let committed_sub_dag = match request.commit_id {
        Some(CommitId::Index(s)) => service
            .reader
            .inner()
            .get_commit_by_index(s)
            .ok_or(CommitNotFoundError::index(s))?,
        Some(CommitId::Digest(digest)) => {
            let digest = digest.parse::<Digest>().map_err(|e| {
                FieldViolation::new("digest")
                    .with_description(format!("invalid digest: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })?;

            service
                .reader
                .inner()
                .get_commit_by_digest(&digest.into())
                .ok_or(CommitNotFoundError::digest(digest))?
        }
        None => service.reader.inner().get_latest_commit()?,
        _ => service.reader.inner().get_latest_commit()?,
    };

    let mut commit = Commit::default();
    commit.index = Some(committed_sub_dag.commit_ref.index);
    commit.digest = Some(committed_sub_dag.commit_ref.digest.to_string());
    commit.timestamp_ms = Some(committed_sub_dag.commit_timestamp_ms());

    if read_mask.contains(Commit::TRANSACTIONS_FIELD.name) {
        let checkpoint_data = service
            .reader
            .inner()
            .get_checkpoint_data(&committed_sub_dag)?;

        if let Some(submask) = read_mask.subtree(Commit::TRANSACTIONS_FIELD.name) {
            commit.transactions = checkpoint_data
                .transactions
                .into_iter()
                .map(|t| {
                    let balance_changes = submask
                        .contains(ExecutedTransaction::BALANCE_CHANGES_FIELD)
                        .then(|| {
                            service
                                .reader
                                .get_transaction_info(&t.transaction.digest())
                                .map(|info| {
                                    info.balance_changes
                                        .into_iter()
                                        .map(BalanceChange::from)
                                        .collect::<Vec<_>>()
                                })
                        })
                        .flatten()
                        .unwrap_or_default();
                    let mut transaction = ExecutedTransaction::merge_from(&t, &submask);
                    transaction.commit = submask
                        .contains(ExecutedTransaction::COMMIT_FIELD)
                        .then_some(committed_sub_dag.commit_ref.index.into());
                    transaction.timestamp = submask
                        .contains(ExecutedTransaction::TIMESTAMP_FIELD)
                        .then(|| timestamp_ms_to_proto(committed_sub_dag.timestamp_ms));
                    transaction.balance_changes = balance_changes;

                    transaction
                })
                .collect();
        }
    }

    Ok(GetCommitResponse::new(commit))
}
