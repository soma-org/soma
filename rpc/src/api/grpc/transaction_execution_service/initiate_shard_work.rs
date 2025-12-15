use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::{ErrorReason, Shard};
use crate::{
    api::RpcService,
    proto::soma::{InitiateShardWorkRequest, InitiateShardWorkResponse},
};
use types::digests::TransactionDigest;
use types::error::SomaError;

pub async fn initiate_shard_work(
    service: &RpcService,
    request: InitiateShardWorkRequest,
) -> Result<InitiateShardWorkResponse> {
    let executor = service
        .executor
        .as_ref()
        .ok_or_else(|| RpcError::new(tonic::Code::Unimplemented, "no transaction executor"))?;

    // Parse tx_digest from the request
    let tx_digest: TransactionDigest = request
        .tx_digest
        .as_ref()
        .ok_or_else(|| FieldViolation::new("tx_digest").with_reason(ErrorReason::FieldMissing))?
        .parse()
        .map_err(|e| {
            FieldViolation::new("tx_digest")
                .with_description(format!("invalid tx_digest: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    // Parse checkpoint_seq from the request
    let checkpoint_seq = request.checkpoint_seq.ok_or_else(|| {
        FieldViolation::new("checkpoint_seq").with_reason(ErrorReason::FieldMissing)
    })?;

    // Build the domain request
    let domain_request = types::quorum_driver::InitiateShardWorkRequest {
        tx_digest,
        checkpoint_seq,
    };

    let response = executor
        .initiate_shard_work(domain_request)
        .await
        .map_err(|e| match e {
            SomaError::TransactionNotFound { .. } => {
                RpcError::new(tonic::Code::NotFound, e.to_string())
            }
            SomaError::NotEmbedDataTransaction => {
                RpcError::new(tonic::Code::InvalidArgument, e.to_string())
            }
            SomaError::TransactionNotFinalized => {
                RpcError::new(tonic::Code::FailedPrecondition, e.to_string())
            }
            SomaError::EncoderServiceUnavailable => {
                RpcError::new(tonic::Code::Unavailable, e.to_string())
            }
            SomaError::VerifiedCheckpointNotFound(_) => {
                RpcError::new(tonic::Code::NotFound, e.to_string())
            }
            SomaError::CheckpointContentsNotFound(_) => {
                RpcError::new(tonic::Code::NotFound, e.to_string())
            }
            SomaError::InvalidRequest(_) => {
                RpcError::new(tonic::Code::InvalidArgument, e.to_string())
            }
            SomaError::FailedVDF(_) => RpcError::new(tonic::Code::Internal, e.to_string()),
            SomaError::ShardSamplingError(_) => RpcError::new(tonic::Code::Internal, e.to_string()),
            _ => RpcError::new(tonic::Code::Internal, e.to_string()),
        })?;

    let shard = convert_shard_to_proto(response.shard).map_err(|e| {
        RpcError::new(
            tonic::Code::Internal,
            format!("Failed to convert shard: {e}"),
        )
    })?;

    Ok(InitiateShardWorkResponse { shard: Some(shard) })
}

fn convert_shard_to_proto(shard: types::shard::Shard) -> std::result::Result<Shard, String> {
    let encoders = shard
        .encoders()
        .into_iter()
        .map(|encoder| encoder.to_hex_string())
        .collect::<Vec<_>>();

    // Store the owned value first, then convert to Vec<u8>
    let seed_bytes: Vec<u8> = shard.seed.into_inner().to_vec();

    Ok(Shard {
        quorum_threshold: Some(shard.quorum_threshold()),
        encoders,
        seed: Some(seed_bytes.into()),
        epoch: Some(shard.epoch()),
    })
}
