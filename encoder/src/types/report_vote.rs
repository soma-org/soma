use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use types::report::Report;
use types::shard_crypto::{keys::EncoderPublicKey, signed::Signed};
use types::{
    error::{SharedError, SharedResult},
    shard::Shard,
    shard::ShardAuthToken,
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ReportVoteAPI)]
pub enum ReportVote {
    V1(ReportVoteV1),
}

/// `ReportVoteAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub trait ReportVoteAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn signed_report(&self) -> Signed<Report, min_sig::BLS12381Signature>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReportVoteV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    signed_report: Signed<Report, min_sig::BLS12381Signature>,
}

impl ReportVoteV1 {
    pub const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        signed_report: Signed<Report, min_sig::BLS12381Signature>,
    ) -> Self {
        Self {
            auth_token,
            author,
            signed_report,
        }
    }
}

impl ReportVoteAPI for ReportVoteV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn signed_report(&self) -> Signed<Report, min_sig::BLS12381Signature> {
        self.signed_report.clone()
    }
}

pub fn verify_report_vote(
    report_vote: &ReportVote,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != report_vote.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }

    Ok(())
}
