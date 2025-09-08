use crate::submission::Submission;
use crate::{shard_crypto::digest::Digest, shard_crypto::keys::EncoderPublicKey};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait ReportAPI {
    fn winning_submission(&self) -> &Submission;
    fn accepted_submissions(&self) -> &Vec<(EncoderPublicKey, Digest<Submission>)>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ReportAPI)]
pub enum Report {
    V1(ReportV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ReportV1 {
    winning_submission: Submission,
    accepted_submissions: Vec<(EncoderPublicKey, Digest<Submission>)>,
}

impl ReportV1 {
    pub fn new(
        winning_submission: Submission,
        mut accepted_submissions: Vec<(EncoderPublicKey, Digest<Submission>)>,
    ) -> Self {
        accepted_submissions.sort();
        Self {
            winning_submission,
            accepted_submissions,
        }
    }
}

impl ReportAPI for ReportV1 {
    fn winning_submission(&self) -> &Submission {
        &self.winning_submission
    }
    fn accepted_submissions(&self) -> &Vec<(EncoderPublicKey, Digest<Submission>)> {
        &self.accepted_submissions
    }
}
