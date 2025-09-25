use crate::proto::google::rpc::{BadRequest, ErrorInfo, RetryInfo};
use std::fmt;
use tonic::Code;

pub type Result<T, E = RpcError> = std::result::Result<T, E>;

/// An error encountered while serving an RPC request.
///
/// General error type used by top-level RPC service methods. The main purpose of this error type
/// is to provide a convenient type for converting between internal errors and a response that
/// needs to be sent to a calling client.
#[derive(Debug)]
pub struct RpcError {
    code: Code,
    message: Option<String>,
    details: Option<Box<ErrorDetails>>,
}

impl RpcError {
    pub fn new<T: Into<String>>(code: Code, message: T) -> Self {
        Self {
            code,
            message: Some(message.into()),
            details: None,
        }
    }

    pub fn not_found() -> Self {
        Self {
            code: Code::NotFound,
            message: None,
            details: None,
        }
    }

    pub fn into_status_proto(self) -> crate::proto::google::rpc::Status {
        crate::proto::google::rpc::Status {
            code: self.code.into(),
            message: self.message.unwrap_or_default(),
            details: self
                .details
                .map(ErrorDetails::into_status_details)
                .unwrap_or_default(),
        }
    }
}

impl From<RpcError> for tonic::Status {
    fn from(value: RpcError) -> Self {
        use prost::Message;

        let code = value.code;
        let status = value.into_status_proto();
        let details = status.encode_to_vec().into();
        let message = status.message;

        tonic::Status::with_details(code, message, details)
    }
}

impl From<types::storage::storage_error::Error> for RpcError {
    fn from(value: types::storage::storage_error::Error) -> Self {
        Self {
            code: Code::Internal,
            message: Some(value.to_string()),
            details: None,
        }
    }
}

impl From<anyhow::Error> for RpcError {
    fn from(value: anyhow::Error) -> Self {
        Self {
            code: Code::Internal,
            message: Some(value.to_string()),
            details: None,
        }
    }
}

impl From<crate::utils::types_conversions::SdkTypeConversionError> for RpcError {
    fn from(value: crate::utils::types_conversions::SdkTypeConversionError) -> Self {
        Self {
            code: Code::Internal,
            message: Some(value.to_string()),
            details: None,
        }
    }
}

impl From<bcs::Error> for RpcError {
    fn from(value: bcs::Error) -> Self {
        Self {
            code: Code::Internal,
            message: Some(value.to_string()),
            details: None,
        }
    }
}

impl From<types::quorum_driver::QuorumDriverError> for RpcError {
    fn from(error: types::quorum_driver::QuorumDriverError) -> Self {
        use types::quorum_driver::QuorumDriverError::*;

        match error {
            InvalidUserSignature(err) => {
                // Since you don't have UserInputError, just use the error directly
                let message = format!("Invalid user signature: {}", err);
                RpcError::new(Code::InvalidArgument, message)
            }

            QuorumDriverInternalError(err) => RpcError::new(Code::Internal, err.to_string()),

            ObjectsDoubleUsed {
                conflicting_txes,
                retried_tx,
                retried_tx_success,
            } => {
                // Transform the conflicting_txes similar to reference, but include your additional fields
                let conflicts: Vec<String> = conflicting_txes
                    .into_iter()
                    .map(|(digest, (validators, stake))| {
                        format!(
                            "{}: {} validators (stake: {})",
                            digest,
                            validators.len(),
                            stake
                        )
                    })
                    .collect();

                let mut message = format!(
                    "Failed to sign transaction by a quorum of validators because of locked objects. \
                     Conflicting Transactions: [{}]",
                    conflicts.join(", ")
                );

                // Add retry information if available
                if let Some(tx) = retried_tx {
                    message.push_str(&format!(". Retried transaction: {}", tx));
                    if let Some(success) = retried_tx_success {
                        message.push_str(&format!(" (success: {})", success));
                    }
                }

                RpcError::new(Code::FailedPrecondition, message)
            }

            TimeoutBeforeFinality => RpcError::new(
                Code::Unavailable,
                "Transaction timed out before finality could be reached",
            ),

            FailedWithTransientErrorAfterMaximumAttempts { total_attempts } => RpcError::new(
                Code::Unavailable,
                format!(
                    "Failed with transient error after {} attempts",
                    total_attempts
                ),
            ),

            NonRecoverableTransactionError { errors } => {
                // Since you don't have the same error handling as reference,
                // just format the errors directly
                let error_msg = format!(
                    "Transaction execution failed due to issues with transaction inputs: {:?}",
                    errors
                );
                RpcError::new(Code::InvalidArgument, error_msg)
            }

            SystemOverload {
                overloaded_stake, ..
            } => RpcError::new(
                Code::Unavailable,
                format!(
                    "System is overloaded: {} validators by stake are overloaded",
                    overloaded_stake
                ),
            ),

            SystemOverloadRetryAfter {
                retry_after_secs, ..
            } => {
                // TODO: Add Retry-After header when your RpcError supports it
                RpcError::new(
                    Code::Unavailable,
                    format!(
                        "System is overloaded, retry after {} seconds",
                        retry_after_secs
                    ),
                )
            }

            TxAlreadyFinalizedWithDifferentUserSignatures => RpcError::new(
                Code::Aborted,
                "The transaction is already finalized but with different user signatures",
            ),
        }
    }
}

impl From<crate::proto::google::rpc::bad_request::FieldViolation> for RpcError {
    fn from(value: crate::proto::google::rpc::bad_request::FieldViolation) -> Self {
        BadRequest::from(value).into()
    }
}

impl From<BadRequest> for RpcError {
    fn from(value: BadRequest) -> Self {
        let message = value
            .field_violations
            .first()
            .map(|violation| violation.description.clone());
        let details = ErrorDetails::new().with_bad_request(value);

        RpcError {
            code: Code::InvalidArgument,
            message,
            details: Some(Box::new(details)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ErrorDetails {
    error_info: Option<ErrorInfo>,
    bad_request: Option<BadRequest>,
    retry_info: Option<RetryInfo>,
}

impl ErrorDetails {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn error_info(&self) -> Option<&ErrorInfo> {
        self.error_info.as_ref()
    }

    pub fn bad_request(&self) -> Option<&BadRequest> {
        self.bad_request.as_ref()
    }

    pub fn retry_info(&self) -> Option<&RetryInfo> {
        self.retry_info.as_ref()
    }

    pub fn details(&self) -> &[prost_types::Any] {
        &[]
    }

    pub fn with_bad_request(mut self, bad_request: BadRequest) -> Self {
        self.bad_request = Some(bad_request);
        self
    }

    #[allow(clippy::boxed_local)]
    fn into_status_details(self: Box<Self>) -> Vec<prost_types::Any> {
        let mut details = Vec::new();

        if let Some(error_info) = &self.error_info {
            details.push(
                prost_types::Any::from_msg(error_info).expect("Message encoding cannot fail"),
            );
        }

        if let Some(bad_request) = &self.bad_request {
            details.push(
                prost_types::Any::from_msg(bad_request).expect("Message encoding cannot fail"),
            );
        }

        if let Some(retry_info) = &self.retry_info {
            details.push(
                prost_types::Any::from_msg(retry_info).expect("Message encoding cannot fail"),
            );
        }
        details
    }
}

#[derive(Debug)]
pub struct ObjectNotFoundError {
    object_id: crate::types::Address,
    version: Option<crate::types::Version>,
}

impl ObjectNotFoundError {
    pub fn new(object_id: crate::types::Address) -> Self {
        Self {
            object_id,
            version: None,
        }
    }

    pub fn new_with_version(
        object_id: crate::types::Address,
        version: crate::types::Version,
    ) -> Self {
        Self {
            object_id,
            version: Some(version),
        }
    }
}

impl std::fmt::Display for ObjectNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Object {}", self.object_id)?;

        if let Some(version) = self.version {
            write!(f, " with version {version}")?;
        }

        write!(f, " not found")
    }
}

impl std::error::Error for ObjectNotFoundError {}

impl From<ObjectNotFoundError> for RpcError {
    fn from(value: ObjectNotFoundError) -> Self {
        Self::new(tonic::Code::NotFound, value.to_string())
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Start with the gRPC status code
        write!(f, "RpcError[{}]", self.code)?;

        // Add the message if present
        if let Some(ref message) = self.message {
            write!(f, ": {}", message)?;
        }

        // Add details if present
        if let Some(ref details) = self.details {
            write!(f, " | Details: {}", details)?;
        }

        Ok(())
    }
}

impl fmt::Display for ErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut details = Vec::new();

        // Add error_info if present
        if let Some(ref error_info) = self.error_info {
            details.push(format!(
                "ErrorInfo(domain: {}, reason: {}, metadata: {:?})",
                error_info.domain, error_info.reason, error_info.metadata
            ));
        }

        // Add bad_request if present
        if let Some(ref bad_request) = self.bad_request {
            let violations: Vec<String> = bad_request
                .field_violations
                .iter()
                .map(|v| format!("{}: {}", v.field, v.description))
                .collect();

            if !violations.is_empty() {
                details.push(format!("BadRequest[{}]", violations.join(", ")));
            }
        }

        // Add retry_info if present
        if let Some(ref retry_info) = self.retry_info {
            let retry_after = retry_info
                .retry_delay
                .as_ref()
                .map(|d| format!("{}s", d.seconds))
                .unwrap_or_else(|| "unspecified".to_string());

            details.push(format!("RetryInfo(retry_after: {})", retry_after));
        }

        // Join all details or show "none" if empty
        if details.is_empty() {
            write!(f, "none")
        } else {
            write!(f, "{}", details.join(" | "))
        }
    }
}

// Also implement std::error::Error for RpcError if not already present
impl std::error::Error for RpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
