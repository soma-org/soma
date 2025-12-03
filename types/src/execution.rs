use std::collections::HashSet;

use once_cell::sync::Lazy;

use crate::{
    digests::TransactionDigest,
    effects::CongestedObjects,
    error::{ExecutionErrorKind, SomaError},
    object::Version,
    transaction::CheckedInputObjects,
};

/// Captures the output of executing a transaction in the execution driver.
pub enum ExecutionOutput<T> {
    /// The expected typical path - transaction executed successfully.
    Success(T),
    /// Validator has halted at epoch end or epoch mismatch. This is a valid state that should
    /// be handled gracefully.
    EpochEnded,
    /// Execution failed with an error. This should never happen - we use fatal! when encountered.
    Fatal(SomaError),
    /// Execution should be retried later due to unsatisfied constraints such as insufficient object
    /// balance withdrawals that require waiting for the balance to reach a deterministic amount.
    /// When this happens, the transaction is auto-rescheduled from AuthorityState.
    RetryLater,
}

impl<T> ExecutionOutput<T> {
    /// Unwraps the ExecutionOutput, panicking if it's not Success.
    /// This is primarily for test code.
    pub fn unwrap(self) -> T {
        match self {
            ExecutionOutput::Success(value) => value,
            ExecutionOutput::EpochEnded => {
                panic!("called `ExecutionOutput::unwrap()` on `EpochEnded`")
            }
            ExecutionOutput::Fatal(e) => {
                panic!("called `ExecutionOutput::unwrap()` on `Fatal`: {e}")
            }
            ExecutionOutput::RetryLater => {
                panic!("called `ExecutionOutput::unwrap()` on `RetryLater`")
            }
        }
    }

    /// Expect the execution output to be an error (i.e. not Success).
    pub fn unwrap_err<S>(self) -> ExecutionOutput<S> {
        match self {
            Self::Success(_) => {
                panic!("called `ExecutionOutput::unwrap_err()` on `Success`")
            }
            Self::EpochEnded => ExecutionOutput::EpochEnded,
            Self::Fatal(e) => ExecutionOutput::Fatal(e),
            Self::RetryLater => ExecutionOutput::RetryLater,
        }
    }
}

pub type ExecutionOrEarlyError = Result<(), ExecutionErrorKind>;

/// Determine if a transaction is predetermined to fail execution.
/// If so, return the error kind, otherwise return `None`.
/// When we pass this to the execution engine, we will not execute the transaction
/// if it is predetermined to fail execution.
pub fn get_early_execution_error(
    transaction_digest: &TransactionDigest,
    input_objects: &CheckedInputObjects,
    config_certificate_deny_set: &HashSet<TransactionDigest>,
) -> Option<ExecutionErrorKind> {
    if is_certificate_denied(transaction_digest, config_certificate_deny_set) {
        return Some(ExecutionErrorKind::CertificateDenied);
    }

    if input_objects.inner().contains_deleted_objects() {
        return Some(ExecutionErrorKind::InputObjectDeleted);
    }

    let cancelled_objects = input_objects.inner().get_cancelled_objects();
    if let Some((cancelled_objects, reason)) = cancelled_objects {
        match reason {
            Version::CONGESTED => {
                return Some(ExecutionErrorKind::ExecutionCancelledDueToSharedObjectCongestion);
            }

            _ => panic!("invalid cancellation reason Version: {:?}", reason),
        }
    }

    None
}

/// If a transaction digest shows up in this list, when executing such transaction,
/// we will always return `ExecutionError::CertificateDenied` without executing it (but still do
/// gas smashing). Because this list is not gated by protocol version, there are a few important
/// criteria for adding a digest to this list:
/// 1. The certificate must be causing all validators to either panic or hang forever deterministically.
/// 2. If we ever ship a fix to make it no longer panic or hang when executing such transaction, we
///    must make sure the transaction is already in this list. Otherwise nodes running the newer
///    version without these transactions in the list will generate forked result.
///
/// Below is a scenario of when we need to use this list:
/// 1. We detect that a specific transaction is causing all validators to either panic or hang forever deterministically.
/// 2. We push a CertificateDenyConfig to deny such transaction to all validators asap.
/// 3. To make sure that all fullnodes are able to sync to the latest version, we need to add the
///    transaction digest to this list as well asap, and ship this binary to all fullnodes, so that
///    they can sync past this transaction.
/// 4. We then can start fixing the issue, and ship the fix to all nodes.
/// 5. Unfortunately, we can't remove the transaction digest from this list, because if we do so,
///    any future node that sync from genesis will fork on this transaction. We may be able to
///    remove it once we have stable snapshots and the binary has a minimum supported protocol
///    version past the epoch.
fn get_denied_certificates() -> &'static HashSet<TransactionDigest> {
    static DENIED_CERTIFICATES: Lazy<HashSet<TransactionDigest>> = Lazy::new(|| HashSet::from([]));
    Lazy::force(&DENIED_CERTIFICATES)
}

fn is_certificate_denied(
    transaction_digest: &TransactionDigest,
    certificate_deny_set: &HashSet<TransactionDigest>,
) -> bool {
    certificate_deny_set.contains(transaction_digest)
        || get_denied_certificates().contains(transaction_digest)
}
