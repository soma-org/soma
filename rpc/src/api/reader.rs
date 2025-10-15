use std::collections::HashMap;
use std::sync::Arc;

use crate::types::{Address, Object, Version};
use crate::types::{EpochId, SignedTransaction, ValidatorCommittee};
use types::balance_change::BalanceChange;
use types::object::{ObjectID, ObjectType};
use types::storage::read_store::{RpcStateReader, TransactionInfo};
use types::storage::storage_error::{Error as StorageError, Result};

#[derive(Clone)]
pub struct StateReader {
    inner: Arc<dyn RpcStateReader>,
}

impl StateReader {
    pub fn new(inner: Arc<dyn RpcStateReader>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Arc<dyn RpcStateReader> {
        &self.inner
    }

    #[tracing::instrument(skip(self))]
    pub fn get_object(&self, object_id: Address) -> crate::api::error::Result<Option<Object>> {
        self.inner
            .get_object(&object_id.into())?
            .map(TryInto::try_into)
            .transpose()
            .map_err(Into::into)
    }

    #[tracing::instrument(skip(self))]
    pub fn get_object_with_version(
        &self,
        object_id: Address,
        version: Version,
    ) -> crate::api::error::Result<Option<Object>> {
        self.inner
            .get_object_by_key(&object_id.into(), types::object::Version::from_u64(version))?
            .map(TryInto::try_into)
            .transpose()
            .map_err(Into::into)
    }

    #[tracing::instrument(skip(self))]
    pub fn get_committee(&self, epoch: EpochId) -> Option<ValidatorCommittee> {
        self.inner
            .get_committee(epoch)
            .ok()
            .flatten()
            .map(|committee| (*committee).clone().into())
    }

    #[tracing::instrument(skip(self))]
    pub fn get_system_state(&self) -> Result<types::system_state::SystemState> {
        types::system_state::get_system_state(&self.inner().as_ref())
            .map_err(StorageError::custom)
            .map_err(StorageError::custom)
    }

    #[tracing::instrument(skip(self))]
    pub fn get_transaction(
        &self,
        digest: crate::types::Digest,
    ) -> crate::api::error::Result<(
        crate::types::SignedTransaction,
        crate::types::TransactionEffects,
    )> {
        use types::effects::TransactionEffectsAPI;

        let transaction_digest = digest.into();

        let transaction = (*self
            .inner()
            .get_transaction(&transaction_digest)?
            .ok_or(TransactionNotFoundError(digest))?)
        .clone()
        .into_inner();
        let effects = self
            .inner()
            .get_transaction_effects(&transaction_digest)?
            .ok_or(TransactionNotFoundError(digest))?;

        Ok((transaction.try_into()?, effects.try_into()?))
    }

    #[tracing::instrument(skip(self))]
    pub fn get_transaction_info(
        &self,
        digest: &types::digests::TransactionDigest,
    ) -> Option<TransactionInfo> {
        self.inner()
            .indexes()?
            .get_transaction_info(digest)
            .ok()
            .flatten()
    }

    #[tracing::instrument(skip(self))]
    pub fn get_transaction_read(
        &self,
        digest: crate::types::Digest,
    ) -> crate::api::error::Result<TransactionRead> {
        let (
            SignedTransaction {
                transaction,
                signatures,
            },
            effects,
        ) = self.get_transaction(digest)?;

        let (commit, balance_changes, object_types) =
            if let Some(info) = self.get_transaction_info(&(digest.into())) {
                (
                    Some(info.commit),
                    Some(info.balance_changes),
                    Some(info.object_types),
                )
            } else {
                (None, None, None)
            };
        let timestamp_ms = if let Some(commit) = commit {
            self.inner()
                .get_commit_by_index(commit.try_into().unwrap()) // TODO: handle this cleaner
                .map(|checkpoint| checkpoint.timestamp_ms)
        } else {
            None
        };

        Ok(TransactionRead {
            digest: transaction.digest(),
            transaction,
            signatures,
            effects,
            commit,
            timestamp_ms,
            balance_changes,
            object_types,
        })
    }
}

#[derive(Debug)]
pub struct TransactionRead {
    pub digest: crate::types::Digest,
    pub transaction: crate::types::Transaction,
    pub signatures: Vec<crate::types::UserSignature>,
    pub effects: crate::types::TransactionEffects,
    pub commit: Option<u64>,
    pub timestamp_ms: Option<u64>,
    pub balance_changes: Option<Vec<BalanceChange>>,
    pub object_types: Option<HashMap<ObjectID, ObjectType>>,
}

#[derive(Debug)]
pub struct TransactionNotFoundError(pub crate::types::Digest);

impl std::fmt::Display for TransactionNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Transaction {} not found", self.0)
    }
}

impl std::error::Error for TransactionNotFoundError {}

impl From<TransactionNotFoundError> for crate::api::error::RpcError {
    fn from(value: TransactionNotFoundError) -> Self {
        Self::new(tonic::Code::NotFound, value.to_string())
    }
}
