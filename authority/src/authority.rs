// Portions of this file are derived from Sui (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/crates/sui-core/src/
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write as _;
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{pin::Pin, sync::Arc};

use arc_swap::{ArcSwap, Guard};
use fastcrypto::encoding::{Base58, Encoding as _};
use fastcrypto::hash::MultisetHash;
use itertools::Itertools as _;
use parking_lot::Mutex;
use protocol_config::{ProtocolConfig, ProtocolVersion};
use serde::{Deserialize, Serialize};
use tap::TapFallible as _;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::{mpsc::unbounded_channel, oneshot};
use tokio::time::timeout;
use tracing::{debug, error, info, instrument, trace, warn};
use types::SYSTEM_STATE_OBJECT_ID;
use types::base::{ConciseableName as _, FullObjectID};
use types::checkpoints::{
    CheckpointCommitment, CheckpointContents, CheckpointRequest, CheckpointResponse,
    CheckpointSequenceNumber, CheckpointSummary, CheckpointSummaryResponse, CheckpointTimestamp,
    ECMHLiveObjectSetDigest, VerifiedCheckpoint,
};
use types::config::node_config::{ExpensiveSafetyCheckConfig, StateDebugDumpConfig};
use types::consensus::AuthorityCapabilitiesV1;
use types::digests::{
    ChainIdentifier, CheckpointContentsDigest, CheckpointDigest, ObjectDigest,
    TransactionEffectsDigest,
};
use types::effects::{
    InputSharedObject, SignedTransactionEffects, TransactionEffects, TransactionEffectsAPI,
    VerifiedSignedTransactionEffects,
};
use types::envelope::Message;
use types::error::{ExecutionError, ExecutionResult};
use types::execution::{ExecutionOrEarlyError, ExecutionOutput, get_early_execution_error};
use types::full_checkpoint_content::ObjectSet;
use types::messages_grpc::TransactionInfoRequest;
use types::object::{OBJECT_START_VERSION, Object, ObjectID, ObjectRef, Owner, Version};
use types::storage::InputKey;
use types::storage::object_store::ObjectStore;
use types::supported_protocol_versions::SupportedProtocolVersions;
use types::system_state::get_system_state;
use types::system_state::{SystemState, epoch_start::EpochStartSystemStateTrait};
use types::temporary_store::InnerTemporaryStore;
use types::transaction::{
    CheckedInputObjects, InputObjects, ObjectReadResult, SenderSignedData, TransactionData,
    TransactionKey,
};
use types::transaction_executor::{SimulateTransactionResult, TransactionChecks};
use types::transaction_outputs::{TransactionOutputs, WrittenObjects};
use types::tx_fee::TransactionFee;
use types::{
    base::AuthorityName,
    committee::{Committee, EpochId},
    config::node_config::NodeConfig,
    crypto::{AuthoritySignInfo, AuthoritySignature, Signer},
    digests::TransactionDigest,
    error::{SomaError, SomaResult},
    intent::{Intent, IntentScope},
    messages_grpc::{
        HandleTransactionResponse, ObjectInfoRequest, ObjectInfoResponse, TransactionInfoResponse,
        TransactionStatus,
    },
    transaction::{
        VerifiedCertificate, VerifiedExecutableTransaction, VerifiedSignedTransaction,
        VerifiedTransaction,
    },
};

use crate::authority_per_epoch_store::{CertLockGuard, CertTxGuard};
use crate::authority_per_epoch_store_pruner::AuthorityPerEpochStorePruner;
use crate::authority_store::{AuthorityStore, ObjectLockStatus};
use crate::authority_store_pruner::{
    AuthorityStorePruner, EPOCH_DURATION_MS_FOR_TESTING, PrunerWatermarks,
};
#[cfg(test)]
use crate::authority_store_tables;
use crate::authority_store_tables::AuthorityPrunerTables;
use crate::cache::{
    ExecutionCacheCommit, ExecutionCacheReconfigAPI, ExecutionCacheTraitPointers,
    ExecutionCacheWrite, ObjectCacheRead, StateSyncAPI, TransactionCacheRead,
};
use crate::checkpoints::{CheckpointBuilderError, CheckpointBuilderResult, CheckpointStore};
use crate::execution::execute_transaction;
use crate::execution_driver::execution_process;
use crate::execution_scheduler::{ExecutionScheduler, SchedulingSource};
use crate::global_state_hasher::{GlobalStateHashStore, GlobalStateHasher};
use crate::rpc_index::RpcIndexStore;
use crate::shared_obj_version_manager::{AssignedVersions, Schedulable};
use crate::stake_aggregator::StakeAggregator;
use crate::start_epoch::EpochStartConfigTrait;
use crate::transaction_input_loader::TransactionInputLoader;
use crate::validator_tx_finalizer::ValidatorTxFinalizer;
use crate::{
    authority_client::NetworkAuthorityClient, authority_per_epoch_store::AuthorityPerEpochStore,
    start_epoch::EpochStartConfiguration,
};
use crate::{consensus_quarantine, transaction_checks};
use types::storage::committee_store::CommitteeStore;

// #[cfg(test)]
// #[path = "unit_tests/authority_tests.rs"]
// pub mod authority_tests;

pub const WAIT_FOR_FASTPATH_INPUT_TIMEOUT: Duration = Duration::from_secs(2);

pub const DEV_INSPECT_GAS_COIN_VALUE: u64 = 1_000_000_000_000_000_000;

/// a Trait object for `Signer` that is:
/// - Pin, i.e. confined to one place in memory (we don't want to copy private keys).
/// - Sync, i.e. can be safely shared between threads.
///
/// Typically instantiated with Box::pin(keypair) where keypair is a `KeyPair`
///
pub type StableSyncAuthoritySigner = Pin<Arc<dyn Signer<AuthoritySignature> + Send + Sync>>;

/// Execution env contains the "environment" for the transaction to be executed in, that is,
/// all the information necessary for execution that is not specified by the transaction itself.
#[derive(Debug, Clone)]
pub struct ExecutionEnv {
    /// The assigned version of each shared object for the transaction.
    pub assigned_versions: AssignedVersions,
    /// The expected digest of the effects of the transaction, if executing from checkpoint or
    /// other sources where the effects are known in advance.
    pub expected_effects_digest: Option<TransactionEffectsDigest>,
    /// The source of the scheduling of the transaction.
    pub scheduling_source: SchedulingSource,
    /// Transactions that must finish before this transaction can be executed.
    /// Used to schedule barrier transactions after non-exclusive writes.
    pub barrier_dependencies: Vec<TransactionDigest>,
}

impl Default for ExecutionEnv {
    fn default() -> Self {
        Self {
            assigned_versions: Default::default(),
            expected_effects_digest: None,
            scheduling_source: SchedulingSource::NonFastPath,
            barrier_dependencies: Default::default(),
        }
    }
}

impl ExecutionEnv {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_scheduling_source(mut self, scheduling_source: SchedulingSource) -> Self {
        self.scheduling_source = scheduling_source;
        self
    }

    pub fn with_expected_effects_digest(
        mut self,
        expected_effects_digest: TransactionEffectsDigest,
    ) -> Self {
        self.expected_effects_digest = Some(expected_effects_digest);
        self
    }

    pub fn with_assigned_versions(mut self, assigned_versions: AssignedVersions) -> Self {
        self.assigned_versions = assigned_versions;
        self
    }

    pub fn with_barrier_dependencies(
        mut self,
        barrier_dependencies: BTreeSet<TransactionDigest>,
    ) -> Self {
        self.barrier_dependencies = barrier_dependencies.into_iter().collect();
        self
    }
}

#[derive(Debug)]
pub struct ForkRecoveryState {
    /// Transaction digest to effects digest overrides
    transaction_overrides:
        parking_lot::RwLock<HashMap<TransactionDigest, TransactionEffectsDigest>>,
}

impl Default for ForkRecoveryState {
    fn default() -> Self {
        Self { transaction_overrides: parking_lot::RwLock::new(HashMap::new()) }
    }
}

impl ForkRecoveryState {
    pub fn new(
        config: Option<&types::config::node_config::ForkRecoveryConfig>,
    ) -> Result<Self, SomaError> {
        let Some(config) = config else {
            return Ok(Self::default());
        };

        let mut transaction_overrides = HashMap::new();
        for (tx_digest_str, effects_digest_str) in &config.transaction_overrides {
            let tx_digest = TransactionDigest::from_str(tx_digest_str).map_err(|_| {
                SomaError::Unknown(format!("Invalid transaction digest: {}", tx_digest_str))
            })?;
            let effects_digest =
                TransactionEffectsDigest::from_str(effects_digest_str).map_err(|_| {
                    SomaError::Unknown(format!("Invalid effects digest: {}", effects_digest_str))
                })?;
            transaction_overrides.insert(tx_digest, effects_digest);
        }

        Ok(Self { transaction_overrides: parking_lot::RwLock::new(transaction_overrides) })
    }

    pub fn get_transaction_override(
        &self,
        tx_digest: &TransactionDigest,
    ) -> Option<TransactionEffectsDigest> {
        self.transaction_overrides.read().get(tx_digest).copied()
    }
}

pub struct AuthorityState {
    // Fixed size, static, identity of the authority
    /// The name of this authority.
    pub name: AuthorityName,
    /// The signature key of the authority.
    pub secret: StableSyncAuthoritySigner,

    /// The database
    input_loader: TransactionInputLoader,
    execution_cache_trait_pointers: ExecutionCacheTraitPointers,

    epoch_store: ArcSwap<AuthorityPerEpochStore>,

    /// This lock denotes current 'execution epoch'.
    /// Execution acquires read lock, checks certificate epoch and holds it until all writes are complete.
    /// Reconfiguration acquires write lock, changes the epoch and revert all transactions
    /// from previous epoch that are executed but did not make into checkpoint.
    execution_lock: RwLock<EpochId>,

    pub rpc_index: Option<Arc<RpcIndexStore>>,

    pub checkpoint_store: Arc<CheckpointStore>,

    committee_store: Arc<CommitteeStore>,

    /// Schedules transaction execution.
    pub execution_scheduler: Arc<ExecutionScheduler>,

    /// Shuts down the execution task. Used only in testing.
    #[allow(unused)]
    tx_execution_shutdown: Mutex<Option<oneshot::Sender<()>>>,

    _pruner: AuthorityStorePruner,
    _authority_per_epoch_pruner: AuthorityPerEpochStorePruner,

    pub config: NodeConfig,

    pub validator_tx_finalizer: Option<Arc<ValidatorTxFinalizer<NetworkAuthorityClient>>>,
    // The chain identifier is derived from the digest of the genesis checkpoint.
    chain_identifier: ChainIdentifier,
    //TODO: Traffic controller for core servers (json-rpc, validator service)
    // pub traffic_controller: Option<Arc<TrafficController>>,
    /// Fork recovery state for handling equivocation after forks
    fork_recovery_state: Option<ForkRecoveryState>,
}

impl AuthorityState {
    pub fn is_validator(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        epoch_store.committee().authority_exists(&self.name)
    }

    pub fn is_fullnode(&self, epoch_store: &AuthorityPerEpochStore) -> bool {
        !self.is_validator(epoch_store)
    }

    pub fn committee_store(&self) -> &Arc<CommitteeStore> {
        &self.committee_store
    }

    pub fn clone_committee_store(&self) -> Arc<CommitteeStore> {
        self.committee_store.clone()
    }

    pub fn get_epoch_state_commitments(
        &self,
        epoch: EpochId,
    ) -> SomaResult<Option<Vec<CheckpointCommitment>>> {
        self.checkpoint_store.get_epoch_state_commitments(epoch)
    }

    fn handle_transaction_deny_checks(
        &self,
        transaction: &VerifiedTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<CheckedInputObjects> {
        let tx_digest = transaction.digest();
        let tx_data = transaction.data().transaction_data();

        let input_object_kinds = tx_data.input_objects()?;
        let receiving_objects_refs = tx_data.receiving_objects();

        // Note: the deny checks may do redundant package loads but:
        // - they only load packages when there is an active package deny map
        // - the loads are cached anyway
        transaction_checks::check_transaction_for_signing(
            tx_data,
            transaction.tx_signatures(),
            &input_object_kinds,
            &receiving_objects_refs,
            &self.config.transaction_deny_config,
        )?;

        let (input_objects, receiving_objects) = self.input_loader.read_objects_for_signing(
            Some(tx_digest),
            &input_object_kinds,
            &receiving_objects_refs,
            epoch_store.epoch(),
        )?;

        let checked_input_objects = transaction_checks::check_transaction_input(
            epoch_store.protocol_config(),
            tx_data,
            input_objects,
            &receiving_objects,
        )?;

        Ok(checked_input_objects)
    }

    /// This is a private method and should be kept that way. It doesn't check whether
    /// the provided transaction is a system transaction, and hence can only be called internally.
    #[instrument(level = "trace", skip_all)]
    fn handle_transaction_impl(
        &self,
        transaction: VerifiedTransaction,
        sign: bool,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Option<VerifiedSignedTransaction>> {
        // Ensure that validator cannot reconfigure while we are signing the tx
        let _execution_lock = self.execution_lock_for_signing()?;

        let checked_input_objects =
            self.handle_transaction_deny_checks(&transaction, epoch_store)?;

        let owned_objects = checked_input_objects.inner().filter_owned_objects();

        let tx_digest = *transaction.digest();
        let signed_transaction = if sign {
            Some(VerifiedSignedTransaction::new(
                epoch_store.epoch(),
                transaction,
                self.name,
                &*self.secret,
            ))
        } else {
            None
        };

        // Check and write locks, to signed transaction, into the database
        // The call to self.set_transaction_lock checks the lock is not conflicting,
        // and returns ConflictingTransaction error in case there is a lock on a different
        // existing transaction.
        self.get_cache_writer().acquire_transaction_locks(
            epoch_store,
            &owned_objects,
            tx_digest,
            signed_transaction.clone(),
        )?;

        Ok(signed_transaction)
    }

    /// Initiate a new transaction.
    #[instrument(level = "trace", skip_all)]
    pub async fn handle_transaction(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: VerifiedTransaction,
    ) -> SomaResult<HandleTransactionResponse> {
        let tx_digest = *transaction.digest();
        debug!("handle_transaction");

        // Ensure an idempotent answer.
        if let Some((_, status)) = self.get_transaction_status(&tx_digest, epoch_store)? {
            return Ok(HandleTransactionResponse { status });
        }

        if !self.wait_for_fastpath_dependency_objects(&transaction, epoch_store.epoch()).await? {
            debug!("fastpath input objects are still unavailable after waiting");
            // Proceed with input checks to generate a proper error.
        }

        self.handle_sign_transaction(epoch_store, transaction).await
    }

    /// Signs a transaction. Exposed for testing.
    pub async fn handle_sign_transaction(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: VerifiedTransaction,
    ) -> SomaResult<HandleTransactionResponse> {
        let tx_digest = *transaction.digest();
        let signed = self.handle_transaction_impl(transaction, /* sign */ true, epoch_store);
        match signed {
            Ok(Some(s)) => {
                if self.is_validator(epoch_store) {
                    if let Some(validator_tx_finalizer) = &self.validator_tx_finalizer {
                        let tx = s.clone();
                        let validator_tx_finalizer = validator_tx_finalizer.clone();
                        let cache_reader = self.get_transaction_cache_reader().clone();
                        let epoch_store = epoch_store.clone();
                        tokio::spawn(async move {
                            epoch_store
                                .within_alive_epoch(validator_tx_finalizer.track_signed_tx(
                                    cache_reader,
                                    &epoch_store,
                                    tx,
                                ))
                                .await
                        });
                    }
                }
                Ok(HandleTransactionResponse {
                    status: TransactionStatus::Signed(s.into_inner().into_sig()),
                })
            }
            Ok(None) => panic!("handle_transaction_impl should return a signed transaction"),
            // It happens frequently that while we are checking the validity of the transaction, it
            // has just been executed.
            // In that case, we could still return Ok to avoid showing confusing errors.
            Err(err) => Ok(HandleTransactionResponse {
                status: self.get_transaction_status(&tx_digest, epoch_store)?.ok_or(err)?.1,
            }),
        }
    }

    /// Vote for a transaction, either when validator receives a submit_transaction request,
    /// or sees a transaction from consensus. Performs the same types of checks as
    /// transaction signing, but does not explicitly sign the transaction.
    /// Note that if the transaction has been executed, we still go through the
    /// same checks. If the transaction has only been executed through mysticeti fastpath,
    /// but not yet finalized, the signing will still work since the objects are still available.
    /// But if the transaction has been finalized, the signing will fail.
    /// TODO(mysticeti-fastpath): Assess whether we want to optimize the case when the transaction
    /// has already been finalized executed.
    #[instrument(level = "trace", skip_all, fields(tx_digest = ?transaction.digest()))]
    pub(crate) fn handle_vote_transaction(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: VerifiedTransaction,
    ) -> SomaResult<()> {
        debug!("handle_vote_transaction");

        // The should_accept_user_certs check here is best effort, because
        // between a validator signs a tx and a cert is formed, the validator
        // could close the window.
        if !epoch_store.get_reconfig_state_read_lock_guard().should_accept_user_certs() {
            return Err(SomaError::ValidatorHaltedAtEpochEnd.into());
        }

        // Accept executed transactions, instead of voting to reject them.
        // Execution is limited to the current epoch. Otherwise there can be a race where
        // the transaction is accepted but the executed effects are pruned.
        if let Some(effects) =
            self.get_transaction_cache_reader().get_executed_effects(transaction.digest())
        {
            if effects.executed_epoch() == epoch_store.epoch() {
                return Ok(());
            }
        }

        let result =
            self.handle_transaction_impl(transaction, false /* sign */, epoch_store)?;
        assert!(
            result.is_none(),
            "handle_transaction_impl should not return a signed transaction when sign is false"
        );
        Ok(())
    }

    /// Used for early client validation check for transactions before submission to server.
    /// Performs the same validation checks as handle_vote_transaction without acquiring locks.
    /// This allows for fast failure feedback to clients for non-retriable errors.
    ///
    /// The key addition is checking that owned object versions match live object versions.
    /// This is necessary because handle_transaction_deny_checks fetches objects at their
    /// requested version (which may exist in storage as historical versions), whereas
    /// validators check against the live/current version during locking (verify_live_object).
    pub fn check_transaction_validity(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction: &VerifiedTransaction,
    ) -> SomaResult<()> {
        if !epoch_store.get_reconfig_state_read_lock_guard().should_accept_user_certs() {
            return Err(SomaError::ValidatorHaltedAtEpochEnd.into());
        }

        let checked_input_objects =
            self.handle_transaction_deny_checks(transaction, epoch_store)?;

        // Check that owned object versions match live objects
        // This closely mimics verify_live_object logic from acquire_transaction_locks
        let owned_objects = checked_input_objects.inner().filter_owned_objects();
        let cache_reader = self.get_object_cache_reader();

        for obj_ref in &owned_objects {
            if let Some(live_object) = cache_reader.get_object(&obj_ref.0) {
                // Only reject if transaction references an old version. Allow newer
                // versions that may exist on validators but not yet synced to fullnode.
                if obj_ref.1 < live_object.version() {
                    return Err(SomaError::ObjectVersionUnavailableForConsumption {
                        provided_obj_ref: *obj_ref,
                        current_version: live_object.version(),
                    });
                }

                // If version matches, verify digest also matches
                if obj_ref.1 == live_object.version() && obj_ref.2 != live_object.digest() {
                    return Err(SomaError::InvalidObjectDigest {
                        object_id: obj_ref.0,
                        expected_digest: live_object.digest(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Waits for fastpath (owned, package) dependency objects to become available.
    /// Returns true if a chosen set of fastpath dependency objects are available,
    /// returns false otherwise after an internal timeout.
    pub(crate) async fn wait_for_fastpath_dependency_objects(
        &self,
        transaction: &VerifiedTransaction,
        epoch: EpochId,
    ) -> SomaResult<bool> {
        let txn_data = transaction.data().transaction_data();
        let (objects, receiving_objects) = txn_data.fastpath_dependency_objects()?;

        // Gather and filter input objects to wait for.
        let fastpath_dependency_objects: Vec<_> = objects
            .into_iter()
            .filter_map(|obj_ref| self.should_wait_for_dependency_object(obj_ref))
            .collect();
        let receiving_keys: HashSet<_> = receiving_objects
            .into_iter()
            .filter_map(|receiving_obj_ref| {
                self.should_wait_for_dependency_object(receiving_obj_ref)
            })
            .collect();
        if fastpath_dependency_objects.is_empty() && receiving_keys.is_empty() {
            return Ok(true);
        }

        // Use shorter wait timeout in simtests to exercise server-side error paths and
        // client-side retry logic.
        let max_wait =
            if cfg!(msim) { Duration::from_millis(200) } else { WAIT_FOR_FASTPATH_INPUT_TIMEOUT };

        match timeout(
            max_wait,
            self.get_object_cache_reader().notify_read_input_objects(
                &fastpath_dependency_objects,
                &receiving_keys,
                epoch,
            ),
        )
        .await
        {
            Ok(()) => Ok(true),
            // Maybe return an error for unavailable input objects,
            // and allow the caller to skip the rest of input checks?
            Err(_) => Ok(false),
        }
    }

    /// Returns Some(inputKey) if the object reference should be waited on until it is
    /// finalized, before proceeding to input checks.
    ///
    /// Incorrect decisions here should only affect user experience, not safety:
    /// - Waiting unnecessarily adds latency to transaction signing and submission.
    /// - Not waiting when needed may cause the transaction to be rejected because input object is unavailable.
    fn should_wait_for_dependency_object(&self, obj_ref: ObjectRef) -> Option<InputKey> {
        let (obj_id, cur_version, _digest) = obj_ref;
        let Some(latest_obj_ref) =
            self.get_object_cache_reader().get_latest_object_ref_or_tombstone(obj_id)
        else {
            // Object might not have been created.
            return Some(InputKey::VersionedObject {
                id: FullObjectID::new(obj_id, None),
                version: cur_version,
            });
        };
        let latest_digest = latest_obj_ref.2;
        if latest_digest == ObjectDigest::OBJECT_DIGEST_DELETED {
            // Do not wait for deleted object and rely on input check instead.
            return None;
        }
        let latest_version = latest_obj_ref.1;
        if cur_version <= latest_version {
            // Do not wait for version that already exists or has been consumed.
            // Let the input check to handle them and return the proper error.
            return None;
        }
        // Wait for the object version to become available.
        Some(InputKey::VersionedObject {
            id: FullObjectID::new(obj_id, None),
            version: cur_version,
        })
    }

    /// Wait for a certificate to be executed.
    /// For consensus transactions, it needs to be sequenced by the consensus.
    /// For owned object transactions, this function will enqueue the transaction for execution.
    // TODO: The next 3 functions are very similar. We should refactor them.
    #[instrument(level = "trace", skip_all)]
    pub async fn wait_for_certificate_execution(
        &self,
        certificate: &VerifiedCertificate,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<TransactionEffects> {
        self.wait_for_transaction_execution(
            &VerifiedExecutableTransaction::new_from_certificate(certificate.clone()),
            epoch_store,
        )
        .await
    }

    /// Wait for a transaction to be executed.
    /// For consensus transactions, it needs to be sequenced by the consensus.
    /// For owned object transactions, this function will enqueue the transaction for execution.
    #[instrument(level = "trace", skip_all)]
    pub async fn wait_for_transaction_execution(
        &self,
        transaction: &VerifiedExecutableTransaction,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<TransactionEffects> {
        trace!("execute_transaction");

        if !transaction.is_consensus_tx() {
            // Shared object transactions need to be sequenced by the consensus before enqueueing
            // for execution, done in AuthorityPerEpochStore::handle_consensus_transaction().
            // For owned object transactions, they can be enqueued for execution immediately.
            self.execution_scheduler.enqueue(
                vec![(
                    Schedulable::Transaction(transaction.clone()),
                    ExecutionEnv::new().with_scheduling_source(SchedulingSource::NonFastPath),
                )],
                epoch_store,
            );
        }

        // tx could be reverted when epoch ends, so we must be careful not to return a result
        // here after the epoch ends.
        epoch_store
            .within_alive_epoch(self.notify_read_effects(*transaction.digest()))
            .await
            .map_err(|_| SomaError::EpochEnded(epoch_store.epoch()).into())
            .and_then(|r| r)
    }

    /// Awaits the effects of executing a user transaction.
    ///
    /// Relies on consensus to enqueue the transaction for execution.
    pub async fn await_transaction_effects(
        &self,
        digest: TransactionDigest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<TransactionEffects> {
        debug!("await_transaction");

        epoch_store
            .within_alive_epoch(self.notify_read_effects(digest))
            .await
            .map_err(|_| SomaError::EpochEnded(epoch_store.epoch()).into())
            .and_then(|r| r)
    }

    /// Internal logic to execute a certificate.
    ///
    /// Guarantees that
    /// - If input objects are available, return no permanent failure.
    /// - Execution and output commit are atomic. i.e. outputs are only written to storage,
    /// on successful execution; crashed execution has no observable effect and can be retried.
    ///
    /// It is caller's responsibility to ensure input objects are available and locks are set.
    /// If this cannot be satisfied by the caller, execute_certificate() should be called instead.
    ///
    /// Should only be called within sui-core.
    #[instrument(level = "trace", skip_all)]
    pub async fn try_execute_immediately(
        &self,
        certificate: &VerifiedExecutableTransaction,
        mut execution_env: ExecutionEnv,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> ExecutionOutput<(TransactionEffects, Option<ExecutionError>)> {
        let tx_digest = certificate.digest();

        if let Some(fork_recovery) = &self.fork_recovery_state {
            if let Some(override_digest) = fork_recovery.get_transaction_override(tx_digest) {
                warn!(
                    ?tx_digest,
                    original_digest = ?execution_env.expected_effects_digest,
                    override_digest = ?override_digest,
                    "Applying fork recovery override for transaction effects digest"
                );
                execution_env.expected_effects_digest = Some(override_digest);
            }
        }

        // prevent concurrent executions of the same tx.
        let tx_guard = epoch_store.acquire_tx_guard(certificate);

        let tx_cache_reader = self.get_transaction_cache_reader();
        if let Some(effects) = tx_cache_reader.get_executed_effects(tx_digest) {
            if let Some(expected_effects_digest) = execution_env.expected_effects_digest {
                assert_eq!(
                    effects.digest(),
                    expected_effects_digest,
                    "Unexpected effects digest for transaction {:?}",
                    tx_digest
                );
            }
            tx_guard.release();
            return ExecutionOutput::Success((effects, None));
        }

        let execution_start_time = Instant::now();

        // Any caller that verifies the signatures on the certificate will have already checked the
        // epoch. But paths that don't verify sigs (e.g. execution from checkpoint, reading from db)
        // present the possibility of an epoch mismatch. If this cert is not finalzied in previous
        // epoch, then it's invalid.
        let Some(execution_guard) = self.execution_lock_for_executable_transaction(certificate)
        else {
            tx_guard.release();
            return ExecutionOutput::EpochEnded;
        };
        // Since we obtain a reference to the epoch store before taking the execution lock, it's
        // possible that reconfiguration has happened and they no longer match.
        // TODO: We may not need the following check anymore since the scheduler
        // should have checked that the certificate is from the same epoch as epoch_store.
        if *execution_guard != epoch_store.epoch() {
            tx_guard.release();
            info!("The epoch of the execution_guard doesn't match the epoch store");
            return ExecutionOutput::EpochEnded;
        }

        let scheduling_source = execution_env.scheduling_source;
        let mysticeti_fp_outputs = tx_cache_reader.get_mysticeti_fastpath_outputs(tx_digest);

        let (transaction_outputs, execution_error_opt) = if let Some(outputs) = mysticeti_fp_outputs
        {
            assert!(
                !certificate.is_consensus_tx(),
                "Mysticeti fastpath executed transactions cannot be consensus transactions"
            );
            // If this transaction is not scheduled from fastpath, it must be either
            // from consensus or from checkpoint, i.e. it must be finalized.
            // To avoid re-executing fastpath transactions, we check if the outputs are already
            // in the mysticeti fastpath outputs cache. If so, we skip the execution and commit the transaction.
            debug!(
                ?tx_digest,
                "Mysticeti fastpath certified transaction outputs found in cache, skipping execution and committing"
            );
            (outputs, None)
        } else {
            let (transaction_outputs, execution_error_opt) = match self.process_certificate(
                &tx_guard,
                &execution_guard,
                certificate,
                execution_env,
                epoch_store,
            ) {
                ExecutionOutput::Success(result) => result,
                output => return output.unwrap_err(),
            };
            (Arc::new(transaction_outputs), execution_error_opt)
        };

        let effects = transaction_outputs.effects.clone();
        debug!(
            ?tx_digest,
            fx_digest=?effects.digest(),
            "process_certificate succeeded in {:.3}ms",
            (execution_start_time.elapsed().as_micros() as f64) / 1000.0
        );

        if scheduling_source == SchedulingSource::MysticetiFastPath {
            self.get_cache_writer().write_fastpath_transaction_outputs(transaction_outputs);
        } else {
            utils::fail_point!("crash-before-commit-certificate");
            let commit_result =
                self.commit_certificate(certificate, transaction_outputs, epoch_store);
            if let Err(err) = commit_result {
                error!(?tx_digest, "Error committing transaction: {err}");
                tx_guard.release();
                return ExecutionOutput::Fatal(err);
            }
        }

        tx_guard.commit_tx();

        ExecutionOutput::Success((effects, execution_error_opt))
    }

    pub fn read_objects_for_execution(
        &self,
        tx_lock: &CertLockGuard,
        certificate: &VerifiedExecutableTransaction,
        assigned_shared_object_versions: AssignedVersions,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<InputObjects> {
        let input_objects = &certificate.data().transaction_data().input_objects()?;
        self.input_loader.read_objects_for_execution(
            &certificate.key(),
            tx_lock,
            input_objects,
            &assigned_shared_object_versions,
            epoch_store.epoch(),
        )
    }

    /// Test only wrapper for `try_execute_immediately()` above, useful for executing change epoch
    /// transactions. Assumes execution will always succeed.
    pub async fn try_execute_for_test(
        &self,
        certificate: &VerifiedCertificate,
        execution_env: ExecutionEnv,
    ) -> (VerifiedSignedTransactionEffects, Option<ExecutionError>) {
        let epoch_store = self.epoch_store_for_testing();
        let (effects, execution_error_opt) = self
            .try_execute_immediately(
                &VerifiedExecutableTransaction::new_from_certificate(certificate.clone()),
                execution_env,
                &epoch_store,
            )
            .await
            .unwrap();
        let signed_effects = self.sign_effects(effects, &epoch_store).unwrap();
        (signed_effects, execution_error_opt)
    }

    pub async fn notify_read_effects(
        &self,
        digest: TransactionDigest,
    ) -> SomaResult<TransactionEffects> {
        Ok(self
            .get_transaction_cache_reader()
            .notify_read_executed_effects(&[digest])
            .await
            .pop()
            .expect("must return correct number of effects"))
    }

    fn check_owned_locks(&self, owned_object_refs: &[ObjectRef]) -> SomaResult {
        self.get_object_cache_reader().check_owned_objects_are_live(owned_object_refs)
    }

    /// This function captures the required state to debug a forked transaction.
    /// The dump is written to a file in dir `path`, with name prefixed by the transaction digest.
    /// NOTE: Since this info escapes the validator context,
    /// make sure not to leak any private info here
    pub(crate) fn debug_dump_transaction_state(
        &self,
        tx_digest: &TransactionDigest,
        effects: &TransactionEffects,
        expected_effects_digest: TransactionEffectsDigest,
        inner_temporary_store: &InnerTemporaryStore,
        certificate: &VerifiedExecutableTransaction,
        debug_dump_config: &StateDebugDumpConfig,
    ) -> SomaResult<PathBuf> {
        let dump_dir =
            debug_dump_config.dump_file_directory.as_ref().cloned().unwrap_or(std::env::temp_dir());
        let epoch_store = self.load_epoch_store_one_call_per_task();

        NodeStateDump::new(
            tx_digest,
            effects,
            expected_effects_digest,
            self.get_object_store().as_ref(),
            &epoch_store,
            inner_temporary_store,
            certificate,
        )?
        .write_to_file(&dump_dir)
        .map_err(|e| SomaError::FileIOError(e.to_string()).into())
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn process_certificate(
        &self,
        tx_guard: &CertTxGuard,
        execution_guard: &ExecutionLockReadGuard<'_>,
        certificate: &VerifiedExecutableTransaction,
        execution_env: ExecutionEnv,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> ExecutionOutput<(TransactionOutputs, Option<ExecutionError>)> {
        let tx_digest = *certificate.digest();

        let input_objects = match self.read_objects_for_execution(
            tx_guard.as_lock_guard(),
            certificate,
            execution_env.assigned_versions,
            epoch_store,
        ) {
            Ok(objects) => objects,
            Err(e) => return ExecutionOutput::Fatal(e),
        };

        let expected_effects_digest = match execution_env.expected_effects_digest {
            Some(expected_effects_digest) => Some(expected_effects_digest),
            None => {
                // We could be re-executing a previously executed but uncommitted transaction, perhaps after
                // restarting with a new binary. In this situation, if we have published an effects signature,
                // we must be sure not to equivocate.
                // TODO: read from cache instead of DB
                match epoch_store.get_signed_effects_digest(&tx_digest) {
                    Ok(digest) => digest,
                    Err(e) => return ExecutionOutput::Fatal(e),
                }
            }
        };

        // Errors originating from prepare_certificate may be transient (failure to read locks) or
        // non-transient (transaction input is invalid, move vm errors). However, all errors from
        // this function occur before we have written anything to the db, so we commit the tx
        // guard and rely on the client to retry the tx (if it was transient).
        self.execute_certificate(
            execution_guard,
            certificate,
            input_objects,
            expected_effects_digest,
            epoch_store,
        )
    }

    #[instrument(level = "trace", skip_all)]
    fn commit_certificate(
        &self,
        certificate: &VerifiedExecutableTransaction,
        transaction_outputs: Arc<TransactionOutputs>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        let tx_digest = certificate.digest();

        // The insertion to epoch_store is not atomic with the insertion to the perpetual store. This is OK because
        // we insert to the epoch store first. And during lookups we always look up in the perpetual store first.
        epoch_store.insert_executed_in_epoch(tx_digest);
        let key = certificate.key();
        if !matches!(key, TransactionKey::Digest(_)) {
            epoch_store.insert_tx_key(key, *tx_digest)?;
        }

        self.get_cache_writer().write_transaction_outputs(epoch_store.epoch(), transaction_outputs);

        Ok(())
    }

    /// execute_certificate validates the transaction input, and executes the certificate,
    /// returning transaction outputs.
    ///
    /// It reads state from the db (both owned and shared locks), but it has no side effects.
    ///
    /// Executes a certificate and returns an ExecutionOutput.
    /// The function can fail with Fatal errors (e.g., the transaction input is invalid,
    /// locks are not held correctly, etc.) or transient errors (e.g., db read errors).
    #[instrument(level = "trace", skip_all)]
    fn execute_certificate(
        &self,
        _execution_guard: &ExecutionLockReadGuard<'_>,
        certificate: &VerifiedExecutableTransaction,
        input_objects: InputObjects,
        expected_effects_digest: Option<TransactionEffectsDigest>,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> ExecutionOutput<(TransactionOutputs, Option<ExecutionError>)> {
        // The cost of partially re-auditing a transaction before execution is tolerated.
        // This step is required for correctness because, for example, ConsensusAddressOwner
        // object owner may have changed between signing and execution.
        let input_objects = match transaction_checks::check_certificate_input(
            certificate,
            input_objects,
            epoch_store.protocol_config(),
        ) {
            Ok(result) => result,
            Err(e) => return ExecutionOutput::Fatal(e),
        };

        let owned_object_refs = input_objects.inner().filter_owned_objects();
        if let Err(e) = self.check_owned_locks(&owned_object_refs) {
            return ExecutionOutput::Fatal(e);
        }
        let tx_digest = *certificate.digest();
        let protocol_config = epoch_store.protocol_config();
        let transaction_data = &certificate.data().intent_message().value;
        let (kind, signer, gas_payment) = transaction_data.execution_parts();
        let early_execution_error = get_early_execution_error(
            &tx_digest,
            &input_objects,
            self.config.certificate_deny_config.certificate_deny_set(),
        );
        let execution_params = match early_execution_error {
            Some(error) => ExecutionOrEarlyError::Err(error),
            None => ExecutionOrEarlyError::Ok(()),
        };

        #[allow(unused_mut)]
        let (inner, effects, execution_error_opt) = execute_transaction(
            epoch_store.epoch(),
            epoch_store.protocol_config().execution_version(),
            self.get_object_store().as_ref(),
            tx_digest,
            kind,
            signer,
            gas_payment,
            input_objects,
            execution_params,
            epoch_store.epoch_start_state().fee_parameters(),
        );

        if let Some(expected_effects_digest) = expected_effects_digest {
            if effects.digest() != expected_effects_digest {
                // We dont want to mask the original error, so we log it and continue.
                match self.debug_dump_transaction_state(
                    &tx_digest,
                    &effects,
                    expected_effects_digest,
                    &inner,
                    certificate,
                    &self.config.state_debug_dump_config,
                ) {
                    Ok(out_path) => {
                        info!(
                            "Dumped node state for transaction {} to {}",
                            tx_digest,
                            out_path.as_path().display().to_string()
                        );
                    }
                    Err(e) => {
                        error!("Error dumping state for transaction {}: {e}", tx_digest);
                    }
                }
                error!(
                    ?tx_digest,
                    ?expected_effects_digest,
                    actual_effects = ?effects,
                    "fork detected!"
                );
                if let Err(e) = self.checkpoint_store.record_transaction_fork_detected(
                    tx_digest,
                    expected_effects_digest,
                    effects.digest(),
                ) {
                    error!("Failed to record transaction fork: {e}");
                }

                panic!(
                    "Transaction {} is expected to have effects digest {}, but got {}!",
                    tx_digest,
                    expected_effects_digest,
                    effects.digest()
                );
            }
        }

        // index certificate
        let _ = self.post_process_one_tx(certificate, &effects, &inner, epoch_store).tap_err(|e| {
            error!(?tx_digest, "tx post processing failed: {e}");
        });

        let transaction_outputs = TransactionOutputs::build_transaction_outputs(
            certificate.clone().into_unsigned(),
            effects,
            inner,
        );

        ExecutionOutput::Success((transaction_outputs, execution_error_opt))
    }

    pub fn simulate_transaction(
        &self,
        mut transaction: TransactionData,
        checks: TransactionChecks,
    ) -> SomaResult<SimulateTransactionResult> {
        if transaction.kind().is_system_tx() {
            return Err(SomaError::UnsupportedFeatureError {
                error: "simulate does not support system transactions".to_string(),
            }
            .into());
        }

        let epoch_store = self.load_epoch_store_one_call_per_task();
        if !self.is_fullnode(&epoch_store) {
            return Err(SomaError::UnsupportedFeatureError {
                error: "simulate is only supported on fullnodes".to_string(),
            }
            .into());
        }

        let input_object_kinds = transaction.input_objects()?;
        let receiving_object_refs = transaction.receiving_objects();

        transaction_checks::check_transaction_for_signing(
            &transaction,
            &[],
            &input_object_kinds,
            &receiving_object_refs,
            &self.config.transaction_deny_config,
        )?;

        let (mut input_objects, receiving_objects) = self.input_loader.read_objects_for_signing(
            // We don't want to cache this transaction since it's a simulation.
            None,
            &input_object_kinds,
            &receiving_object_refs,
            epoch_store.epoch(),
        )?;

        // mock a gas object if one was not provided
        let mock_gas_id = if transaction.gas().is_empty() {
            let mock_gas_object = Object::new_coin(
                ObjectID::MAX,
                DEV_INSPECT_GAS_COIN_VALUE,
                Owner::AddressOwner(transaction.sender()),
                TransactionDigest::genesis_marker(),
            );
            let mock_gas_object_ref = mock_gas_object.compute_object_reference();
            *transaction.gas_mut() = vec![mock_gas_object_ref];
            input_objects.push(ObjectReadResult::new_from_gas_object(&mock_gas_object));
            Some(mock_gas_object.id())
        } else {
            None
        };

        let checked_input_objects = transaction_checks::check_transaction_input(
            epoch_store.protocol_config(),
            &transaction,
            input_objects,
            &receiving_objects,
        )?;

        let (kind, signer, gas_payment) = transaction.execution_parts();
        let early_execution_error = get_early_execution_error(
            &transaction.digest(),
            &checked_input_objects,
            self.config.certificate_deny_config.certificate_deny_set(),
        );
        let execution_params = match early_execution_error {
            Some(error) => ExecutionOrEarlyError::Err(error),
            None => ExecutionOrEarlyError::Ok(()),
        };

        let (inner, effects, execution_error_opt) = execute_transaction(
            epoch_store.epoch(),
            epoch_store.protocol_config().execution_version(),
            self.get_object_store().as_ref(),
            transaction.digest(),
            kind,
            signer,
            gas_payment,
            checked_input_objects,
            execution_params,
            epoch_store.epoch_start_state().fee_parameters(),
        );

        let execution_result: ExecutionResult = match execution_error_opt {
            None => Ok(()),
            Some(error) => Err(error.to_execution_status()),
        };

        let object_set = {
            let objects = {
                let mut objects = ObjectSet(BTreeMap::new());

                for o in inner.input_objects.into_values().chain(inner.written.into_values()) {
                    objects.insert(o);
                }

                objects
            };

            let object_keys = types::storage::get_transaction_object_set(&transaction, &effects);

            let mut set = types::full_checkpoint_content::ObjectSet::default();
            for k in object_keys {
                if let Some(o) = objects.get(&k) {
                    set.insert(o.clone());
                }
            }

            set
        };

        Ok(SimulateTransactionResult {
            objects: object_set,
            effects,
            execution_result,
            mock_gas_id,
        })
    }

    pub fn is_tx_already_executed(&self, digest: &TransactionDigest) -> bool {
        self.get_transaction_cache_reader().is_tx_already_executed(digest)
    }

    #[instrument(level = "trace", skip_all, err(level = "debug"))]
    fn post_process_one_tx(
        &self,
        certificate: &VerifiedExecutableTransaction,
        effects: &TransactionEffects,
        inner_temporary_store: &InnerTemporaryStore,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult {
        let Some(rpc_index) = &self.rpc_index else {
            return Ok(());
        };

        let tx_digest = certificate.digest();

        // Index the transaction's object changes immediately
        rpc_index
            .index_executed_tx(
                effects,
                &inner_temporary_store.written,
                &inner_temporary_store.input_objects,
            )
            .tap_err(|e| error!(?tx_digest, "Post processing - Couldn't index tx: {e}"))?;

        Ok(())
    }

    pub fn unixtime_now_ms() -> u64 {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_millis();
        u64::try_from(now).expect("Travelling in time machine")
    }

    // TODO(fastpath): update this handler for Mysticeti fastpath.
    // There will no longer be validator quorum signed transactions or effects.
    // The proof of finality needs to come from checkpoints.
    #[instrument(level = "trace", skip_all)]
    pub async fn handle_transaction_info_request(
        &self,
        request: TransactionInfoRequest,
    ) -> SomaResult<TransactionInfoResponse> {
        let epoch_store = self.load_epoch_store_one_call_per_task();
        let (transaction, status) = self
            .get_transaction_status(&request.transaction_digest, &epoch_store)?
            .ok_or(SomaError::TransactionNotFound { digest: request.transaction_digest })?;
        Ok(TransactionInfoResponse { transaction, status })
    }

    #[instrument(level = "trace", skip_all)]
    pub async fn handle_object_info_request(
        &self,
        request: ObjectInfoRequest,
    ) -> SomaResult<ObjectInfoResponse> {
        let (_, requested_object_seq, _) =
            self.get_object_or_tombstone(request.object_id).await.ok_or(
                SomaError::ObjectNotFound { object_id: request.object_id, version: None }
            )?;

        let object = self
            .get_object_store()
            .get_object_by_key(&request.object_id, requested_object_seq)
            .ok_or(SomaError::ObjectNotFound {
                object_id: request.object_id,
                version: Some(requested_object_seq),
            })?;

        Ok(ObjectInfoResponse { object })
    }

    #[instrument(level = "trace", skip_all)]
    pub fn handle_checkpoint_request(
        &self,
        request: &CheckpointRequest,
    ) -> SomaResult<CheckpointResponse> {
        let summary = if request.certified {
            let summary = match request.sequence_number {
                Some(seq) => self.checkpoint_store.get_checkpoint_by_sequence_number(seq)?,
                None => self.checkpoint_store.get_latest_certified_checkpoint()?,
            }
            .map(|v| v.into_inner());
            summary.map(CheckpointSummaryResponse::Certified)
        } else {
            let summary = match request.sequence_number {
                Some(seq) => self.checkpoint_store.get_locally_computed_checkpoint(seq)?,
                None => self.checkpoint_store.get_latest_locally_computed_checkpoint()?,
            };
            summary.map(CheckpointSummaryResponse::Pending)
        };
        let contents = match &summary {
            Some(s) => self.checkpoint_store.get_checkpoint_contents(&s.content_digest())?,
            None => None,
        };
        Ok(CheckpointResponse { checkpoint: summary, contents })
    }

    fn check_protocol_version(
        supported_protocol_versions: SupportedProtocolVersions,
        current_version: ProtocolVersion,
    ) {
        info!("current protocol version is now {:?}", current_version);
        info!("supported versions are: {:?}", supported_protocol_versions);
        if !supported_protocol_versions.is_version_supported(current_version) {
            let msg = format!(
                "Unsupported protocol version. The network is at {:?}, but this SomaNode only supports: {:?}. Shutting down.",
                current_version, supported_protocol_versions,
            );

            error!("{}", msg);
            eprintln!("{}", msg);

            #[cfg(not(msim))]
            std::process::exit(1);

            #[cfg(msim)]
            msim::task::shutdown_current_node();
        }
    }

    #[allow(clippy::disallowed_methods)] // allow unbounded_channel()
    pub async fn new(
        name: AuthorityName,
        secret: StableSyncAuthoritySigner,
        supported_protocol_versions: SupportedProtocolVersions,
        store: Arc<AuthorityStore>,
        execution_cache_trait_pointers: ExecutionCacheTraitPointers,
        epoch_store: Arc<AuthorityPerEpochStore>,
        committee_store: Arc<CommitteeStore>,
        rpc_index: Option<Arc<RpcIndexStore>>,
        checkpoint_store: Arc<CheckpointStore>,
        genesis_objects: &[Object],
        config: NodeConfig,
        validator_tx_finalizer: Option<Arc<ValidatorTxFinalizer<NetworkAuthorityClient>>>,
        chain_identifier: ChainIdentifier,
        pruner_db: Option<Arc<AuthorityPrunerTables>>,
        pruner_watermarks: Arc<PrunerWatermarks>,
    ) -> Arc<Self> {
        Self::check_protocol_version(supported_protocol_versions, epoch_store.protocol_version());

        let (tx_ready_certificates, rx_ready_certificates) = unbounded_channel();
        let execution_scheduler = Arc::new(ExecutionScheduler::new(
            execution_cache_trait_pointers.object_cache_reader.clone(),
            execution_cache_trait_pointers.transaction_cache_reader.clone(),
            tx_ready_certificates,
            &epoch_store,
        ));
        let (tx_execution_shutdown, rx_execution_shutdown) = oneshot::channel();

        let _authority_per_epoch_pruner = AuthorityPerEpochStorePruner::new(
            epoch_store.get_parent_path(),
            &config.authority_store_pruning_config,
        );
        let _pruner = AuthorityStorePruner::new(
            store.perpetual_tables.clone(),
            checkpoint_store.clone(),
            rpc_index.clone(),
            config.authority_store_pruning_config.clone(),
            epoch_store.committee().authority_exists(&name),
            epoch_store.epoch_start_state().epoch_duration_ms(),
            pruner_db,
            pruner_watermarks,
        );
        let input_loader =
            TransactionInputLoader::new(execution_cache_trait_pointers.object_cache_reader.clone());
        let epoch = epoch_store.epoch();

        let fork_recovery_state = config.fork_recovery.as_ref().map(|fork_config| {
            ForkRecoveryState::new(Some(fork_config))
                .expect("Failed to initialize fork recovery state")
        });

        let state = Arc::new(AuthorityState {
            name,
            secret,
            execution_lock: RwLock::new(epoch),
            epoch_store: ArcSwap::new(epoch_store.clone()),
            input_loader,
            execution_cache_trait_pointers,
            rpc_index,
            checkpoint_store,
            committee_store,
            execution_scheduler,
            tx_execution_shutdown: Mutex::new(Some(tx_execution_shutdown)),
            _pruner,
            _authority_per_epoch_pruner,
            config,
            validator_tx_finalizer,
            chain_identifier,
            fork_recovery_state,
        });

        // Start a task to execute ready certificates.
        let authority_state = Arc::downgrade(&state);
        tokio::spawn(execution_process(
            authority_state,
            rx_ready_certificates,
            rx_execution_shutdown,
        ));

        state
    }

    // TODO: Consolidate our traits to reduce the number of methods here.
    pub fn get_object_cache_reader(&self) -> &Arc<dyn ObjectCacheRead> {
        &self.execution_cache_trait_pointers.object_cache_reader
    }

    pub fn get_transaction_cache_reader(&self) -> &Arc<dyn TransactionCacheRead> {
        &self.execution_cache_trait_pointers.transaction_cache_reader
    }

    pub fn get_cache_writer(&self) -> &Arc<dyn ExecutionCacheWrite> {
        &self.execution_cache_trait_pointers.cache_writer
    }

    pub fn get_object_store(&self) -> &Arc<dyn ObjectStore + Send + Sync> {
        &self.execution_cache_trait_pointers.object_store
    }

    pub fn get_reconfig_api(&self) -> &Arc<dyn ExecutionCacheReconfigAPI> {
        &self.execution_cache_trait_pointers.reconfig_api
    }

    pub fn get_global_state_hash_store(&self) -> &Arc<dyn GlobalStateHashStore> {
        &self.execution_cache_trait_pointers.global_state_hash_store
    }

    pub fn get_state_sync_store(&self) -> &Arc<dyn StateSyncAPI> {
        &self.execution_cache_trait_pointers.state_sync_store
    }

    pub fn get_cache_commit(&self) -> &Arc<dyn ExecutionCacheCommit> {
        &self.execution_cache_trait_pointers.cache_commit
    }

    pub fn database_for_testing(&self) -> Arc<AuthorityStore> {
        self.execution_cache_trait_pointers.testing_api.database_for_testing()
    }

    pub async fn prune_checkpoints_for_eligible_epochs_for_testing(
        &self,
        config: NodeConfig,
    ) -> anyhow::Result<()> {
        use crate::authority_store_pruner::PrunerWatermarks;
        let watermarks = Arc::new(PrunerWatermarks::default());
        AuthorityStorePruner::prune_checkpoints_for_eligible_epochs(
            &self.database_for_testing().perpetual_tables,
            &self.checkpoint_store,
            self.rpc_index.as_deref(),
            None,
            config.authority_store_pruning_config,
            EPOCH_DURATION_MS_FOR_TESTING,
            &watermarks,
        )
        .await
    }

    pub fn execution_scheduler(&self) -> &Arc<ExecutionScheduler> {
        &self.execution_scheduler
    }

    /// Attempts to acquire execution lock for an executable transaction.
    /// Returns Some(lock) if the transaction is matching current executed epoch.
    /// Returns None if validator is halted at epoch end or epoch mismatch.
    pub fn execution_lock_for_executable_transaction(
        &self,
        transaction: &VerifiedExecutableTransaction,
    ) -> Option<ExecutionLockReadGuard<'_>> {
        let lock = self.execution_lock.try_read().ok()?;
        if *lock == transaction.auth_sig().epoch() {
            Some(lock)
        } else {
            // TODO: Can this still happen?
            None
        }
    }

    /// Acquires the execution lock for the duration of a transaction signing request.
    /// This prevents reconfiguration from starting until we are finished handling the signing request.
    /// Otherwise, in-memory lock state could be cleared (by `ObjectLocks::clear_cached_locks`)
    /// while we are attempting to acquire locks for the transaction.
    pub fn execution_lock_for_signing(&self) -> SomaResult<ExecutionLockReadGuard<'_>> {
        self.execution_lock.try_read().map_err(|_| SomaError::ValidatorHaltedAtEpochEnd.into())
    }

    pub async fn execution_lock_for_reconfiguration(&self) -> ExecutionLockWriteGuard<'_> {
        self.execution_lock.write().await
    }

    #[instrument(level = "error", skip_all)]
    pub async fn reconfigure(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        supported_protocol_versions: SupportedProtocolVersions,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
        state_hasher: Arc<GlobalStateHasher>,
        expensive_safety_check_config: &ExpensiveSafetyCheckConfig,
        epoch_last_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        Self::check_protocol_version(
            supported_protocol_versions,
            epoch_start_configuration.epoch_start_state().protocol_version(),
        );

        self.committee_store.insert_new_committee(&new_committee)?;

        // Wait until no transactions are being executed.
        let mut execution_lock = self.execution_lock_for_reconfiguration().await;

        // Terminate all epoch-specific tasks (those started with within_alive_epoch).
        cur_epoch_store.epoch_terminated().await;

        // Safe to being reconfiguration now. No transactions are being executed,
        // and no epoch-specific tasks are running.

        {
            let state = cur_epoch_store.get_reconfig_state_write_lock_guard();
            if state.should_accept_user_certs() {
                // Need to change this so that consensus adapter do not accept certificates from user.
                // This can happen if our local validator did not initiate epoch change locally,
                // but 2f+1 nodes already concluded the epoch.
                //
                // This lock is essentially a barrier for
                // `epoch_store.pending_consensus_certificates` table we are reading on the line after this block
                cur_epoch_store.close_user_certs(state);
            }
            // lock is dropped here
        }

        self.get_reconfig_api().clear_state_end_of_epoch(&execution_lock);
        self.check_system_consistency(cur_epoch_store, state_hasher, expensive_safety_check_config);

        self.get_reconfig_api().set_epoch_start_configuration(&epoch_start_configuration);

        // TODO: consider checkpointing dbs on reconfig
        // if let Some(checkpoint_path) = &self.db_checkpoint_config.checkpoint_path {
        //     if self
        //         .db_checkpoint_config
        //         .perform_db_checkpoints_at_epoch_end
        //     {
        //         let checkpoint_indexes = self
        //             .db_checkpoint_config
        //             .perform_index_db_checkpoints_at_epoch_end
        //             .unwrap_or(false);
        //         let current_epoch = cur_epoch_store.epoch();
        //         let epoch_checkpoint_path =
        //             checkpoint_path.join(format!("epoch_{}", current_epoch));
        //         self.checkpoint_all_dbs(
        //             &epoch_checkpoint_path,
        //             cur_epoch_store,
        //             checkpoint_indexes,
        //         )?;
        //     }
        // }

        self.get_reconfig_api().reconfigure_cache(&epoch_start_configuration).await;

        let new_epoch = new_committee.epoch;
        let new_epoch_store = self
            .reopen_epoch_db(
                cur_epoch_store,
                new_committee,
                epoch_start_configuration,
                epoch_last_checkpoint,
            )
            .await?;
        assert_eq!(new_epoch_store.epoch(), new_epoch);

        *execution_lock = new_epoch;
        // drop execution_lock after epoch store was updated
        // see also assert in AuthorityState::process_certificate
        // on the epoch store and execution lock epoch match
        Ok(new_epoch_store)
    }

    #[instrument(level = "error", skip_all)]
    fn check_system_consistency(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        state_hasher: Arc<GlobalStateHasher>,
        expensive_safety_check_config: &ExpensiveSafetyCheckConfig,
    ) {
        info!(
            "Performing soma conservation consistency check for epoch {}",
            cur_epoch_store.epoch()
        );

        // check for root state hash consistency with live object set
        if expensive_safety_check_config.enable_state_consistency_check() {
            info!("Performing state consistency check for epoch {}", cur_epoch_store.epoch());
            self.expensive_check_is_consistent_state(state_hasher.clone(), cur_epoch_store);
        }

        // Supply conservation check: verify total SOMA across all objects equals genesis total.
        // Gated behind enable_state_consistency_check (enabled in debug/test builds via
        // cfg!(debug_assertions)).
        if expensive_safety_check_config.enable_state_consistency_check() {
            info!(
                "Performing supply conservation check for epoch {}",
                cur_epoch_store.epoch()
            );
            self.check_soma_conservation(state_hasher, cur_epoch_store);
        }
    }

    /// Verify that the total SOMA supply is conserved across all live objects.
    ///
    /// Iterates every live object, sums value by category, and compares against
    /// `TOTAL_SUPPLY_SHANNONS`. Any mismatch indicates a bug in emission,
    /// staking reward, or fee accounting logic.
    fn check_soma_conservation(
        &self,
        _state_hasher: Arc<GlobalStateHasher>,
        cur_epoch_store: &AuthorityPerEpochStore,
    ) {
        use types::config::genesis_config::TOTAL_SUPPLY_SHANNONS;
        use types::object::ObjectType;

        // Get system state from the object store for accurate balance accounting
        let system_state = self
            .get_system_state_object_for_testing()
            .expect("SystemState must exist for conservation check");

        let mut system_state_balance: u128 = 0;

        // Emission pool
        system_state_balance += system_state.emission_pool().balance as u128;
        // Safe mode accumulators
        system_state_balance += system_state.safe_mode_accumulated_fees() as u128;
        system_state_balance += system_state.safe_mode_accumulated_emissions() as u128;

        // Validator staking pools (active, pending, inactive)
        for v in &system_state.validators().validators {
            system_state_balance += v.staking_pool.soma_balance as u128;
            system_state_balance += v.staking_pool.pending_stake as u128;
        }
        for v in &system_state.validators().pending_validators {
            system_state_balance += v.staking_pool.soma_balance as u128;
            system_state_balance += v.staking_pool.pending_stake as u128;
        }
        for v in system_state.validators().inactive_validators.values() {
            system_state_balance += v.staking_pool.soma_balance as u128;
            system_state_balance += v.staking_pool.pending_stake as u128;
        }

        // Model staking pools (active, pending, inactive)
        for m in system_state.model_registry().active_models.values() {
            system_state_balance += m.staking_pool.soma_balance as u128;
            system_state_balance += m.staking_pool.pending_stake as u128;
        }
        for m in system_state.model_registry().pending_models.values() {
            system_state_balance += m.staking_pool.soma_balance as u128;
            system_state_balance += m.staking_pool.pending_stake as u128;
        }
        for m in system_state.model_registry().inactive_models.values() {
            system_state_balance += m.staking_pool.soma_balance as u128;
            system_state_balance += m.staking_pool.pending_stake as u128;
        }

        // Iterate all live objects to sum coin, target, and challenge balances
        let mut coin_balance: u128 = 0;
        let mut target_balance: u128 = 0;
        let mut challenge_balance: u128 = 0;
        let mut object_count: u64 = 0;

        for live_obj in self.get_global_state_hash_store().iter_live_object_set() {
            let obj = match live_obj {
                types::object::LiveObject::Normal(obj) => obj,
            };
            object_count += 1;

            match obj.type_() {
                ObjectType::Coin => {
                    if let Some(balance) = obj.as_coin() {
                        coin_balance += balance as u128;
                    }
                }
                ObjectType::Target => {
                    if let Some(target) = obj.as_target() {
                        target_balance += target.reward_pool as u128;
                        target_balance += target.bond_amount as u128;
                    }
                }
                ObjectType::Challenge => {
                    if let Some(challenge) = obj.as_challenge() {
                        challenge_balance += challenge.challenger_bond as u128;
                    }
                }
                // SystemState: accounted above via get_system_state_object_for_testing
                // StakedSoma, Submission: no SOMA value (receipts only)
                _ => {}
            }
        }

        let total_accounted =
            coin_balance + system_state_balance + target_balance + challenge_balance;
        let expected = TOTAL_SUPPLY_SHANNONS as u128;

        if total_accounted != expected {
            let msg = format!(
                "SUPPLY CONSERVATION VIOLATION at epoch {}! \
                 Expected {expected}, got {total_accounted} \
                 (coins={coin_balance}, system_state={system_state_balance}, \
                 targets={target_balance}, challenges={challenge_balance}, \
                 objects_scanned={object_count})",
                cur_epoch_store.epoch(),
            );
            if cfg!(msim) {
                panic!("{msg}");
            } else {
                error!("{msg}");
            }
        } else {
            info!(
                "Supply conservation check passed for epoch {} \
                 (total={expected}, coins={coin_balance}, \
                 system_state={system_state_balance}, targets={target_balance}, \
                 challenges={challenge_balance}, objects={object_count})",
                cur_epoch_store.epoch(),
            );
        }
    }

    fn expensive_check_is_consistent_state(
        &self,
        state_hasher: Arc<GlobalStateHasher>,
        cur_epoch_store: &AuthorityPerEpochStore,
    ) {
        let live_object_set_hash = state_hasher.digest_live_object_set();

        let root_state_hash: ECMHLiveObjectSetDigest = self
            .get_global_state_hash_store()
            .get_root_state_hash_for_epoch(cur_epoch_store.epoch())
            .expect("Retrieving root state hash cannot fail")
            .expect("Root state hash for epoch must exist")
            .1
            .digest()
            .into();

        let is_inconsistent = root_state_hash != live_object_set_hash;
        if is_inconsistent {
            if cfg!(msim) {
                panic!(
                    "Inconsistent state detected at epoch {}: root state hash: {:?}, \
                     live object set hash: {:?}",
                    cur_epoch_store.epoch(),
                    root_state_hash,
                    live_object_set_hash
                );
            } else {
                error!(
                    "Inconsistent state detected: root state hash: {:?}, live object set hash: {:?}",
                    root_state_hash, live_object_set_hash
                );
            }
        } else {
            info!("State consistency check passed");
        }
    }

    pub fn current_epoch_for_testing(&self) -> EpochId {
        self.epoch_store_for_testing().epoch()
    }

    #[instrument(level = "error", skip_all)]
    pub fn checkpoint_all_dbs(
        &self,
        checkpoint_path: &Path,
        cur_epoch_store: &AuthorityPerEpochStore,
        checkpoint_indexes: bool,
    ) -> SomaResult {
        let current_epoch = cur_epoch_store.epoch();

        if checkpoint_path.exists() {
            info!("Skipping db checkpoint as it already exists for epoch: {current_epoch}");
            return Ok(());
        }

        let checkpoint_path_tmp = checkpoint_path.with_extension("tmp");
        let store_checkpoint_path_tmp = checkpoint_path_tmp.join("store");

        if checkpoint_path_tmp.exists() {
            fs::remove_dir_all(&checkpoint_path_tmp)
                .map_err(|e| SomaError::FileIOError(e.to_string()))?;
        }

        fs::create_dir_all(&checkpoint_path_tmp)
            .map_err(|e| SomaError::FileIOError(e.to_string()))?;
        fs::create_dir(&store_checkpoint_path_tmp)
            .map_err(|e| SomaError::FileIOError(e.to_string()))?;

        // NOTE: Do not change the order of invoking these checkpoint calls
        // We want to snapshot checkpoint db first to not race with state sync
        self.checkpoint_store.checkpoint_db(&checkpoint_path_tmp.join("checkpoints"))?;

        // TODO: checkpoint dbs
        // self.get_reconfig_api()
        //     .checkpoint_db(&store_checkpoint_path_tmp.join("perpetual"))?;

        // self.committee_store
        //     .checkpoint_db(&checkpoint_path_tmp.join("epochs"))?;

        fs::rename(checkpoint_path_tmp, checkpoint_path)
            .map_err(|e| SomaError::FileIOError(e.to_string()))?;
        Ok(())
    }

    /// Load the current epoch store. This can change during reconfiguration. To ensure that
    /// we never end up accessing different epoch stores in a single task, we need to make sure
    /// that this is called once per task. Each call needs to be carefully audited to ensure it is
    /// the case. This also means we should minimize the number of call-sites. Only call it when
    /// there is no way to obtain it from somewhere else.
    pub fn load_epoch_store_one_call_per_task(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.epoch_store.load()
    }

    // Load the epoch store, should be used in tests only.
    pub fn epoch_store_for_testing(&self) -> Guard<Arc<AuthorityPerEpochStore>> {
        self.load_epoch_store_one_call_per_task()
    }

    pub fn clone_committee_for_testing(&self) -> Committee {
        Committee::clone(self.epoch_store_for_testing().committee())
    }

    #[instrument(level = "trace", skip_all)]
    pub async fn get_object(&self, object_id: &ObjectID) -> Option<Object> {
        self.get_object_store().get_object(object_id)
    }

    // This function is only used for testing.
    pub fn get_system_state_object_for_testing(&self) -> SomaResult<SystemState> {
        self.get_object_cache_reader().get_system_state_object()
    }

    #[instrument(level = "trace", skip_all)]
    fn get_transaction_checkpoint_sequence(
        &self,
        digest: &TransactionDigest,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<Option<CheckpointSequenceNumber>> {
        epoch_store.get_transaction_checkpoint(digest)
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> SomaResult<Option<VerifiedCheckpoint>> {
        Ok(self.checkpoint_store.get_checkpoint_by_sequence_number(sequence_number)?)
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_transaction_checkpoint_for_tests(
        &self,
        digest: &TransactionDigest,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<Option<VerifiedCheckpoint>> {
        let checkpoint = self.get_transaction_checkpoint_sequence(digest, epoch_store)?;
        let Some(checkpoint) = checkpoint else {
            return Ok(None);
        };
        let checkpoint = self.checkpoint_store.get_checkpoint_by_sequence_number(checkpoint)?;
        Ok(checkpoint)
    }

    /// Chain Identifier is the digest of the genesis checkpoint.
    pub fn get_chain_identifier(&self) -> ChainIdentifier {
        self.chain_identifier
    }

    pub fn get_fork_recovery_state(&self) -> Option<&ForkRecoveryState> {
        self.fork_recovery_state.as_ref()
    }

    #[instrument(level = "trace", skip_all)]
    fn read_object_at_version(
        &self,
        object_id: &ObjectID,
        version: Version,
    ) -> SomaResult<Option<Object>> {
        let Some(object) = self.get_object_cache_reader().get_object_by_key(object_id, version)
        else {
            return Ok(None);
        };

        Ok(Some(object))
    }

    #[instrument(level = "trace", skip_all)]
    pub fn multi_get_checkpoint_by_sequence_number(
        &self,
        sequence_numbers: &[CheckpointSequenceNumber],
    ) -> SomaResult<Vec<Option<VerifiedCheckpoint>>> {
        Ok(self.checkpoint_store.multi_get_checkpoint_by_sequence_number(sequence_numbers)?)
    }

    pub fn get_transaction_input_objects(
        &self,
        effects: &TransactionEffects,
    ) -> SomaResult<Vec<Object>> {
        types::storage::get_transaction_input_objects(self.get_object_store(), effects)
            .map_err(Into::into)
    }

    pub fn get_transaction_output_objects(
        &self,
        effects: &TransactionEffects,
    ) -> SomaResult<Vec<Object>> {
        types::storage::get_transaction_output_objects(self.get_object_store(), effects)
            .map_err(Into::into)
    }

    pub fn get_checkpoint_store(&self) -> &Arc<CheckpointStore> {
        &self.checkpoint_store
    }

    pub fn get_latest_checkpoint_sequence_number(&self) -> SomaResult<CheckpointSequenceNumber> {
        self.get_checkpoint_store()
            .get_highest_executed_checkpoint_seq_number()?
            .ok_or(SomaError::LatestCheckpointSequenceNumberNotFound)
    }

    #[cfg(msim)]
    pub fn get_highest_pruned_checkpoint_for_testing(
        &self,
    ) -> SomaResult<CheckpointSequenceNumber> {
        self.database_for_testing()
            .perpetual_tables
            .get_highest_pruned_checkpoint()
            .map(|c| c.unwrap_or(0))
            .map_err(Into::into)
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_checkpoint_summary_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> SomaResult<CheckpointSummary> {
        let verified_checkpoint =
            self.get_checkpoint_store().get_checkpoint_by_sequence_number(sequence_number)?;
        match verified_checkpoint {
            Some(verified_checkpoint) => Ok(verified_checkpoint.into_inner().into_data()),
            None => Err(SomaError::VerifiedCheckpointNotFound(sequence_number)),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_checkpoint_summary_by_digest(
        &self,
        digest: CheckpointDigest,
    ) -> SomaResult<CheckpointSummary> {
        let verified_checkpoint = self.get_checkpoint_store().get_checkpoint_by_digest(&digest)?;
        match verified_checkpoint {
            Some(verified_checkpoint) => Ok(verified_checkpoint.into_inner().into_data()),
            None => Err(SomaError::VerifiedCheckpointDigestNotFound(Base58::encode(digest))),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub fn find_genesis_txn_digest(&self) -> SomaResult<TransactionDigest> {
        let summary = self.get_verified_checkpoint_by_sequence_number(0)?.into_message();
        let content = self.get_checkpoint_contents(summary.content_digest)?;
        let genesis_transaction = content.enumerate_transactions(&summary).next();
        Ok(genesis_transaction.ok_or(SomaError::GenesisTransactionNotFound)?.1.transaction)
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_verified_checkpoint_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> SomaResult<VerifiedCheckpoint> {
        let verified_checkpoint =
            self.get_checkpoint_store().get_checkpoint_by_sequence_number(sequence_number)?;
        match verified_checkpoint {
            Some(verified_checkpoint) => Ok(verified_checkpoint),
            None => Err(SomaError::VerifiedCheckpointNotFound(sequence_number)),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_verified_checkpoint_summary_by_digest(
        &self,
        digest: CheckpointDigest,
    ) -> SomaResult<VerifiedCheckpoint> {
        let verified_checkpoint = self.get_checkpoint_store().get_checkpoint_by_digest(&digest)?;
        match verified_checkpoint {
            Some(verified_checkpoint) => Ok(verified_checkpoint),
            None => Err(SomaError::VerifiedCheckpointDigestNotFound(Base58::encode(digest))),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_checkpoint_contents(
        &self,
        digest: CheckpointContentsDigest,
    ) -> SomaResult<CheckpointContents> {
        self.get_checkpoint_store()
            .get_checkpoint_contents(&digest)?
            .ok_or(SomaError::CheckpointContentsNotFound(digest))
    }

    #[instrument(level = "trace", skip_all)]
    pub fn get_checkpoint_contents_by_sequence_number(
        &self,
        sequence_number: CheckpointSequenceNumber,
    ) -> SomaResult<CheckpointContents> {
        let verified_checkpoint =
            self.get_checkpoint_store().get_checkpoint_by_sequence_number(sequence_number)?;
        match verified_checkpoint {
            Some(verified_checkpoint) => {
                let content_digest = verified_checkpoint.into_inner().content_digest;
                self.get_checkpoint_contents(content_digest)
            }
            None => Err(SomaError::VerifiedCheckpointNotFound(sequence_number)),
        }
    }

    pub async fn insert_genesis_object(&self, object: Object) {
        self.get_reconfig_api().insert_genesis_object(object);
    }

    pub async fn insert_genesis_objects(&self, objects: &[Object]) {
        futures::future::join_all(objects.iter().map(|o| self.insert_genesis_object(o.clone())))
            .await;
    }

    /// Make a status response for a transaction
    #[instrument(level = "trace", skip_all)]
    pub fn get_transaction_status(
        &self,
        transaction_digest: &TransactionDigest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Option<(SenderSignedData, TransactionStatus)>> {
        // TODO: In the case of read path, we should not have to re-sign the effects.
        if let Some(effects) =
            self.get_signed_effects_and_maybe_resign(transaction_digest, epoch_store)?
        {
            if let Some(transaction) =
                self.get_transaction_cache_reader().get_transaction_block(transaction_digest)
            {
                let cert_sig = epoch_store.get_transaction_cert_sig(transaction_digest)?;

                return Ok(Some((
                    (*transaction).clone().into_message(),
                    TransactionStatus::Executed(cert_sig, effects.into_inner()),
                )));
            } else {
                // The read of effects and read of transaction are not atomic. It's possible that we reverted
                // the transaction (during epoch change) in between the above two reads, and we end up
                // having effects but not transaction. In this case, we just fall through.
                debug!(tx_digest=?transaction_digest, "Signed effects exist but no transaction found");
            }
        }
        if let Some(signed) = epoch_store.get_signed_transaction(transaction_digest)? {
            let (transaction, sig) = signed.into_inner().into_data_and_sig();
            Ok(Some((transaction, TransactionStatus::Signed(sig))))
        } else {
            Ok(None)
        }
    }

    /// Get the signed effects of the given transaction. If the effects was signed in a previous
    /// epoch, re-sign it so that the caller is able to form a cert of the effects in the current
    /// epoch.
    #[instrument(level = "trace", skip_all)]
    pub fn get_signed_effects_and_maybe_resign(
        &self,
        transaction_digest: &TransactionDigest,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<Option<VerifiedSignedTransactionEffects>> {
        let effects = self.get_transaction_cache_reader().get_executed_effects(transaction_digest);
        match effects {
            Some(effects) => {
                // If the transaction was executed in previous epochs, the validator will
                // re-sign the effects with new current epoch so that a client is always able to
                // obtain an effects certificate at the current epoch.
                //
                // Why is this necessary? Consider the following case:
                // - assume there are 4 validators
                // - Quorum driver gets 2 signed effects before reconfig halt
                // - The tx makes it into final checkpoint.
                // - 2 validators go away and are replaced in the new epoch.
                // - The new epoch begins.
                // - The quorum driver cannot complete the partial effects cert from the previous epoch,
                //   because it may not be able to reach either of the 2 former validators.
                // - But, if the 2 validators that stayed are willing to re-sign the effects in the new
                //   epoch, the QD can make a new effects cert and return it to the client.
                //
                // This is a considered a short-term workaround. Eventually, Quorum Driver should be able
                // to return either an effects certificate, -or- a proof of inclusion in a checkpoint. In
                // the case above, the Quorum Driver would return a proof of inclusion in the final
                // checkpoint, and this code would no longer be necessary.
                if effects.executed_epoch() != epoch_store.epoch() {
                    debug!(
                        tx_digest=?transaction_digest,
                        effects_epoch=?effects.executed_epoch(),
                        epoch=?epoch_store.epoch(),
                        "Re-signing the effects with the current epoch"
                    );
                }
                Ok(Some(self.sign_effects(effects, epoch_store)?))
            }
            None => Ok(None),
        }
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn sign_effects(
        &self,
        effects: TransactionEffects,
        epoch_store: &Arc<AuthorityPerEpochStore>,
    ) -> SomaResult<VerifiedSignedTransactionEffects> {
        let tx_digest = *effects.transaction_digest();
        let signed_effects = match epoch_store.get_effects_signature(&tx_digest)? {
            Some(sig) => {
                debug_assert!(sig.epoch == epoch_store.epoch());
                SignedTransactionEffects::new_from_data_and_sig(effects, sig)
            }
            _ => {
                let sig = AuthoritySignInfo::new(
                    epoch_store.epoch(),
                    &effects,
                    Intent::soma_app(IntentScope::TransactionEffects),
                    self.name,
                    &*self.secret,
                );

                let effects = SignedTransactionEffects::new_from_data_and_sig(effects, sig.clone());

                epoch_store.insert_effects_digest_and_signature(
                    &tx_digest,
                    effects.digest(),
                    &sig,
                )?;

                effects
            }
        };

        Ok(VerifiedSignedTransactionEffects::new_unchecked(signed_effects))
    }

    /// Get the TransactionEnvelope that currently locks the given object, if any.
    /// Since object locks are only valid for one epoch, we also need the epoch_id in the query.
    /// Returns UserInputError::ObjectNotFound if no lock records for the given object can be found.
    /// Returns UserInputError::ObjectVersionUnavailableForConsumption if the object record is at a different version.
    /// Returns Some(VerifiedEnvelope) if the given ObjectRef is locked by a certain transaction.
    /// Returns None if a lock record is initialized for the given ObjectRef but not yet locked by any transaction,
    ///     or cannot find the transaction in transaction table, because of data race etc.
    #[instrument(level = "trace", skip_all)]
    pub async fn get_transaction_lock(
        &self,
        object_ref: &ObjectRef,
        epoch_store: &AuthorityPerEpochStore,
    ) -> SomaResult<Option<VerifiedSignedTransaction>> {
        let lock_info = self.get_object_cache_reader().get_lock(*object_ref, epoch_store)?;
        let lock_info = match lock_info {
            ObjectLockStatus::LockedAtDifferentVersion { locked_ref } => {
                return Err(SomaError::ObjectVersionUnavailableForConsumption {
                    provided_obj_ref: *object_ref,
                    current_version: locked_ref.1,
                });
            }
            ObjectLockStatus::Initialized => {
                return Ok(None);
            }
            ObjectLockStatus::LockedToTx { locked_by_tx } => locked_by_tx,
        };

        epoch_store.get_signed_transaction(&lock_info)
    }

    pub async fn get_objects(&self, objects: &[ObjectID]) -> Vec<Option<Object>> {
        self.get_object_cache_reader().get_objects(objects)
    }

    pub async fn get_object_or_tombstone(&self, object_id: ObjectID) -> Option<ObjectRef> {
        self.get_object_cache_reader().get_latest_object_ref_or_tombstone(object_id)
    }

    /// Ordinarily, protocol upgrades occur when 2f + 1 + (f *
    /// ProtocolConfig::buffer_stake_for_protocol_upgrade_bps) vote for the upgrade.
    ///
    /// This method can be used to dynamic adjust the amount of buffer. If set to 0, the upgrade
    /// will go through with only 2f+1 votes.
    ///
    /// IMPORTANT: If this is used, it must be used on >=2f+1 validators (all should have the same
    /// value), or you risk halting the chain.
    pub fn set_override_protocol_upgrade_buffer_stake(
        &self,
        expected_epoch: EpochId,
        buffer_stake_bps: u64,
    ) -> SomaResult {
        let epoch_store = self.load_epoch_store_one_call_per_task();
        let actual_epoch = epoch_store.epoch();
        if actual_epoch != expected_epoch {
            return Err(SomaError::WrongEpoch { expected_epoch, actual_epoch }.into());
        }

        epoch_store.set_override_protocol_upgrade_buffer_stake(buffer_stake_bps)
    }

    pub fn clear_override_protocol_upgrade_buffer_stake(
        &self,
        expected_epoch: EpochId,
    ) -> SomaResult {
        let epoch_store = self.load_epoch_store_one_call_per_task();
        let actual_epoch = epoch_store.epoch();
        if actual_epoch != expected_epoch {
            return Err(SomaError::WrongEpoch { expected_epoch, actual_epoch }.into());
        }

        epoch_store.clear_override_protocol_upgrade_buffer_stake()
    }

    fn is_protocol_version_supported(
        current_protocol_version: ProtocolVersion,
        proposed_protocol_version: ProtocolVersion,
        protocol_config: &ProtocolConfig,
        committee: &Committee,
        capabilities: Vec<AuthorityCapabilitiesV1>,
        mut buffer_stake_bps: u64,
    ) -> Option<ProtocolVersion> {
        if buffer_stake_bps > 10000 {
            warn!("clamping buffer_stake_bps to 10000");
            buffer_stake_bps = 10000;
        }

        // For each validator, gather the protocol version and system packages that it would like
        // to upgrade to in the next epoch.
        let mut desired_upgrades: Vec<_> = capabilities
            .into_iter()
            .filter_map(|mut cap| {
                info!(
                    "validator {:?} supports {:?}",
                    cap.authority.concise(),
                    cap.supported_protocol_versions,
                );

                // A validator that only supports the current protocol version is also voting
                // against any change, because framework upgrades always require a protocol version
                // bump.
                cap.supported_protocol_versions
                    .get_version_digest(proposed_protocol_version)
                    .map(|digest| (digest, cap.authority))
            })
            .collect();

        // There can only be one set of votes that have a majority, find one if it exists.
        desired_upgrades.sort();
        desired_upgrades.into_iter().chunk_by(|(digest, _authority)| *digest).into_iter().find_map(
            |(digest, group)| {
                let mut stake_aggregator: StakeAggregator<(), true> =
                    StakeAggregator::new(Arc::new(committee.clone()));

                for (_, authority) in group {
                    stake_aggregator.insert_generic(authority, ());
                }

                let total_votes = stake_aggregator.total_votes();
                let quorum_threshold = committee.quorum_threshold();
                let f = committee.total_votes() - committee.quorum_threshold();

                // multiple by buffer_stake_bps / 10000, rounded up.
                let buffer_stake = (f * buffer_stake_bps).div_ceil(10000);
                let effective_threshold = quorum_threshold + buffer_stake;

                info!(
                    protocol_config_digest = ?digest,
                    ?total_votes,
                    ?quorum_threshold,
                    ?buffer_stake_bps,
                    ?effective_threshold,
                    ?proposed_protocol_version,
                    "support for upgrade"
                );

                let has_support = total_votes >= effective_threshold;
                has_support.then_some(proposed_protocol_version)
            },
        )
    }

    fn choose_protocol_version(
        current_protocol_version: ProtocolVersion,
        protocol_config: &ProtocolConfig,
        committee: &Committee,
        capabilities: Vec<AuthorityCapabilitiesV1>,
        buffer_stake_bps: u64,
    ) -> ProtocolVersion {
        let mut next_protocol_version = current_protocol_version;

        while let Some(version) = Self::is_protocol_version_supported(
            current_protocol_version,
            next_protocol_version + 1,
            protocol_config,
            committee,
            capabilities.clone(),
            buffer_stake_bps,
        ) {
            next_protocol_version = version;
        }

        next_protocol_version
    }

    /// Creates and execute the advance epoch transaction to effects without committing it to the database.
    /// The effects of the change epoch tx are only written to the database after a certified checkpoint has been
    /// formed and executed by CheckpointExecutor.
    ///
    /// When a framework upgraded has been decided on, but the validator does not have the new
    /// versions of the packages locally, the validator cannot form the ChangeEpochTx. In this case
    /// it returns Err, indicating that the checkpoint builder should give up trying to make the
    /// final checkpoint. As long as the network is able to create a certified checkpoint (which
    /// should be ensured by the capabilities vote), it will arrive via state sync and be executed
    /// by CheckpointExecutor.
    #[instrument(level = "error", skip_all)]
    pub async fn create_and_execute_advance_epoch_tx(
        &self,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        transaction_fees: &TransactionFee,
        checkpoint: CheckpointSequenceNumber,
        epoch_start_timestamp_ms: CheckpointTimestamp,
        // This may be less than `checkpoint - 1` if the end-of-epoch PendingCheckpoint produced
        // >1 checkpoint.
        last_checkpoint: CheckpointSequenceNumber,
        epoch_randomness: Vec<u8>,
    ) -> CheckpointBuilderResult<(SystemState, TransactionEffects)> {
        let next_epoch = epoch_store.epoch() + 1;

        let buffer_stake_bps = epoch_store.get_effective_buffer_stake_bps();

        let next_epoch_protocol_version = Self::choose_protocol_version(
            epoch_store.protocol_version(),
            epoch_store.protocol_config(),
            epoch_store.committee(),
            epoch_store.get_capabilities().expect("read capabilities from db cannot fail"),
            buffer_stake_bps,
        );

        let config = epoch_store.protocol_config();

        let tx = VerifiedTransaction::new_change_epoch_transaction(
            next_epoch,
            next_epoch_protocol_version,
            transaction_fees.total_fee,
            epoch_start_timestamp_ms,
            epoch_randomness,
        );

        let executable_tx = VerifiedExecutableTransaction::new_from_checkpoint(
            tx.clone(),
            epoch_store.epoch(),
            checkpoint,
        );

        let tx_digest = executable_tx.digest();

        info!(?next_epoch, ?tx_digest, "Creating advance epoch transaction");

        utils::fail_point_async!("change_epoch_tx_delay");

        let tx_lock = epoch_store.acquire_tx_lock(tx_digest);

        // The tx could have been executed by state sync already - if so simply return an error.
        // The checkpoint builder will shortly be terminated by reconfiguration anyway.
        if self.get_transaction_cache_reader().is_tx_already_executed(tx_digest) {
            warn!("change epoch tx has already been executed via state sync");
            return Err(CheckpointBuilderError::ChangeEpochTxAlreadyExecuted);
        }

        let Some(execution_guard) = self.execution_lock_for_executable_transaction(&executable_tx)
        else {
            return Err(CheckpointBuilderError::ChangeEpochTxAlreadyExecuted);
        };

        // We must manually assign the shared object versions to the transaction before executing it.
        // This is because we do not sequence end-of-epoch transactions through consensus.
        let assigned_versions = epoch_store.assign_shared_object_versions_idempotent(
            self.get_object_cache_reader().as_ref(),
            std::iter::once(&Schedulable::Transaction(&executable_tx)),
        )?;

        assert_eq!(assigned_versions.0.len(), 1);
        let assigned_versions = assigned_versions.0.into_iter().next().unwrap().1;

        info!("Assigned versions for advance epoch: {:?}", assigned_versions);

        let input_objects = self.read_objects_for_execution(
            &tx_lock,
            &executable_tx,
            assigned_versions,
            epoch_store,
        )?;

        info!("Input objects for advance epoch: {:?}", input_objects);

        let (transaction_outputs, _execution_error_opt) = self
            .execute_certificate(&execution_guard, &executable_tx, input_objects, None, epoch_store)
            .unwrap();
        let system_obj = get_system_state(&transaction_outputs.written)
            .expect("change epoch tx must write to system object");

        let effects = transaction_outputs.effects;
        // We must write tx and effects to the state sync tables so that state sync is able to
        // deliver to the transaction to CheckpointExecutor after it is included in a certified
        // checkpoint.
        self.get_state_sync_store().insert_transaction_and_effects(&tx, &effects);

        // info!(
        //     "Effects summary of the change epoch transaction: {:?}",
        //     effects.summary_for_debug()
        // );

        // Allow tests to detect unexpected safe mode entry (e.g. from race conditions).
        // The test_advance_epoch_tx_race test registers a panic on this failpoint to verify
        // that the is_tx_already_executed guard prevents double-execution.
        if system_obj.safe_mode() {
            utils::fail_point!("checkpoint_builder_advance_epoch_is_safe_mode");
        }

        // With safe mode, the change epoch transaction should always succeed.
        // If it somehow fails even with safe mode, log an error rather than
        // crashing all validators simultaneously.
        if !effects.status().is_ok() {
            tracing::error!(
                "ChangeEpoch transaction failed with status: {:?}. \
                 This should not happen — safe mode should have caught this.",
                effects.status()
            );
        }
        Ok((system_obj, effects))
    }

    #[instrument(level = "error", skip_all)]
    async fn reopen_epoch_db(
        &self,
        cur_epoch_store: &AuthorityPerEpochStore,
        new_committee: Committee,
        epoch_start_configuration: EpochStartConfiguration,
        epoch_last_checkpoint: CheckpointSequenceNumber,
    ) -> SomaResult<Arc<AuthorityPerEpochStore>> {
        let new_epoch = new_committee.epoch;
        info!(new_epoch = ?new_epoch, "re-opening AuthorityEpochTables for new epoch");
        assert_eq!(epoch_start_configuration.epoch_start_state().epoch(), new_committee.epoch);

        utils::fail_point!("before-open-new-epoch-store");

        let new_epoch_store = cur_epoch_store.new_at_next_epoch(
            self.name,
            new_committee,
            epoch_start_configuration,
            epoch_last_checkpoint,
        )?;
        self.epoch_store.store(new_epoch_store.clone());
        Ok(new_epoch_store)
    }

    #[cfg(test)]
    pub(crate) fn iter_live_object_set_for_testing(
        &self,
    ) -> impl Iterator<Item = types::object::LiveObject> + '_ {
        self.get_global_state_hash_store().iter_cached_live_object_set_for_testing(false)
    }

    #[cfg(test)]
    pub(crate) fn shutdown_execution_for_test(&self) {
        self.tx_execution_shutdown.lock().take().unwrap().send(()).unwrap();
    }

    /// NOTE: this function is only to be used for fuzzing and testing. Never use in prod
    pub async fn insert_objects_unsafe_for_testing_only(&self, objects: &[Object]) -> SomaResult {
        self.get_reconfig_api().bulk_insert_genesis_objects(objects);

        self.get_reconfig_api()
            .clear_state_end_of_epoch(&self.execution_lock_for_reconfiguration().await);
        Ok(())
    }
}

pub type ExecutionLockReadGuard<'a> = RwLockReadGuard<'a, EpochId>;
pub type ExecutionLockWriteGuard<'a> = RwLockWriteGuard<'a, EpochId>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ObjDumpFormat {
    pub id: ObjectID,
    pub version: Version,
    pub digest: ObjectDigest,
    pub object: Object,
}

impl ObjDumpFormat {
    fn new(object: Object) -> Self {
        let oref = object.compute_object_reference();
        Self { id: oref.0, version: oref.1, digest: oref.2, object }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NodeStateDump {
    pub tx_digest: TransactionDigest,
    pub sender_signed_data: SenderSignedData,
    pub executed_epoch: u64,
    // pub reference_gas_price: u64,
    // pub protocol_version: u64,
    pub epoch_start_timestamp_ms: u64,
    pub computed_effects: TransactionEffects,
    pub expected_effects_digest: TransactionEffectsDigest,
    pub shared_objects: Vec<ObjDumpFormat>,
    pub modified_at_versions: Vec<ObjDumpFormat>,
    pub input_objects: Vec<ObjDumpFormat>,
}

impl NodeStateDump {
    pub fn new(
        tx_digest: &TransactionDigest,
        effects: &TransactionEffects,
        expected_effects_digest: TransactionEffectsDigest,
        object_store: &dyn ObjectStore,
        epoch_store: &Arc<AuthorityPerEpochStore>,
        inner_temporary_store: &InnerTemporaryStore,
        certificate: &VerifiedExecutableTransaction,
    ) -> SomaResult<Self> {
        // Epoch info
        let executed_epoch = epoch_store.epoch();
        // let reference_gas_price = epoch_store.reference_gas_price();
        let epoch_start_config = epoch_store.epoch_start_config();
        // let protocol_version = epoch_store.protocol_version().as_u64();
        let epoch_start_timestamp_ms = epoch_start_config.epoch_data().epoch_start_timestamp();

        // Record all the shared objects
        let mut shared_objects = Vec::new();
        for kind in effects.input_shared_objects() {
            match kind {
                InputSharedObject::Mutate(obj_ref) | InputSharedObject::ReadOnly(obj_ref) => {
                    if let Some(w) = object_store.get_object_by_key(&obj_ref.0, obj_ref.1) {
                        shared_objects.push(ObjDumpFormat::new(w))
                    }
                }
                InputSharedObject::ReadDeleted(..)
                | InputSharedObject::MutateDeleted(..)
                | InputSharedObject::Cancelled(..) => (), // TODO: consider record congested objects.
            }
        }

        // Record all modified objects
        let mut modified_at_versions = Vec::new();
        for (id, ver) in effects.modified_at_versions() {
            if let Some(w) = object_store.get_object_by_key(&id, ver) {
                modified_at_versions.push(ObjDumpFormat::new(w))
            }
        }

        // All other input objects should already be in `inner_temporary_store.objects`

        Ok(Self {
            tx_digest: *tx_digest,
            executed_epoch,
            // reference_gas_price,
            epoch_start_timestamp_ms,
            // protocol_version,
            shared_objects,
            modified_at_versions,
            sender_signed_data: certificate.clone().into_message(),
            input_objects: inner_temporary_store
                .input_objects
                .values()
                .map(|o| ObjDumpFormat::new(o.clone()))
                .collect(),
            computed_effects: effects.clone(),
            expected_effects_digest,
        })
    }

    pub fn all_objects(&self) -> Vec<ObjDumpFormat> {
        let mut objects = Vec::new();

        objects.extend(self.shared_objects.clone());

        objects.extend(self.modified_at_versions.clone());

        objects.extend(self.input_objects.clone());
        objects
    }

    pub fn write_to_file(&self, path: &Path) -> Result<PathBuf, anyhow::Error> {
        let file_name =
            format!("{}_{}_NODE_DUMP.json", self.tx_digest, AuthorityState::unixtime_now_ms());
        let mut path = path.to_path_buf();
        path.push(&file_name);
        let mut file = File::create(path.clone())?;
        file.write_all(serde_json::to_string_pretty(self)?.as_bytes())?;
        Ok(path)
    }

    pub fn read_from_file(path: &PathBuf) -> Result<Self, anyhow::Error> {
        let file = File::open(path)?;
        serde_json::from_reader(file).map_err(|e| anyhow::anyhow!(e))
    }
}
