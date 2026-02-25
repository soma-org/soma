// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{path::PathBuf, sync::Arc};

use crate::{
    authority::{AuthorityState, ExecutionEnv},
    authority_per_epoch_store::AuthorityPerEpochStore,
    authority_store::AuthorityStore,
    authority_store_pruner::{ObjectsCompactionFilter, PrunerWatermarks},
    authority_store_tables::{
        AuthorityPerpetualTables, AuthorityPerpetualTablesOptions, AuthorityPrunerTables,
    },
    backpressure_manager::BackpressureManager,
    cache::build_execution_cache,
    checkpoints::CheckpointStore,
    execution_scheduler::SchedulingSource,
    rpc_index::RpcIndexStore,
    start_epoch::EpochStartConfiguration,
};
use fastcrypto::traits::KeyPair;
use protocol_config::{Chain, ProtocolConfig};
use store::nondeterministic;
use types::{
    base::AuthorityName,
    config::{
        certificate_deny_config::CertificateDenyConfig,
        genesis_config::AccountConfig,
        network_config::{ConfigBuilder, NetworkConfig},
        node_config::{
            AuthorityStorePruningConfig, ExecutionCacheConfig, ExpensiveSafetyCheckConfig,
        },
        transaction_deny_config::TransactionDenyConfig,
    },
    crypto::AuthorityKeyPair,
    digests::ChainIdentifier,
    genesis::Genesis,
    object::{Object, ObjectID},
    storage::committee_store::CommitteeStore,
    supported_protocol_versions::SupportedProtocolVersions,
    system_state::SystemStateTrait,
    transaction::{VerifiedExecutableTransaction, VerifiedTransaction},
};
#[derive(Default, Clone)]
pub struct TestAuthorityBuilder<'a> {
    store_base_path: Option<PathBuf>,
    store: Option<Arc<AuthorityStore>>,
    transaction_deny_config: Option<TransactionDenyConfig>,
    certificate_deny_config: Option<CertificateDenyConfig>,
    protocol_config: Option<ProtocolConfig>,
    reference_gas_price: Option<u64>,
    node_keypair: Option<&'a AuthorityKeyPair>,
    genesis: Option<&'a Genesis>,
    starting_objects: Option<&'a [Object]>,
    expensive_safety_checks: Option<ExpensiveSafetyCheckConfig>,
    disable_indexer: bool,
    accounts: Vec<AccountConfig>,
    /// By default, we don't insert the genesis checkpoint, which isn't needed by most tests.
    insert_genesis_checkpoint: bool,
    // authority_overload_config: Option<AuthorityOverloadConfig>,
    cache_config: Option<ExecutionCacheConfig>,
    chain_override: Option<Chain>,
}

impl<'a> TestAuthorityBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_store_base_path(mut self, path: PathBuf) -> Self {
        assert!(self.store_base_path.replace(path).is_none());
        self
    }

    pub fn with_starting_objects(mut self, objects: &'a [Object]) -> Self {
        assert!(self.starting_objects.replace(objects).is_none());
        self
    }

    pub fn with_store(mut self, store: Arc<AuthorityStore>) -> Self {
        assert!(self.store.replace(store).is_none());
        self
    }

    pub fn with_transaction_deny_config(mut self, config: TransactionDenyConfig) -> Self {
        assert!(self.transaction_deny_config.replace(config).is_none());
        self
    }

    pub fn with_certificate_deny_config(mut self, config: CertificateDenyConfig) -> Self {
        assert!(self.certificate_deny_config.replace(config).is_none());
        self
    }

    pub fn with_protocol_config(mut self, config: ProtocolConfig) -> Self {
        assert!(self.protocol_config.replace(config).is_none());
        self
    }

    pub fn with_reference_gas_price(mut self, reference_gas_price: u64) -> Self {
        // If genesis is already set then setting rgp is meaningless since it will be overwritten.
        assert!(self.genesis.is_none());
        assert!(self.reference_gas_price.replace(reference_gas_price).is_none());
        self
    }

    pub fn with_genesis_and_keypair(
        mut self,
        genesis: &'a Genesis,
        keypair: &'a AuthorityKeyPair,
    ) -> Self {
        assert!(self.genesis.replace(genesis).is_none());
        assert!(self.node_keypair.replace(keypair).is_none());
        self
    }

    pub fn with_keypair(mut self, keypair: &'a AuthorityKeyPair) -> Self {
        assert!(self.node_keypair.replace(keypair).is_none());
        self
    }

    /// When providing a network config, we will use the \node_idx validator's
    /// key as the keypair for the new node.
    pub fn with_network_config(self, config: &'a NetworkConfig, node_idx: usize) -> Self {
        self.with_genesis_and_keypair(
            &config.genesis,
            config.validator_configs()[node_idx].protocol_key_pair(),
        )
    }

    pub fn disable_indexer(mut self) -> Self {
        self.disable_indexer = true;
        self
    }

    pub fn insert_genesis_checkpoint(mut self) -> Self {
        self.insert_genesis_checkpoint = true;
        self
    }

    pub fn with_expensive_safety_checks(mut self, config: ExpensiveSafetyCheckConfig) -> Self {
        assert!(self.expensive_safety_checks.replace(config).is_none());
        self
    }

    pub fn with_accounts(mut self, accounts: Vec<AccountConfig>) -> Self {
        self.accounts = accounts;
        self
    }

    pub fn with_cache_config(mut self, config: ExecutionCacheConfig) -> Self {
        self.cache_config = Some(config);
        self
    }

    pub fn with_chain_override(mut self, chain: Chain) -> Self {
        self.chain_override = Some(chain);
        self
    }

    pub async fn build(self) -> Arc<AuthorityState> {
        // `_guard` must be declared here so it is not dropped before
        // `AuthorityPerEpochStore::new` is called
        let protocol_config = self.protocol_config.clone();
        // let _guard = protocol_config
        //     .map(|config| ProtocolConfig::apply_overrides_for_testing(move |_, _| config.clone()));

        let mut local_network_config_builder =
            ConfigBuilder::new_with_temp_dir().with_accounts(self.accounts);
        // if let Some(protocol_config) = &self.protocol_config {
        //     local_network_config_builder =
        //         local_network_config_builder.with_protocol_version(protocol_config.version);
        // }
        let local_network_config = local_network_config_builder.build();
        let genesis = &self.genesis.unwrap_or(&local_network_config.genesis);
        let genesis_committee = genesis.committee().unwrap();
        let path = self.store_base_path.unwrap_or_else(|| {
            let dir = std::env::temp_dir();
            let store_base_path =
                dir.join(format!("DB_{:?}", nondeterministic!(ObjectID::random())));
            std::fs::create_dir(&store_base_path).unwrap();
            store_base_path
        });
        let mut config = local_network_config.validator_configs()[0].clone();

        let mut pruner_db = None;
        if config.authority_store_pruning_config.enable_compaction_filter {
            pruner_db = Some(Arc::new(AuthorityPrunerTables::open(&path.join("store"))));
        }
        let compaction_filter = pruner_db.clone().map(ObjectsCompactionFilter::new);

        let authority_store = match self.store {
            Some(store) => store,
            None => {
                let perpetual_tables_options =
                    AuthorityPerpetualTablesOptions { compaction_filter, ..Default::default() };
                let perpetual_tables = Arc::new(AuthorityPerpetualTables::open(
                    &path.join("store"),
                    Some(perpetual_tables_options),
                ));
                // unwrap ok - for testing only.
                AuthorityStore::open_with_committee_for_testing(
                    perpetual_tables,
                    &genesis_committee,
                    genesis,
                )
                .await
                .unwrap()
            }
        };
        if let Some(cache_config) = self.cache_config {
            config.execution_cache = cache_config;
        }

        let keypair = if let Some(keypair) = self.node_keypair {
            keypair
        } else {
            config.protocol_key_pair()
        };

        let secret = Arc::pin(keypair.copy());
        let name: AuthorityName = secret.public().into();

        let epoch_start_configuration = EpochStartConfiguration::new(
            genesis.system_object().into_epoch_start_state(),
            *genesis.checkpoint().digest(),
            // &genesis.objects(),
        );
        let expensive_safety_checks = self.expensive_safety_checks.unwrap_or_default();

        let pruner_watermarks = Arc::new(PrunerWatermarks::default());
        let checkpoint_store =
            CheckpointStore::new(&path.join("checkpoints"), pruner_watermarks.clone());
        let backpressure_manager =
            BackpressureManager::new_from_checkpoint_store(&checkpoint_store);

        let cache_traits = build_execution_cache(
            &Default::default(),
            &authority_store,
            backpressure_manager.clone(),
        );

        let chain_id = ChainIdentifier::from(*genesis.checkpoint().digest());
        let chain = match self.chain_override {
            Some(chain) => chain,
            None => chain_id.chain(),
        };

        let epoch_store = AuthorityPerEpochStore::new(
            name,
            Arc::new(genesis_committee.clone()),
            &path.join("store"),
            None,
            epoch_start_configuration,
            (chain_id, chain),
            checkpoint_store.get_highest_executed_checkpoint_seq_number().unwrap().unwrap_or(0),
        )
        .expect("failed to create authority per epoch store");
        let committee_store =
            Arc::new(CommitteeStore::new(path.join("epochs"), &genesis_committee, None));

        if self.insert_genesis_checkpoint {
            checkpoint_store.insert_genesis_checkpoint(
                genesis.checkpoint(),
                genesis.checkpoint_contents().clone(),
                &epoch_store,
            );
        }

        let rpc_index = if self.disable_indexer {
            None
        } else {
            Some(Arc::new(RpcIndexStore::new(&path, &authority_store, &checkpoint_store).await))
        };

        let transaction_deny_config = self.transaction_deny_config.unwrap_or_default();
        let certificate_deny_config = self.certificate_deny_config.unwrap_or_default();

        let mut pruning_config = AuthorityStorePruningConfig::default();

        config.transaction_deny_config = transaction_deny_config;
        config.certificate_deny_config = certificate_deny_config;

        config.authority_store_pruning_config = pruning_config;

        let chain_identifier = ChainIdentifier::from(*genesis.checkpoint().digest());

        let state = AuthorityState::new(
            name,
            secret,
            SupportedProtocolVersions::SYSTEM_DEFAULT,
            authority_store,
            cache_traits,
            epoch_store.clone(),
            committee_store,
            rpc_index,
            checkpoint_store,
            genesis.objects(),
            config.clone(),
            None,
            chain_identifier,
            pruner_db,
            Arc::new(PrunerWatermarks::default()),
        )
        .await;

        // For any type of local testing that does not actually spawn a node, the checkpoint executor
        // won't be started, which means we won't actually execute the genesis transaction. In that case,
        // the genesis objects (e.g. all the genesis test coins) won't be accessible. Executing it
        // explicitly makes sure all genesis objects are ready for use.
        state
            .try_execute_immediately(
                &VerifiedExecutableTransaction::new_from_checkpoint(
                    VerifiedTransaction::new_unchecked(genesis.transaction().clone()),
                    genesis.epoch(),
                    genesis.checkpoint().sequence_number,
                ),
                ExecutionEnv::new().with_scheduling_source(SchedulingSource::NonFastPath),
                &state.epoch_store_for_testing(),
            )
            .await
            .unwrap();

        let batch = state
            .get_cache_commit()
            .build_db_batch(epoch_store.epoch(), &[*genesis.transaction().digest()]);

        state.get_cache_commit().commit_transaction_outputs(
            epoch_store.epoch(),
            batch,
            &[*genesis.transaction().digest()],
        );

        // We want to insert these objects directly instead of relying on genesis because
        // genesis process would set the previous transaction field for these objects, which would
        // change their object digest. This makes it difficult to write tests that want to use
        // these objects directly.
        // TODO: we should probably have a better way to do this.
        if let Some(starting_objects) = self.starting_objects {
            state.insert_objects_unsafe_for_testing_only(starting_objects).await.unwrap();
        };
        state
    }
}
