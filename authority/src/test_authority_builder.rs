use std::{path::PathBuf, sync::Arc};

use fastcrypto::traits::KeyPair;
use types::{
    base::AuthorityName,
    config::{
        genesis_config::AccountConfig,
        network_config::{ConfigBuilder, NetworkConfig},
    },
    crypto::AuthorityKeyPair,
    genesis::Genesis,
    object::{Object, ObjectID},
    protocol::ProtocolConfig,
    storage::committee_store::CommitteeStore,
    system_state::SystemStateTrait,
    transaction::{VerifiedExecutableTransaction, VerifiedTransaction},
};

use crate::{
    cache::build_execution_cache,
    commit::CommitStore,
    epoch_store::AuthorityPerEpochStore,
    rpc_index::RpcIndexStore,
    start_epoch::EpochStartConfiguration,
    state::AuthorityState,
    state_accumulator::StateAccumulator,
    store::AuthorityStore,
    store_pruner::ObjectsCompactionFilter,
    store_tables::{
        AuthorityPerpetualTables, AuthorityPerpetualTablesOptions, AuthorityPrunerTables,
    },
};

#[derive(Default, Clone)]
pub struct TestAuthorityBuilder<'a> {
    store_base_path: Option<PathBuf>,
    store: Option<Arc<AuthorityStore>>,
    protocol_config: Option<ProtocolConfig>,
    // reference_gas_price: Option<u64>,
    node_keypair: Option<&'a AuthorityKeyPair>,
    genesis: Option<&'a Genesis>,
    starting_objects: Option<&'a [Object]>,
    accounts: Vec<AccountConfig>,
    /// By default, we don't insert the genesis commit, which isn't needed by most tests.
    insert_genesis_commit: bool,
    // cache_config: Option<ExecutionCacheConfig>,
    // chain_override: Option<Chain>,
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

    pub fn with_protocol_config(mut self, config: ProtocolConfig) -> Self {
        assert!(self.protocol_config.replace(config).is_none());
        self
    }

    // pub fn with_reference_gas_price(mut self, reference_gas_price: u64) -> Self {
    //     // If genesis is already set then setting rgp is meaningless since it will be overwritten.
    //     assert!(self.genesis.is_none());
    //     assert!(self
    //         .reference_gas_price
    //         .replace(reference_gas_price)
    //         .is_none());
    //     self
    // }

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

    // pub fn disable_indexer(mut self) -> Self {
    //     self.disable_indexer = true;
    //     self
    // }

    pub fn insert_genesis_commit(mut self) -> Self {
        self.insert_genesis_commit = true;
        self
    }

    pub fn with_accounts(mut self, accounts: Vec<AccountConfig>) -> Self {
        self.accounts = accounts;
        self
    }

    // pub fn with_cache_config(mut self, config: ExecutionCacheConfig) -> Self {
    //     self.cache_config = Some(config);
    //     self
    // }

    // pub fn with_chain_override(mut self, chain: Chain) -> Self {
    //     self.chain_override = Some(chain);
    //     self
    // }

    pub async fn build(self) -> Arc<AuthorityState> {
        let mut local_network_config_builder = ConfigBuilder::new().with_accounts(self.accounts);
        // .with_reference_gas_price(self.reference_gas_price.unwrap_or(500));
        // if let Some(protocol_config) = &self.protocol_config {
        //     local_network_config_builder =
        //         local_network_config_builder.with_protocol_version(protocol_config.version);
        // }
        let local_network_config = local_network_config_builder.build();
        let genesis = &self.genesis.unwrap_or(&local_network_config.genesis);
        let genesis_committee = genesis.committee().unwrap();
        let path = self.store_base_path.unwrap_or_else(|| {
            let dir = std::env::temp_dir();
            let store_base_path = dir.join(format!("DB_{:?}", ObjectID::random()));
            std::fs::create_dir(&store_base_path).unwrap();
            store_base_path
        });
        let mut config = local_network_config.validator_configs()[0].clone();
        let mut pruner_db = None;
        // if config
        //     .authority_store_pruning_config
        //     .enable_compaction_filter
        // {
        //     pruner_db = Some(Arc::new(AuthorityPrunerTables::open(&path.join("store"))));
        // }
        let compaction_filter = pruner_db.clone().map(|db| ObjectsCompactionFilter::new(db));

        let authority_store = match self.store {
            Some(store) => store,
            None => {
                let perpetual_tables_options = AuthorityPerpetualTablesOptions {
                    compaction_filter,
                    ..Default::default()
                };
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
        // if let Some(cache_config) = self.cache_config {
        //     config.execution_cache = cache_config;
        // }

        let keypair = if let Some(keypair) = self.node_keypair {
            keypair
        } else {
            config.protocol_key_pair()
        };

        let secret = Arc::pin(keypair.copy());
        let name: AuthorityName = secret.public().into();
        // `_guard` must be declared here so it is not dropped before
        // `AuthorityPerEpochStore::new` is called
        // let _guard = self
        //     .protocol_config
        //     .map(|config| ProtocolConfig::apply_overrides_for_testing(move |_, _| config.clone()));
        let epoch_start_configuration = EpochStartConfiguration::new(
            genesis.system_object().into_epoch_start_state(),
            // *genesis.commit().digest(),
            // &genesis.objects(),
            // epoch_flags,
        );

        let commit_store = CommitStore::new(&path.join("commits"));

        let cache_traits = build_execution_cache(&authority_store);

        // let chain_id = ChainIdentifier::from(*genesis.commit().digest());
        // let chain = match self.chain_override {
        //     Some(chain) => chain,
        //     None => chain_id.chain(),
        // };

        let epoch_store = AuthorityPerEpochStore::new(
            name,
            Arc::new(genesis_committee.clone()),
            &path.join("store"),
            None,
            epoch_start_configuration,
            0,
        )
        .expect("failed to create authority per epoch store");
        let committee_store = Arc::new(CommitteeStore::new(
            path.join("epochs"),
            &genesis_committee,
            None,
        ));
        let accumulator = Arc::new(StateAccumulator::new(
            cache_traits.accumulator_store.clone(),
        ));

        if self.insert_genesis_commit {
            commit_store.insert_genesis_commit(genesis.commit());
        }

        let rpc_index = Some(Arc::new(
            RpcIndexStore::new(&path, &authority_store, &commit_store).await,
        ));

        // let chain_identifier = ChainIdentifier::from(*genesis.commit().digest());

        let state = AuthorityState::new(
            name,
            secret,
            // SupportedProtocolVersions::SYSTEM_DEFAULT,
            epoch_store.clone(),
            committee_store,
            config.clone(),
            cache_traits,
            accumulator,
            rpc_index,
        )
        .await;

        // For any type of local testing that does not actually spawn a node, the commit executor
        // won't be started, which means we won't actually execute the genesis transaction. In that case,
        // the genesis objects (e.g. all the genesis test coins) won't be accessible. Executing it
        // explicitly makes sure all genesis objects are ready for use.
        state
            .try_execute_immediately(
                &VerifiedExecutableTransaction::new_from_commit(
                    VerifiedTransaction::new_unchecked(genesis.transaction().clone()),
                    genesis.epoch(),
                    genesis.commit().commit_ref.index,
                ),
                None,
                Some(genesis.commit().commit_ref.index),
                &state.epoch_store_for_testing(),
            )
            .await
            .unwrap();

        state
            .get_cache_commit()
            .commit_transaction_outputs(epoch_store.epoch(), &[*genesis.transaction().digest()]);

        // We want to insert these objects directly instead of relying on genesis because
        // genesis process would set the previous transaction field for these objects, which would
        // change their object digest. This makes it difficult to write tests that want to use
        // these objects directly.
        // TODO: we should probably have a better way to do this.
        if let Some(starting_objects) = self.starting_objects {
            state
                .insert_objects_unsafe_for_testing_only(starting_objects)
                .await
                .unwrap();
        };
        state
    }
}
