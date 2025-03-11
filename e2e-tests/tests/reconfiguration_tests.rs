use std::{
    collections::HashSet,
    fmt::Write as FmtWrite,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
    time::Duration,
};

use futures::future::join_all;
use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use test_cluster::{
    config::genesis_config::{ValidatorGenesisConfig, ValidatorGenesisConfigBuilder},
    TestCluster, TestClusterBuilder,
};
use tokio::time::sleep;
use tracing::{field::{Field, Visit}, Event, info, level_filters::LevelFilter, Metadata, Subscriber};
use tracing_subscriber::{
    fmt, 
    layer::{Context, Layer, SubscriberExt}, 
    registry::LookupSpan, 
    EnvFilter, Layer as _, Registry,
};
use types::{
    base::SomaAddress,
    crypto::{KeypairTraits, PublicKey},
    system_state::SystemStateTrait,
    transaction::{
        AddValidatorArgs, RemoveValidatorArgs, StateTransaction, StateTransactionKind, Transaction,
        TransactionData, TransactionKind,
    },
};

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{ RwLock};
use regex::Regex;
use tracing::span;


const LOG_DIR: &str = "logs";
static INIT: Once = Once::new();

/// A visitor that captures all field values as strings
struct FieldVisitor {
    values: HashMap<String, String>,
}

impl FieldVisitor {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.values.insert(field.name().to_string(), format!("{:?}", value));
    }
    
    fn record_str(&mut self, field: &Field, value: &str) {
        self.values.insert(field.name().to_string(), value.to_string());
    }
    
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.values.insert(field.name().to_string(), value.to_string());
    }
    
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.values.insert(field.name().to_string(), value.to_string());
    }
    
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.values.insert(field.name().to_string(), value.to_string());
    }
    
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.values.insert(field.name().to_string(), value.to_string());
    }
}

/// A thread-safe container for node files with fine-grained locking
struct NodeFiles {
    files: RwLock<HashMap<u32, Arc<Mutex<File>>>>,
    log_dir: PathBuf,
}

impl NodeFiles {
    fn new(log_dir: PathBuf) -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            log_dir,
        }
    }
    
    /// Get or create a file for the given node ID
    fn get_or_create(&self, node_id: u32) -> Option<Arc<Mutex<File>>> {
        // First try to get the file with a read lock
        if let Ok(files) = self.files.read() {
            if let Some(file) = files.get(&node_id) {
                return Some(file.clone());
            }
        }
        
        // If not found, acquire a write lock and create the file
        if let Ok(mut files) = self.files.write() {
            // Check again in case another thread created it while we were waiting
            if let Some(file) = files.get(&node_id) {
                return Some(file.clone());
            }
            
            // Create the file
            let file_path = self.log_dir.join(format!("node_{}.log", node_id));
            match File::create(&file_path) {
                Ok(file) => {
                    let file_mutex = Arc::new(Mutex::new(file));
                    files.insert(node_id, file_mutex.clone());
                    Some(file_mutex)
                },
                Err(_) => None
            }
        } else {
            None
        }
    }
}

/// A simple logging layer with fine-grained locking for node files
struct FineGrainedRoutingLayer {
    log_dir: PathBuf,
    combined_file: Arc<Mutex<File>>,
    // transaction_file: Arc<Mutex<File>>,
    // consensus_file: Arc<Mutex<File>>,
    node_files: NodeFiles,
    span_values: Arc<RwLock<HashMap<span::Id, HashMap<String, String>>>>,
    node_id_regex: Regex,
}

impl FineGrainedRoutingLayer {
    fn new(log_dir: &Path) -> Self {
        // Ensure log directory exists
        fs::create_dir_all(log_dir).expect("Failed to create logs directory");
        
        // Create combined log file
        let combined_file = File::create(log_dir.join("combined.log"))
            .expect("Failed to create combined log file");
        
        // // Create transaction log file
        // let transaction_file = File::create(log_dir.join("transactions.log"))
        //     .expect("Failed to create transaction log file");
            
        // // Create consensus log file
        // let consensus_file = File::create(log_dir.join("consensus.log"))
        //     .expect("Failed to create consensus log file");
            
        // Regex to extract node ID from log lines
        let node_id_regex = Regex::new(r"node\{(?:[^{}]*\s)?id=(\d+)(?:\s[^{}]*)?\}").unwrap();
        
        FineGrainedRoutingLayer {
            log_dir: log_dir.to_path_buf(),
            combined_file: Arc::new(Mutex::new(combined_file)),
            // transaction_file: Arc::new(Mutex::new(transaction_file)),
            // consensus_file: Arc::new(Mutex::new(consensus_file)),
            node_files: NodeFiles::new(log_dir.to_path_buf()),
            span_values: Arc::new(RwLock::new(HashMap::new())),
            node_id_regex,
        }
    }
    
    /// Format a span's fields as a string "{field1=value1 field2=value2}"
    fn format_span_fields(&self, values: &HashMap<String, String>) -> String {
        if values.is_empty() {
            return String::new();
        }
        
        let mut result = String::new();
        result.push('{');
        
        let mut first = true;
        for (key, value) in values {
            if !first {
                result.push(' ');
            }
            first = false;
            
            // Format the field
            if value.starts_with('"') && value.ends_with('"') {
                // Already quoted string
                write!(result, "{}={}", key, value).unwrap();
            } else if value.parse::<f64>().is_ok() || value.parse::<i64>().is_ok() || 
                     value == "true" || value == "false" {
                // Numeric or boolean value
                write!(result, "{}={}", key, value).unwrap();
            } else {
                // String value that needs quotes
                write!(result, "{}=\"{}\"", key, value.replace('"', "\\\"")).unwrap();
            }
        }
        
        result.push('}');
        result
    }
}

impl<S> Layer<S> for FineGrainedRoutingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    /// Capture values when a new span is created
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::new();
        attrs.record(&mut visitor);
        
        if let Ok(mut cache) = self.span_values.write() {
            cache.insert(id.clone(), visitor.values);
        }
    }
    
    /// Capture values when a span is recorded to
    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::new();
        values.record(&mut visitor);
        
        if let Ok(mut cache) = self.span_values.write() {
            if let Some(span_values) = cache.get_mut(id) {
                span_values.extend(visitor.values);
            }
        }
    }
    
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // Extract event fields
        let mut event_visitor = FieldVisitor::new();
        event.record(&mut event_visitor);
        let event_fields = event_visitor.values;
        
        // Get basic event information
        let metadata = event.metadata();
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ");
        let level = metadata.level();
        
        // Get message if present
        let message = event_fields.get("message")
            .cloned()
            .unwrap_or_default();
        
        // Build span context string
        let mut span_str = String::new();
        
        // Process spans in the event's scope (from root to leaf)
        if let Some(scope) = ctx.event_scope(event) {
            let spans = scope.from_root();
            
            for span in spans {
                // Get span name
                let span_name = span.name();
                
                // Get cached values for this span
                let span_fields = if let Ok(cache) = self.span_values.read() {
                    cache.get(&span.id()).cloned().unwrap_or_default()
                } else {
                    HashMap::new()
                };
                
                // Format this span
                if !span_str.is_empty() {
                    span_str.push_str(": ");
                }
                
                // Add span name
                span_str.push_str(span_name);
                
                // Format and add span fields if present
                let fields_str = self.format_span_fields(&span_fields);
                if !fields_str.is_empty() {
                    span_str.push_str(&fields_str);
                }
            }
            
            // Add final colon if we have spans
            if !span_str.is_empty() {
                span_str.push_str(": ");
            }
        }
        
        // Format the complete log line
        let log_line = format!("{} {} {}{}: {}", 
            timestamp, 
            level, 
            span_str, 
            metadata.target(), 
            message
        );
        
        // ----- ROUTING LOGIC -----
        
        // Always write to combined log
        if let Ok(mut file) = self.combined_file.lock() {
            let _ = writeln!(file, "{}", log_line);
        }
        
        // Check if it's transaction related and write to transaction log
        // let is_transaction = metadata.target().contains("transaction") || 
        //                      span_str.contains("transaction") ||
        //                      message.contains("transaction") ||
        //                      message.contains("tx_digest");
        // if is_transaction {
        //     if let Ok(mut file) = self.transaction_file.lock() {
        //         let _ = writeln!(file, "{}", log_line);
        //     }
        // }
        
        // Check if it's consensus related and write to consensus log
        // let is_consensus = metadata.target().contains("consensus") || 
        //                    span_str.contains("consensus") ||
        //                    message.contains("consensus");
        // if is_consensus {
        //     if let Ok(mut file) = self.consensus_file.lock() {
        //         let _ = writeln!(file, "{}", log_line);
        //     }
        // }
        
        // Use regex to extract node ID from the formatted log line
        if let Some(captures) = self.node_id_regex.captures(&log_line) {
            if let Some(id_match) = captures.get(1) {
                if let Ok(node_id) = id_match.as_str().parse::<u32>() {
                    // Get (or create) the mutex for this node's file
                    if let Some(file_mutex) = self.node_files.get_or_create(node_id) {
                        // Lock only this specific file
                        if let Ok(mut file) = file_mutex.lock() {
                            let _ = writeln!(file, "{}", log_line);
                        }
                    }
                }
            }
        }
    }
    
    // Clean up when spans are closed
    fn on_close(&self, id: span::Id, _ctx: Context<'_, S>) {
        // Remove cached values when spans are closed
        if let Ok(mut cache) = self.span_values.write() {
            cache.remove(&id);
        }
    }
}

/// Enhanced tracing initialization with improved organization
// fn init_tracing() {
//     INIT.call_once(|| {
//         // Create a basic subscriber that writes to a single file
//         fs::create_dir_all(LOG_DIR).expect("Failed to create logs directory");
//         let file = File::create(format!("{}/test.log", LOG_DIR)).expect("Failed to create log file");

//         let env_filter = EnvFilter::try_from_default_env()
//             .unwrap_or_else(|_| EnvFilter::new("debug"));

//         let subscriber = fmt::Subscriber::builder()
//             .with_max_level(LevelFilter::DEBUG)
//             .with_writer(file)
//             .with_env_filter(env_filter)
//             .with_ansi(false)
//             .with_thread_ids(true)
//             .with_file(true)
//             .with_line_number(true)
//             .finish();

//         tracing::subscriber::set_global_default(subscriber)
//             .expect("setting default subscriber failed");
//     });
// }

/// Initializes tracing with node IDs for node-specific file logging
fn init_tracing() {
    INIT.call_once(|| {
        let layer = FineGrainedRoutingLayer::new(Path::new(LOG_DIR));
        let subscriber = Registry::default().with(layer).with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("debug"))
        );
        
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global default subscriber");
    });
}

#[msim::sim_test]
async fn advance_epoch_tx_test() {
    let _ = tracing_subscriber::fmt::try_init();
    let test_cluster = TestClusterBuilder::new().build().await;
    let states = test_cluster
        .swarm
        .validator_node_handles()
        .into_iter()
        .map(|handle| handle.with(|node| node.state()))
        .collect::<Vec<_>>();
    let tasks: Vec<_> = states
        .iter()
        .map(|state| async {
            let system_state = state
                .create_and_execute_advance_epoch_tx(&state.epoch_store_for_testing(), 1000)
                .await
                .unwrap();
            system_state
        })
        .collect();
    let results: HashSet<_> = join_all(tasks)
        .await
        .into_iter()
        .map(|(state, _)| state.epoch())
        .collect();
    // Check that all validators have the same result.
    assert_eq!(results.len(), 1);
}

#[msim::sim_test]
async fn basic_reconfig_end_to_end_test() {
    let _ = tracing_subscriber::fmt::try_init();
    // TODO remove this sleep when this test passes consistently
    sleep(Duration::from_secs(1)).await;
    let test_cluster = TestClusterBuilder::new().build().await;
    test_cluster.trigger_reconfiguration().await;
}

#[msim::sim_test]
async fn test_state_sync() {
    init_tracing();

    let mut test_cluster = TestClusterBuilder::new().build().await;

    // Make sure the validators are quiescent before bringing up the node.
    sleep(Duration::from_millis(10000)).await;

    // Start a new fullnode that is not on the write path
    let fullnode = test_cluster.spawn_new_fullnode().await.soma_node;

    sleep(Duration::from_millis(30000)).await;
}

#[msim::sim_test]
async fn test_reconfig_with_committee_change_basic() {
    init_tracing();
    // This test exercise the full flow of a validator joining the network, catch up and then leave.

    let new_validator = ValidatorGenesisConfigBuilder::new().build(&mut OsRng);
    let address = (&new_validator.account_key_pair.public()).into();
    let mut test_cluster = TestClusterBuilder::new()
        .with_validator_candidates([address])
        .build()
        .await;

    execute_add_validator_transactions(&test_cluster, &new_validator).await;

    test_cluster.trigger_reconfiguration().await;

    // Check that a new validator has joined the committee.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            5
        );
    });
    let new_validator_handle = test_cluster.spawn_new_validator(new_validator).await;
    test_cluster.wait_for_epoch_all_nodes(1).await;

    new_validator_handle.with(|node| {
        assert!(node
            .state()
            .is_validator(&node.state().epoch_store_for_testing()));
    });

    execute_remove_validator_tx(&test_cluster, &new_validator_handle).await;
    test_cluster.trigger_reconfiguration().await;

    test_cluster.fullnode_handle.soma_node.with(|node| {
        assert_eq!(
            node.state()
                .epoch_store_for_testing()
                .committee()
                .num_members(),
            4
        );
    });
}

// This test just starts up a cluster that reconfigures itself under 0 load.
#[msim::sim_test]
async fn test_passive_reconfig_normal() {
    do_test_passive_reconfig().await;
}

#[msim::sim_test(check_determinism)]
async fn test_passive_reconfig_determinism() {
    do_test_passive_reconfig().await;
}

async fn do_test_passive_reconfig() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(1000)
        .build()
        .await;

    let target_epoch: u64 = std::env::var("RECONFIG_TARGET_EPOCH")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(4);

    test_cluster.wait_for_epoch(Some(target_epoch)).await;
}

#[msim::sim_test]
async fn test_reconfig_with_committee_change_stress_normal() {
    do_test_reconfig_with_committee_change_stress().await;
}

#[msim::sim_test(check_determinism)]
async fn test_reconfig_with_committee_change_stress_determinism() {
    do_test_reconfig_with_committee_change_stress().await;
}

async fn do_test_reconfig_with_committee_change_stress() {
    init_tracing();

    let mut candidates = (0..6)
        .map(|_| ValidatorGenesisConfigBuilder::new().build(&mut OsRng))
        .collect::<Vec<_>>();
    let addresses = candidates
        .iter()
        .map(|c| (&c.account_key_pair.public()).into())
        .collect::<Vec<SomaAddress>>();
    let mut test_cluster = TestClusterBuilder::new()
        .with_num_validators(7)
        .with_validator_candidates(addresses)
        // .with_num_unpruned_validators(2)
        .build()
        .await;

    let mut cur_epoch = 0;

    while let Some(v1) = candidates.pop() {
        let v2 = candidates.pop().unwrap();
        execute_add_validator_transactions(&test_cluster, &v1).await;
        execute_add_validator_transactions(&test_cluster, &v2).await;
        let mut removed_validators = vec![];
        for v in test_cluster.swarm.active_validators().take(2) {
            let h = v.get_node_handle().unwrap();
            removed_validators.push(h.state().name);
            execute_remove_validator_tx(&test_cluster, &h).await;
        }
        let handle1 = test_cluster.spawn_new_validator(v1).await;
        let handle2 = test_cluster.spawn_new_validator(v2).await;

        tokio::join!(
            test_cluster.wait_for_epoch_on_node(&handle1, Some(cur_epoch), Duration::from_secs(60)),
            test_cluster.wait_for_epoch_on_node(&handle2, Some(cur_epoch), Duration::from_secs(60))
        );

        test_cluster.trigger_reconfiguration().await;
        let committee = test_cluster
            .fullnode_handle
            .soma_node
            .with(|node| node.state().epoch_store_for_testing().committee().clone());
        cur_epoch = committee.epoch();
        assert_eq!(committee.num_members(), 7);
        assert!(committee.authority_exists(&handle1.state().name));
        assert!(committee.authority_exists(&handle2.state().name));
        removed_validators
            .iter()
            .all(|v| !committee.authority_exists(v));
    }
}

async fn execute_remove_validator_tx(test_cluster: &TestCluster, handle: &SomaNodeHandle) {
    let address = handle.with(|node| node.get_config().soma_address());

    let tx = handle.with(|node| {
        Transaction::from_data_and_signer(
            TransactionData::new(
                TransactionKind::StateTransaction(StateTransaction {
                    kind: StateTransactionKind::RemoveValidator(RemoveValidatorArgs {
                        pubkey_bytes: bcs::to_bytes(
                            &node.get_config().account_key_pair.keypair().public(),
                        )
                        .unwrap(),
                    }),
                    sender: (&node.get_config().account_key_pair.keypair().public()).into(),
                }),
                (&node.get_config().account_key_pair.keypair().public()).into(),
            ),
            vec![node.get_config().account_key_pair.keypair()],
        )
    });

    info!(?tx, "Executing remove validator tx");

    test_cluster.execute_transaction(tx).await;
}

/// Execute a sequence of transactions to add a validator, including adding candidate, adding stake
/// and activate the validator.
/// It does not however trigger reconfiguration yet.
async fn execute_add_validator_transactions(
    test_cluster: &TestCluster,
    new_validator: &ValidatorGenesisConfig,
) {
    let pending_active_count = test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node.state().get_system_state_object_for_testing();
        system_state.validators.pending_active_validators.len()
    });

    let tx = Transaction::from_data_and_signer(
        TransactionData::new(
            TransactionKind::StateTransaction(StateTransaction {
                kind: StateTransactionKind::AddValidator(AddValidatorArgs {
                    pubkey_bytes: bcs::to_bytes(&new_validator.key_pair.public()).unwrap(),
                    network_pubkey_bytes: bcs::to_bytes(&new_validator.network_key_pair.public())
                        .unwrap(),
                    worker_pubkey_bytes: bcs::to_bytes(&new_validator.worker_key_pair.public())
                        .unwrap(),
                    net_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                    p2p_address: bcs::to_bytes(&new_validator.consensus_address).unwrap(),
                    primary_address: bcs::to_bytes(&new_validator.network_address).unwrap(),
                }),
                sender: (&new_validator.account_key_pair.public()).into(),
            }),
            (&new_validator.account_key_pair.public()).into(),
        ),
        vec![&new_validator.account_key_pair],
    );
    test_cluster.execute_transaction(tx).await;

    // Check that we can get the pending validator from 0x5.
    test_cluster.fullnode_handle.soma_node.with(|node| {
        let system_state = node.state().get_system_state_object_for_testing();
        let pending_active_validators = system_state.validators.pending_active_validators;
        assert_eq!(pending_active_validators.len(), pending_active_count + 1);
        assert_eq!(
            pending_active_validators[pending_active_validators.len() - 1]
                .metadata
                .soma_address,
            (&new_validator.account_key_pair.public()).into()
        );
    });
}

// async fn test_inactive_validator_pool_read()
// async fn test_validator_candidate_pool_read()
// async fn test_reconfig_with_failing_validator(
// async fn test_create_advance_epoch_tx_race()
// async fn test_expired_locks()
// async fn do_test_passive_reconfig()
