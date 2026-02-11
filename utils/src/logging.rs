use std::{
    collections::HashMap,
    fmt::Write as FmtWrite,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use regex::Regex;
use std::sync::RwLock;
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
    span,
};
use tracing_subscriber::{
    EnvFilter, Layer, Registry,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
};

const LOG_DIR: &str = "logs";
static INIT: Once = Once::new();

/// A visitor that captures all field values as strings
struct FieldVisitor {
    values: HashMap<String, String>,
}

impl FieldVisitor {
    fn new() -> Self {
        Self { values: HashMap::new() }
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
        Self { files: RwLock::new(HashMap::new()), log_dir }
    }

    /// Get or create a file for the given node ID
    fn get_or_create(&self, node_id: u32) -> Option<Arc<Mutex<File>>> {
        // First try to get the file with a read lock
        if let Ok(files) = self.files.read()
            && let Some(file) = files.get(&node_id)
        {
            return Some(file.clone());
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
                }
                Err(_) => None,
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
        let combined_file =
            File::create(log_dir.join("combined.log")).expect("Failed to create combined log file");

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
            } else if value.parse::<f64>().is_ok()
                || value.parse::<i64>().is_ok()
                || value == "true"
                || value == "false"
            {
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

        if let Ok(mut cache) = self.span_values.write()
            && let Some(span_values) = cache.get_mut(id)
        {
            span_values.extend(visitor.values);
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
        let message = event_fields.get("message").cloned().unwrap_or_default();

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
        let log_line =
            format!("{} {} {}{}: {}", timestamp, level, span_str, metadata.target(), message);

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
        if let Some(captures) = self.node_id_regex.captures(&log_line)
            && let Some(id_match) = captures.get(1)
            && let Ok(node_id) = id_match.as_str().parse::<u32>()
            && let Some(file_mutex) = self.node_files.get_or_create(node_id)
            && let Ok(mut file) = file_mutex.lock()
        {
            let _ = writeln!(file, "{}", log_line);
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

/// Initializes tracing with node IDs for node-specific file logging
pub fn init_tracing() {
    INIT.call_once(|| {
        let layer = FineGrainedRoutingLayer::new(Path::new(LOG_DIR));
        let subscriber = Registry::default()
            .with(layer)
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")));

        // Ignore error if a global default subscriber has already been set
        // (e.g., by a previous test in the same process).
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}
