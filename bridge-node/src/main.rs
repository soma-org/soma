//! `bridge-node` — the off-chain bridge daemon.
//!
//! Loads a `BridgeNodeConfig` from a TOML file, reads the validator's
//! ECDSA bridge key, pulls the live `BridgeCommittee` from a Soma
//! fullnode RPC, and runs all bridge subsystems (Eth syncer, HTTP
//! sig-exchange server, action executor, outbound relayer, watchdog,
//! deposit/withdrawal handlers, WAL). On SIGINT/SIGTERM the spawned
//! task handles are dropped, which cleans up the WAL and lets the
//! process exit.
//!
//! Exit codes:
//!   * 0 — clean shutdown via SIGINT/SIGTERM
//!   * 1 — startup error (config / key / RPC / committee)
//!   * 2 — a spawned subsystem task panicked
//!
//! Operator-facing logs at startup print the bridge pubkey (hex) and
//! HTTP listen address so a sloppy `bridge_key_path` or port collision
//! is obvious immediately.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use fastcrypto::encoding::{Base64, Encoding};
use fastcrypto::secp256k1::Secp256k1KeyPair;
use fastcrypto::traits::{EncodeDecodeBase64, ToFromBytes};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use bridge_node::config::BridgeNodeConfig;
use bridge_node::node::BridgeNode;
use bridge_node::soma_client::SomaBridgeClient;
use types::bridge::{BridgePubkey, SOMA_BRIDGE_CHAIN_ID};

/// CLI entry point for the bridge daemon. The single required arg is
/// `--config <PATH>` pointing at a TOML file matching
/// [`bridge_node::config::BridgeNodeConfig`]. See
/// `bridge-node/configs/base-sepolia.toml.template` for a fully
/// commented example.
#[derive(Parser, Debug)]
#[command(
    name = "bridge-node",
    version,
    about = "Soma <-> Ethereum bridge node daemon.",
    long_about = None,
)]
struct Args {
    /// Path to the bridge node TOML config file. Must match the
    /// `BridgeNodeConfig` schema in `bridge-node/src/config.rs`.
    #[arg(long)]
    config: PathBuf,
}

fn main() -> ExitCode {
    // Initialize tracing first so any subsequent error path is logged.
    // Operators set `RUST_LOG=bridge_node=debug` for verbose logs;
    // default is `info` across the whole tree.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Mirror `bridge-committee-export`: build a current-thread runtime
    // explicitly so the CLI-level exit codes stay readable. The bridge
    // node spawns many subsystem tasks; they need a multi-thread
    // runtime to overlap I/O without HOL-blocking. Use the default
    // worker count (= num_cpus).
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            error!(error = %e, "failed to build tokio runtime");
            return ExitCode::from(1);
        }
    };

    match rt.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(RunError::Startup(msg)) => {
            error!("startup failed: {msg}");
            ExitCode::from(1)
        }
        Err(RunError::TaskPanic(msg)) => {
            error!("subsystem task panicked: {msg}");
            ExitCode::from(2)
        }
    }
}

enum RunError {
    /// Anything that happens before `node.run()` returns its task
    /// handles, or before `tokio::select!` arms get registered. Exit
    /// status 1.
    Startup(String),
    /// One of the spawned subsystem tasks panicked at runtime. Exit
    /// status 2 — distinct from startup so an operator-facing process
    /// supervisor (systemd, runit, k8s) can tell a configuration
    /// problem apart from a live bug.
    TaskPanic(String),
}

async fn run(args: Args) -> Result<(), RunError> {
    // 1. Load + parse the TOML config.
    info!(config_path = %args.config.display(), "loading bridge-node config");
    let config_str = std::fs::read_to_string(&args.config).map_err(|e| {
        RunError::Startup(format!("read config file {}: {e}", args.config.display()))
    })?;
    let config: BridgeNodeConfig = toml::from_str(&config_str).map_err(|e| {
        RunError::Startup(format!("parse {} as BridgeNodeConfig TOML: {e}", args.config.display()))
    })?;
    config.validate().map_err(|e| RunError::Startup(format!("config validation: {e}")))?;

    // 2. Load the bridge keypair. The on-disk format is a base64
    // string (whitespace-trimmed) of either the raw private key bytes
    // (32 bytes for secp256k1) — same as fastcrypto's `EncodeDecodeBase64`
    // convention for keypairs. Fall back to interpreting the file as a
    // raw 32-byte secret if base64 decoding fails, so operator workflows
    // that drop raw bytes (e.g. `openssl rand 32 > bridge.key`) still
    // work.
    let bridge_keypair = load_bridge_keypair(&config.bridge_key_path)?;
    let bridge_pubkey = BridgePubkey::from_keypair(&bridge_keypair);
    let bridge_pubkey_hex = hex::encode(bridge_pubkey.as_bytes());

    info!(
        bridge_pubkey = %bridge_pubkey_hex,
        http_listen_address = %config.http_listen_address,
        bridge_contract = %config.bridge_contract_address,
        soma_rpc = %config.soma_rpc_url,
        wal_path = %config.wal_path.display(),
        "bridge-node starting"
    );
    eprintln!("bridge_pubkey       = 0x{bridge_pubkey_hex}");
    eprintln!("http_listen_address = {}", config.http_listen_address);
    eprintln!("bridge_contract     = {}", config.bridge_contract_address);

    // 3. Pull the live bridge committee from Soma RPC. The
    // `SOMA_BRIDGE_CHAIN_ID` constant is the dev-config default
    // (`SomaCustom`); the field isn't surfaced in `BridgeNodeConfig`
    // yet, so production deployments inherit the constant. Token-transfer
    // record-id derivation + metric labels are the only things that
    // depend on it at this layer.
    let soma_chain_id = SOMA_BRIDGE_CHAIN_ID;
    info!(
        ?soma_chain_id,
        soma_rpc = %config.soma_rpc_url,
        "connecting to Soma RPC and fetching bridge committee"
    );
    let soma_client =
        SomaBridgeClient::new_rpc(&config.soma_rpc_url, soma_chain_id).await.map_err(|e| {
            RunError::Startup(format!("connect to Soma RPC at {}: {e}", config.soma_rpc_url))
        })?;
    let committee = soma_client
        .get_bridge_committee()
        .await
        .map_err(|e| RunError::Startup(format!("fetch bridge committee: {e}")))?;
    info!(members = committee.members.len(), "fetched live bridge committee");

    // Sanity-check membership: a daemon whose pubkey isn't on the
    // committee will still run (and serve sigs that nobody trusts),
    // which is a confusing failure mode. Warn loudly.
    let on_committee = committee.members.contains_key(&bridge_pubkey);
    if !on_committee {
        warn!(
            bridge_pubkey = %bridge_pubkey_hex,
            "this bridge key is NOT in the live BridgeState.bridge_committee — \
             signatures from this node will be ignored by peers. Register the \
             key on-chain (see `soma validator register-bridge-key`) before \
             expecting end-to-end flow."
        );
    } else {
        info!("bridge key is registered in the live committee");
    }

    // We drop the `soma_client` we used for the initial committee fetch
    // — `BridgeNode::run()` constructs its own client internally (so it
    // can wire `Arc<SomaBridgeClient>` through the monitor + executor +
    // watchdog without an extra clone path). Two short-lived RPC
    // connections at startup is fine; the alternative is plumbing a
    // pre-built client through `BridgeNode::new`, which the existing
    // API doesn't support.
    drop(soma_client);

    // 4. Construct the node. The `BridgeNodeConfig` has no Soma-side
    // relayer fields (relayer_address + relayer_keypair would need to
    // be added), so we don't call `.with_relayer()` here. Without it
    // the node runs in **sig-cache-only mode**: it observes events and
    // serves sigs to peers, but no one assembles a quorum cert and
    // submits it on Soma. End-to-end requires adding the relayer
    // fields to `BridgeNodeConfig` and wiring them in.
    //
    // Note: `BridgeNodeConfig.outbound_relayer` is a *different*
    // relayer — it's the Eth-side operator wallet that submits release
    // txs to Ethereum. That one is read inside `BridgeNode::run()`
    // directly from the config and doesn't need wiring at this layer.
    let node = BridgeNode::new(config, bridge_keypair, committee);

    // 5. Spawn all subsystems. `run()` returns the JoinHandles for
    // every spawned task — we wait on them below.
    let handles =
        node.run().await.map_err(|e| RunError::Startup(format!("BridgeNode::run spawn: {e}")))?;
    let handle_count = handles.len();
    info!(handle_count, "bridge-node subsystems spawned; waiting for shutdown signal");

    // 6. Wait for either:
    //   (a) SIGINT / SIGTERM → log + return cleanly; dropping the
    //       JoinHandles aborts the tasks and the WAL Drop impl flushes.
    //   (b) Any spawned task finishes (which for these daemons is
    //       almost always a panic — they're meant to loop forever).
    //
    // `futures::future::select_all` resolves when any future
    // completes, returning the resolved value + the index + the
    // remaining futures. We use it to detect the first task exit.
    let task_watcher = async {
        let (res, idx, _rest) = futures::future::select_all(handles).await;
        // `res: Result<(), JoinError>` — Err means the task panicked or
        // was cancelled. Cancellation shouldn't happen unless we drop
        // the handles (which we don't until shutdown), so any error here
        // is a panic.
        match res {
            Ok(()) => format!(
                "subsystem task #{idx} exited unexpectedly (normal return — bridge tasks loop forever)"
            ),
            Err(join_err) => {
                if join_err.is_panic() {
                    format!("subsystem task #{idx} panicked: {join_err}")
                } else {
                    format!("subsystem task #{idx} cancelled: {join_err}")
                }
            }
        }
    };

    tokio::select! {
        // Watch for ctrl-c. On Unix this triggers on SIGINT. SIGTERM
        // (the systemd / docker stop default) is handled below via a
        // separate signal stream so both produce a clean exit.
        _ = tokio::signal::ctrl_c() => {
            info!("received SIGINT — shutting down");
            Ok(())
        }
        _ = sigterm() => {
            info!("received SIGTERM — shutting down");
            Ok(())
        }
        msg = task_watcher => {
            Err(RunError::TaskPanic(msg))
        }
    }
}

/// Wait for SIGTERM. On non-Unix platforms (build-time only — bridge
/// nodes are Linux-only in production) this future never resolves, so
/// `tokio::select!` falls back to ctrl-c or task exit.
#[cfg(unix)]
async fn sigterm() {
    use tokio::signal::unix::{SignalKind, signal};
    match signal(SignalKind::terminate()) {
        Ok(mut s) => {
            s.recv().await;
        }
        Err(e) => {
            warn!(error = %e, "failed to install SIGTERM handler; only SIGINT will shut down cleanly");
            // Never resolve — fall through to other select arms.
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(not(unix))]
async fn sigterm() {
    std::future::pending::<()>().await;
}

/// Read the bridge keypair from the file at `path`.
///
/// Tries three formats in order, accepting the first that decodes to
/// a valid Secp256k1KeyPair:
///
///   1. **fastcrypto base64** via the `EncodeDecodeBase64` trait — the
///      canonical on-disk format the rest of the codebase emits (see
///      `types/src/config/node_config.rs::read_authority_keypair_from_file`).
///   2. **Plain base64 of raw 32-byte privkey** — what an operator
///      gets from `cargo run --bin soma -- keytool ... | base64`.
///   3. **Raw 32 bytes** — what `openssl rand -out bridge.key 32`
///      produces. Last resort because most workflows base64 the key.
///
/// Each failed attempt is logged at debug level so an operator running
/// with `RUST_LOG=bridge_node=debug` can see which format actually
/// matched.
fn load_bridge_keypair(path: &std::path::Path) -> Result<Secp256k1KeyPair, RunError> {
    let contents = std::fs::read(path)
        .map_err(|e| RunError::Startup(format!("read bridge_key_path {}: {e}", path.display())))?;

    // 1. fastcrypto base64.
    if let Ok(s) = std::str::from_utf8(&contents) {
        let trimmed = s.trim();
        if let Ok(kp) = Secp256k1KeyPair::decode_base64(trimmed) {
            tracing::debug!(
                path = %path.display(),
                "loaded bridge key via fastcrypto base64",
            );
            return Ok(kp);
        }
        // 2. Plain base64 of raw privkey bytes.
        if let Ok(raw) = Base64::decode(trimmed) {
            if let Ok(kp) = Secp256k1KeyPair::from_bytes(&raw) {
                tracing::debug!(
                    path = %path.display(),
                    raw_len = raw.len(),
                    "loaded bridge key via plain base64 raw bytes",
                );
                return Ok(kp);
            }
        }
    }

    // 3. Raw 32 bytes on disk.
    if let Ok(kp) = Secp256k1KeyPair::from_bytes(&contents) {
        tracing::debug!(
            path = %path.display(),
            raw_len = contents.len(),
            "loaded bridge key via raw bytes",
        );
        return Ok(kp);
    }

    Err(RunError::Startup(format!(
        "bridge_key_path {}: none of the supported formats decoded \
         (tried: fastcrypto base64, plain base64 of raw 32-byte privkey, raw bytes)",
        path.display()
    )))
}
