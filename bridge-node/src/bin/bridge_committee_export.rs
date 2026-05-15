//! `bridge-committee-export` — read the live Soma `BridgeState.bridge_committee`
//! via RPC and emit it in the shape the Foundry `DeployBridge.s.sol` script
//! consumes.
//!
//! # Deployment flow this fits into
//!
//! 1. **Bring up the Soma chain** with its committee already registered on
//!    the system-state `BridgeState.bridge_committee` map. (Bridge committee
//!    membership is established via the normal validator-side
//!    `BridgeRegistration` flow, not by this tool.)
//!
//! 2. **Edit `bridge/evm/deploy_configs/<EVM-CHAIN-ID>.json`** once with the
//!    chain-specific bits the operator owns (USDC address, limiter cap,
//!    supported source chains, etc.).
//!
//! 3. **Run this tool** against any Soma fullnode RPC, pointing `--output`
//!    at the same deploy_configs file. It fetches the live committee,
//!    drops blocklisted members, derives the 20-byte Ethereum address for
//!    each via `keccak256(uncompressed_pubkey[1..])[12..]` (Sui parity),
//!    normalizes the voting powers so they sum to exactly 10000 BPS, and
//!    **merges** the committee fields (`committeeMembers`, `committeeStake`,
//!    `somaCommitteeDigest`) into the existing JSON — preserving the
//!    operator's other fields. If `--output` is not provided, the full
//!    JSON is written to stdout instead and no merge is performed.
//!
//! 4. **Run the Foundry deploy:**
//!
//!    ```text
//!    cd bridge/evm
//!    forge script script/DeployBridge.s.sol:DeployBridge \
//!        --rpc-url $BASE_SEPOLIA_RPC \
//!        --private-key $DEPLOYER_PK \
//!        --broadcast
//!    ```
//!
//! 5. **Record `somaCommitteeDigest`** from the JSON in your deployment
//!    ledger. It's a SHA-256 over the (sorted-by-Eth-address) committee
//!    state; if it doesn't match what the bridge-node sees at runtime,
//!    the on-chain `SomaBridge` is bound to a stale committee and
//!    withdrawals will fail signature verification.
//!
//! # Why normalize defensively
//!
//! `BridgeCommittee.initialize` on the Eth side aborts when the sum of
//! stakes != 10000. The on-chain Soma committee *should* already sum to
//! 10000 (`SystemParameters::bridge_total_voting_power`), but the watchdog/
//! audit doesn't trust system invariants for irreversible deployment
//! steps. We rescale and distribute the rounding remainder across the
//! largest entries (by raw stake) so the emitted stakes sum to exactly
//! 10000 even if the on-chain numbers drift.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bridge_node::soma_client::SomaBridgeClient;
use clap::Parser;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use types::bridge::{BridgeChainId, derive_eth_address};

/// Operator tool: export the live Soma BridgeState committee in the format
/// the Foundry `DeployBridge.s.sol` script consumes.
///
/// Connects to a Soma fullnode RPC, reads `BridgeState.bridge_committee`,
/// drops blocklisted members, derives each member's 20-byte Ethereum
/// address from its 33-byte compressed secp256k1 pubkey, and normalizes
/// voting powers so they sum to exactly 10000 BPS (the Solidity-side
/// `BridgeCommittee.initialize` invariant).
///
/// With `--output <FILE>`:
///   * If FILE exists, MERGE the committee fields (`committeeMembers`,
///     `committeeStake`, `somaCommitteeDigest`) into the existing JSON,
///     preserving all other fields. This is the production flow: the
///     operator owns `usdcAddress` / `limiterTotalLimit` / etc., and
///     re-runs this tool whenever the on-chain committee rotates.
///   * If FILE doesn't exist, write a full deploy_configs-shaped JSON
///     with empty placeholders for the operator-owned fields so the
///     missing fields surface as deploy-time errors rather than silent
///     zeros.
///
/// Without `--output`, the full JSON is written to stdout (useful for
/// piping into `jq` or for ad-hoc inspection).
#[derive(Parser, Debug)]
#[command(
    name = "bridge-committee-export",
    version,
    about = "Export the live Soma bridge committee for Foundry deployment.",
    long_about = None,
)]
struct Args {
    /// Soma fullnode RPC endpoint (e.g. http://localhost:9000).
    #[arg(long)]
    soma_rpc: String,

    /// `BridgeChainId` byte for the EVM target the bridge will be deployed
    /// to. Echoed into the JSON output for the Foundry script. Defaults to
    /// 13 (`BaseSepolia`).
    #[arg(long, default_value_t = 13)]
    target_chain_id: u8,

    /// Path to a deploy_configs JSON file to write or merge into.
    ///
    /// * If the path EXISTS, the committee fields (`committeeMembers`,
    ///   `committeeStake`, `somaCommitteeDigest`) are merged into the
    ///   existing JSON. All other fields (`usdcAddress`, `ethChainId`,
    ///   `limiterTotalLimit`, `supportedSomaChains`) are preserved.
    /// * If the path does NOT exist, a fresh deploy_configs JSON is
    ///   written with empty placeholders for the operator-owned fields.
    ///
    /// If omitted, full JSON goes to stdout and no merge happens.
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Tokio is a workspace dep on bridge-node; spin a current-thread
    // runtime here rather than `#[tokio::main]` so the CLI ergonomics
    // (early exits with status codes) stay readable.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ERROR: failed to build tokio runtime: {e}");
            return ExitCode::from(1);
        }
    };

    match rt.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(RunError::Rpc(msg)) => {
            eprintln!("ERROR (rpc): {msg}");
            ExitCode::from(1)
        }
        Err(RunError::Empty(msg)) => {
            eprintln!("ERROR (empty committee): {msg}");
            ExitCode::from(2)
        }
        Err(RunError::Io(msg)) => {
            eprintln!("ERROR (io): {msg}");
            ExitCode::from(1)
        }
    }
}

enum RunError {
    /// RPC unreachable / malformed response. Exit status 1.
    Rpc(String),
    /// Committee is empty after dropping blocklisted members, or every
    /// remaining member has zero voting power. Exit status 2 — this is
    /// an operational error, not a system fault, and demands a human
    /// looking at chain state before retrying.
    Empty(String),
    /// `--output` file read / write / parse failed. Exit status 1.
    Io(String),
}

async fn run(args: Args) -> Result<(), RunError> {
    eprintln!("[1/4] Connecting to Soma RPC at {} ...", args.soma_rpc);

    // The `soma_chain_id` passed to `new_rpc` is used by the client for
    // outbound-tx record-id derivation + metric labels. This tool only
    // *reads*, so any well-formed Soma-side chain id works. Use the
    // dev-config default to keep the construction simple.
    let client = SomaBridgeClient::new_rpc(
        &args.soma_rpc,
        BridgeChainId::SomaCustom,
    )
    .await
    .map_err(|e| RunError::Rpc(format!("connect: {e}")))?;

    eprintln!("[2/4] Fetching BridgeState.bridge_committee ...");
    let committee = client
        .get_bridge_committee()
        .await
        .map_err(|e| RunError::Rpc(format!("get_bridge_committee: {e}")))?;

    eprintln!(
        "       Got {} committee member(s) (including blocklisted).",
        committee.members.len()
    );

    // Drop blocklisted members up front. Blocklisted seats stay in the
    // map on Soma (so sigs from them are still *recognized*, just zero-
    // weight) — but the Eth-side `BridgeCommittee.initialize` doesn't
    // model "blocklisted with zero stake", so we filter them out.
    let active: Vec<(Vec<u8>, u64)> = committee
        .members
        .iter()
        .filter(|(_, m)| !m.is_blocklisted)
        .map(|(pk, m)| (pk.as_bytes().to_vec(), m.voting_power))
        .collect();

    if active.is_empty() {
        return Err(RunError::Empty(
            "committee is empty after dropping blocklisted members — \
             nothing to deploy"
                .to_string(),
        ));
    }

    let blocklisted_count = committee.members.len() - active.len();
    if blocklisted_count > 0 {
        eprintln!(
            "       Dropped {blocklisted_count} blocklisted member(s); \
             {} active.",
            active.len()
        );
    }

    eprintln!("[3/4] Deriving Eth addresses + normalizing stakes ...");

    // Derive the 20-byte Eth address for each non-blocklisted member.
    // `as_bytes` returned the raw 33-byte compressed pubkey above; we
    // need to rebuild a typed `BridgePubkey` for the helper, which is
    // infallible because the stored map keys are already validated.
    let mut derived: Vec<([u8; 20], u64)> = Vec::with_capacity(active.len());
    for (pk_bytes, vp) in &active {
        let pk = types::bridge::BridgePubkey::from_bytes(pk_bytes).map_err(|e| {
            // Should never happen — on-chain map keys are validated at
            // insertion. If it does, the chain is in a corrupted state
            // and we don't want to paper over it.
            RunError::Rpc(format!(
                "stored committee pubkey failed curve validation: {e}"
            ))
        })?;
        let addr = derive_eth_address(&pk);
        derived.push((addr, *vp));
    }

    let raw_total: u128 = derived.iter().map(|(_, vp)| *vp as u128).sum();
    if raw_total == 0 {
        return Err(RunError::Empty(
            "every active committee member has voting_power == 0 — \
             nothing to normalize"
                .to_string(),
        ));
    }

    // Normalize so the stakes sum to *exactly* 10000. Floor-divide first,
    // then distribute the remainder one-BPS-at-a-time across the entries
    // sorted by raw stake (descending). The largest stakes absorb the
    // rounding error, which keeps relative ordering intact.
    //
    // We do this in u128 because raw_total*10000 can overflow u64 if the
    // chain ever stores stakes in something larger than BPS units.
    const TOTAL_BPS: u128 = 10_000;
    let mut scaled: Vec<u16> = derived
        .iter()
        .map(|(_, vp)| (((*vp as u128) * TOTAL_BPS) / raw_total) as u16)
        .collect();

    let scaled_sum: u128 = scaled.iter().map(|s| *s as u128).sum();
    let mut remainder = TOTAL_BPS.saturating_sub(scaled_sum) as usize;

    if remainder > 0 {
        // Sort indices by raw stake descending so the +1 BPS bumps go to
        // the largest entries. Ties broken by lower index to stay
        // deterministic.
        let mut idx: Vec<usize> = (0..derived.len()).collect();
        idx.sort_by(|&a, &b| {
            derived[b]
                .1
                .cmp(&derived[a].1)
                .then(a.cmp(&b))
        });
        // If the remainder exceeds the number of members we'd be racing
        // through the loop a lot — but TOTAL_BPS=10000 and each entry
        // can absorb at most `10000 - scaled[i]` extra, so a single pass
        // through `idx` is more than enough in practice. Cap with a
        // safety limit.
        let mut cursor = 0;
        while remainder > 0 && cursor < idx.len() * 64 {
            let i = idx[cursor % idx.len()];
            scaled[i] = scaled[i].saturating_add(1);
            remainder -= 1;
            cursor += 1;
        }
        if remainder != 0 {
            return Err(RunError::Empty(format!(
                "could not distribute normalization remainder ({remainder} \
                 BPS left); committee size {} is pathologically small",
                derived.len()
            )));
        }
    }

    // Re-check the invariant before emitting. The Solidity contract will
    // revert if this is off, so it's cheap defense to assert here too.
    let final_sum: u128 = scaled.iter().map(|s| *s as u128).sum();
    if final_sum != TOTAL_BPS {
        return Err(RunError::Empty(format!(
            "post-normalization stake sum is {final_sum}, expected 10000 — \
             rejecting export rather than producing a deploy-time revert"
        )));
    }

    // Build the parallel (addr, stake) slice that the JSON and the
    // digest both consume. Keep the original map order — `members.iter()`
    // on a `BTreeMap<BridgePubkey, _>` is pubkey-sorted, which is stable.
    let members_with_stake: Vec<([u8; 20], u16)> = derived
        .iter()
        .zip(scaled.iter())
        .map(|((addr, _), stake)| (*addr, *stake))
        .collect();

    // Committee digest: sort by Eth address (NOT by pubkey order — the
    // Eth-side contract identifies seats by address, so the digest
    // should be invariant under pubkey re-encodings) and hash the
    // (addr || stake_u16_be) concatenation.
    let digest_hex = {
        let mut sorted = members_with_stake.clone();
        sorted.sort_by_key(|(addr, _)| *addr);
        let mut hasher = Sha256::new();
        for (addr, stake) in &sorted {
            hasher.update(addr);
            hasher.update(stake.to_be_bytes());
        }
        let bytes = hasher.finalize();
        format!("0x{}", hex::encode(bytes))
    };

    eprintln!("[4/4] Writing committee fields.");
    eprintln!();
    eprintln!(
        "       active_members           = {}",
        members_with_stake.len()
    );
    eprintln!("       raw_total_voting_power   = {raw_total}");
    eprintln!("       normalized_stake_sum_bps = {final_sum}");
    eprintln!("       target_chain_id          = {}", args.target_chain_id);
    eprintln!("       soma_committee_digest    = {digest_hex}");

    // Build the JSON values that we'll either merge or write fresh.
    let members_json: Vec<Value> = members_with_stake
        .iter()
        .map(|(addr, _)| Value::String(format!("0x{}", hex::encode(addr))))
        .collect();
    let stakes_json: Vec<Value> = members_with_stake
        .iter()
        .map(|(_, s)| Value::Number((*s).into()))
        .collect();

    match &args.output {
        Some(path) => {
            write_or_merge(
                path,
                members_json,
                stakes_json,
                &digest_hex,
                args.target_chain_id,
            )?;
        }
        None => {
            // No output path — write the full deploy_configs JSON to
            // stdout for inspection. Empty operator-owned fields are
            // emitted so a downstream `> file.json` redirect produces
            // a file that's structurally correct but will (loudly) fail
            // the deploy-time validation until the operator fills them.
            let full = build_full_config(
                members_json,
                stakes_json,
                &digest_hex,
                args.target_chain_id,
            );
            let pretty = serde_json::to_string_pretty(&full).map_err(|e| {
                RunError::Io(format!("serialize JSON: {e}"))
            })?;
            println!("{pretty}");
        }
    }

    Ok(())
}

/// Build a full deploy_configs-shaped JSON value, with empty placeholders
/// for the operator-owned fields. Used when `--output` points at a path
/// that doesn't exist yet (greenfield deploy).
fn build_full_config(
    members: Vec<Value>,
    stakes: Vec<Value>,
    digest_hex: &str,
    target_chain_id: u8,
) -> Value {
    json!({
        "committeeMembers": members,
        "committeeStake": stakes,
        "ethChainId": target_chain_id,
        "usdcAddress": "",
        "limiterTotalLimit": "",
        "supportedSomaChains": [],
        "somaCommitteeDigest": digest_hex,
    })
}

/// Write the committee fields to `path`. If the file exists, parse it as
/// JSON and merge only the committee-owned fields, preserving everything
/// else the operator put there. If it doesn't exist, write a fresh
/// deploy_configs-shaped file with empty placeholders for the
/// operator-owned fields.
///
/// Both branches pretty-print with 2-space indent to match the on-disk
/// example file — diffs against the canonical file should be limited to
/// the fields we touched.
fn write_or_merge(
    path: &Path,
    members: Vec<Value>,
    stakes: Vec<Value>,
    digest_hex: &str,
    target_chain_id: u8,
) -> Result<(), RunError> {
    let value = if path.exists() {
        // Read + parse the existing JSON, then mutate the three
        // committee-owned fields in place. `serde_json::Value` preserves
        // ordering of object keys (it uses a BTreeMap by default unless
        // `preserve_order` is enabled — and even with BTreeMap, ordering
        // is stable). The operator's other fields ride through untouched.
        let existing = fs::read_to_string(path).map_err(|e| {
            RunError::Io(format!("read {}: {e}", path.display()))
        })?;
        let mut parsed: Value = serde_json::from_str(&existing).map_err(|e| {
            RunError::Io(format!("parse {} as JSON: {e}", path.display()))
        })?;
        let obj = parsed.as_object_mut().ok_or_else(|| {
            RunError::Io(format!(
                "{}: top-level JSON value is not an object",
                path.display()
            ))
        })?;
        obj.insert("committeeMembers".to_string(), Value::Array(members));
        obj.insert("committeeStake".to_string(), Value::Array(stakes));
        obj.insert(
            "somaCommitteeDigest".to_string(),
            Value::String(digest_hex.to_string()),
        );

        eprintln!(
            "       Merged committee fields into existing {}",
            path.display()
        );
        parsed
    } else {
        eprintln!(
            "       File {} did not exist — writing fresh template.",
            path.display()
        );
        build_full_config(members, stakes, digest_hex, target_chain_id)
    };

    // 2-space indent matches the example file. `serde_json::to_string_pretty`
    // emits 2-space by default — explicit here for clarity / future-proofing.
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"  ");
    let mut buf = Vec::new();
    let mut ser =
        serde_json::Serializer::with_formatter(&mut buf, formatter);
    serde::Serialize::serialize(&value, &mut ser).map_err(|e| {
        RunError::Io(format!("serialize merged JSON: {e}"))
    })?;
    // Trailing newline for POSIX-friendly diffs.
    buf.push(b'\n');

    fs::write(path, &buf).map_err(|e| {
        RunError::Io(format!("write {}: {e}", path.display()))
    })?;
    eprintln!("       Wrote {}", path.display());

    Ok(())
}
