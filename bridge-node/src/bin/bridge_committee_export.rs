//! `bridge-committee-export` — read the live Soma `BridgeState.bridge_committee`
//! via RPC and emit it in the shape the Foundry `Deploy.s.sol` script consumes.
//!
//! # Deployment flow this fits into
//!
//! 1. **Bring up the Soma chain** with its committee already registered on
//!    the system-state `BridgeState.bridge_committee` map. (Bridge committee
//!    membership is established via the normal validator-side
//!    `BridgeRegistration` flow, not by this tool.)
//!
//! 2. **Run this tool** against any Soma fullnode RPC. It fetches the live
//!    committee, drops blocklisted members, derives the 20-byte Ethereum
//!    address for each via `keccak256(uncompressed_pubkey[1..])[12..]`
//!    (Sui parity), and normalizes the voting powers so they sum to
//!    exactly 10000 BPS — which is what `BridgeCommittee.initialize()`
//!    on the Solidity side requires.
//!
//! 3. **Source the emitted env-vars** into your Foundry deploy script:
//!
//!    ```text
//!    bridge-committee-export --soma-rpc http://localhost:9000 > committee.json
//!    eval "$(bridge-committee-export --soma-rpc http://localhost:9000 2>&1 1>/dev/null | grep export)"
//!    # or simpler — redirect stderr to a tmpfile and `source` it:
//!    bridge-committee-export --soma-rpc ... 2>committee.env >committee.json
//!    source committee.env
//!    forge script script/Deploy.s.sol --rpc-url $BASE_SEPOLIA_RPC ...
//!    ```
//!
//! 4. **Record `soma_committee_digest`** from the JSON in your deployment
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
use std::path::PathBuf;
use std::process::ExitCode;

use bridge_node::soma_client::SomaBridgeClient;
use clap::Parser;
use sha2::{Digest, Sha256};
use types::bridge::{BridgeChainId, derive_eth_address};

/// Operator tool: export the live Soma BridgeState committee in the format
/// the Foundry `Deploy.s.sol` script consumes.
///
/// Connects to a Soma fullnode RPC, reads `BridgeState.bridge_committee`,
/// drops blocklisted members, derives each member's 20-byte Ethereum
/// address from its 33-byte compressed secp256k1 pubkey, and normalizes
/// voting powers so they sum to exactly 10000 BPS (the Solidity-side
/// `BridgeCommittee.initialize` invariant).
///
/// Writes JSON to stdout and an `export …` env-var snippet to stderr.
/// Redirect them separately:
///   bridge-committee-export --soma-rpc ... > committee.json 2> committee.env
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

    /// Write JSON to this file instead of stdout. The env-var snippet
    /// still goes to stderr regardless — redirect it separately.
    #[arg(long)]
    output: Option<PathBuf>,
}

/// JSON payload emitted to stdout. Field shape is chosen to match what
/// the Foundry `Deploy.s.sol` script reads via `vm.readJson(...)`.
#[derive(serde::Serialize)]
struct CommitteeExport {
    /// 20-byte Eth addresses, lowercase `0x`-hex, in the order the
    /// `stakes` array indexes into.
    members: Vec<String>,
    /// Normalized stakes in basis points, indexed parallel to `members`.
    /// Sums to exactly 10000.
    stakes: Vec<u16>,
    /// Echo of `--target-chain-id` so a downstream consumer reading the
    /// JSON doesn't need to re-pass the flag.
    chain_id: u8,
    /// Raw on-chain stake total *before* normalization. Operators should
    /// compare this to `SystemParameters::bridge_total_voting_power`
    /// (10000 in dev configs) to detect drift; a large delta means a
    /// bug elsewhere even if normalization "fixed" it.
    total_stake_bps_before_normalize: u64,
    /// SHA-256 over the sorted-by-Eth-address (addr || stake_u16_be)
    /// concatenation. Stable identifier for "this committee state";
    /// record in your deployment ledger and compare against the live
    /// bridge-node digest at runtime to catch silent committee rotations.
    soma_committee_digest: String,
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
    /// `--output` file write failed. Exit status 1.
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

    eprintln!("[4/4] Emitting JSON to stdout and env-vars to stderr.");
    eprintln!();
    eprintln!(
        "       active_members           = {}",
        members_with_stake.len()
    );
    eprintln!("       raw_total_voting_power   = {raw_total}");
    eprintln!("       normalized_stake_sum_bps = {final_sum}");
    eprintln!("       target_chain_id          = {}", args.target_chain_id);
    eprintln!("       soma_committee_digest    = {digest_hex}");

    // --- Build the JSON payload ---------------------------------------
    let export = CommitteeExport {
        members: members_with_stake
            .iter()
            .map(|(addr, _)| format!("0x{}", hex::encode(addr)))
            .collect(),
        stakes: members_with_stake.iter().map(|(_, s)| *s).collect(),
        chain_id: args.target_chain_id,
        total_stake_bps_before_normalize: raw_total.min(u64::MAX as u128)
            as u64,
        soma_committee_digest: digest_hex.clone(),
    };

    let json = serde_json::to_string_pretty(&export).map_err(|e| {
        RunError::Io(format!("serialize JSON: {e}"))
    })?;

    if let Some(path) = &args.output {
        fs::write(path, &json).map_err(|e| {
            RunError::Io(format!("write {}: {e}", path.display()))
        })?;
        eprintln!();
        eprintln!("       Wrote JSON to {}", path.display());
    } else {
        // stdout — the machine-consumable channel. Anything verbose goes
        // to stderr so `cargo run ... > committee.json` works.
        println!("{json}");
    }

    // --- Env-var snippet to stderr ------------------------------------
    let members_csv = export.members.join(",");
    let stakes_csv = export
        .stakes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    eprintln!();
    eprintln!("# Source this in your shell before running the Foundry deploy:");
    eprintln!("export COMMITTEE_MEMBERS=\"{members_csv}\"");
    eprintln!("export COMMITTEE_STAKES=\"{stakes_csv}\"");
    eprintln!("export COMMITTEE_CHAIN_ID=\"{}\"", args.target_chain_id);
    eprintln!("export SOMA_COMMITTEE_DIGEST=\"{digest_hex}\"");

    Ok(())
}
