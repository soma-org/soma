//! `dump_bridge_state` — read the raw SystemState OBJECT (not the proto
//! `GetEpoch.system_state` view) and BCS-deserialize the full
//! `BridgeState`, including `bridge_registrations`.
//!
//! Why not just call `client.get_latest_system_state()`?  Because the proto
//! schema for SystemState doesn't yet carry a `bridge_state` field, and the
//! `TryFrom<proto::SystemState>` impl hardcodes `bridge_state:
//! BridgeState::new(BridgeCommittee::empty())`. Anything that reads bridge
//! state via that path sees empty results regardless of on-chain truth.
//!
//! This bin instead fetches the raw `SystemState` shared object
//! (ObjectID = 0x...05), reads its BCS contents, and decodes
//! `types::system_state::SystemState` directly — bypassing the broken
//! proto layer.

use clap::Parser;
use rpc::api::client::Client;
use types::SYSTEM_STATE_OBJECT_ID;
use types::system_state::SystemState;
use types::system_state::SystemStateTrait;

#[derive(Parser)]
struct Args {
    /// Soma fullnode RPC URL.
    #[arg(long)]
    soma_rpc: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!("Connecting to {} ...", args.soma_rpc);
    let mut client = Client::new(args.soma_rpc.as_str())?;

    eprintln!("Fetching SystemState object {} (raw BCS path) ...", SYSTEM_STATE_OBJECT_ID);
    let obj = client.get_object(SYSTEM_STATE_OBJECT_ID).await?;

    let state: SystemState = bcs::from_bytes(obj.as_inner().data.contents())
        .map_err(|e| anyhow::anyhow!("BCS decode SystemState: {e}"))?;

    let bridge = state.bridge_state();

    println!("--- BridgeState dump (raw BCS path) ---");
    println!("epoch: {}", state.epoch());
    println!("total_usdc_supply: {}", bridge.total_usdc_supply);
    println!("next_withdrawal_nonce: {}", bridge.next_withdrawal_nonce);
    println!();
    println!("bridge_registrations: {} entries", bridge.bridge_registrations.len());
    for (idx, (addr, reg)) in bridge.bridge_registrations.iter().enumerate() {
        println!("  [{}] validator_addr={}", idx, hex::encode(addr.to_inner()));
        println!("      bridge_pubkey=0x{}", hex::encode(reg.bridge_pubkey.as_bytes()));
        println!("      http_url={}", reg.http_url);
    }
    println!();
    println!("bridge_committee.members: {} entries", bridge.bridge_committee.members.len());
    for (idx, (pk, member)) in bridge.bridge_committee.members.iter().enumerate() {
        println!(
            "  [{}] bridge_pubkey=0x{} voting_power={} blocklisted={}",
            idx,
            hex::encode(pk.as_bytes()),
            member.voting_power,
            member.is_blocklisted
        );
        println!(
            "      soma_address={} http_url={}",
            hex::encode(member.soma_address.to_inner()),
            member.http_url
        );
    }

    Ok(())
}
