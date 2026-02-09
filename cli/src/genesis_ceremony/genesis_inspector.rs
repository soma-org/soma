use inquire::Select;
use std::collections::BTreeMap;

use types::config::genesis_config::SHANNONS_PER_SOMA;
use types::genesis::UnsignedGenesis;
use types::system_state::validator::Validator;
use types::system_state::{SystemStateTrait, get_system_state};

const STR_ALL: &str = "All";
const STR_EXIT: &str = "Exit";
const STR_SOMA: &str = "Soma";
const STR_STAKED_SOMA: &str = "StakedSoma";
const STR_OTHER: &str = "Other";
const STR_SOMA_DISTRIBUTION: &str = "Soma Distribution";
const STR_OBJECTS: &str = "Objects";
const STR_VALIDATORS: &str = "Validators";

pub fn examine_genesis_checkpoint(genesis: &UnsignedGenesis) {
    let system_state =
        get_system_state(&genesis.objects()).expect("System state must exist in genesis");

    // Prepare Validator info
    let consensus_set = &system_state.validators.validators;
    let consensus_validator_map: BTreeMap<String, &Validator> =
        consensus_set.iter().map(|v| (v.metadata.soma_address.to_string(), v)).collect();

    let mut consensus_validator_options: Vec<&str> =
        consensus_validator_map.keys().map(|s| s.as_str()).collect();
    consensus_validator_options.push(STR_ALL);
    consensus_validator_options.push(STR_EXIT);
    println!("Total Number of Consensus Validators: {}", consensus_set.len());

    // Prepare Soma distribution info
    let mut soma_distribution: BTreeMap<String, BTreeMap<String, (&str, u64)>> = BTreeMap::new();

    // Add emission pool
    let entry = soma_distribution.entry("System".to_string()).or_default();
    entry.insert("Emission Pool".to_string(), (STR_SOMA, system_state.emission_pool.balance));

    println!("Total Number of Objects: {}", genesis.objects().len());

    // Main loop for inspection
    let main_options: Vec<&str> =
        vec![STR_SOMA_DISTRIBUTION, STR_VALIDATORS, STR_OBJECTS, STR_EXIT];

    loop {
        let ans = Select::new(
            "Select one main category to examine ('Exit' to exit the program):",
            main_options.clone(),
        )
        .prompt();

        match ans {
            Ok(name) if name == STR_SOMA_DISTRIBUTION => {
                examine_soma_distribution(&soma_distribution);
            }
            Ok(name) if name == STR_VALIDATORS => {
                examine_validators(&consensus_validator_options, &consensus_validator_map);
            }
            Ok(name) if name == STR_OBJECTS => {
                println!("Examine Objects (total: {})", genesis.objects().len());
                examine_objects(genesis);
            }
            Ok(name) if name == STR_EXIT => break,
            Ok(_) => (),
            Err(err) => {
                println!("Error: {err}");
                break;
            }
        }
    }
}

fn examine_validators(validator_options: &[&str], validator_map: &BTreeMap<String, &Validator>) {
    if validator_map.is_empty() {
        println!("No validators found.");
        print_divider(&format!("Validators"));
        return;
    }

    loop {
        let ans = Select::new(
            &format!(
                "Select one validator to examine ('All' to display all, 'Exit' to return to Main):"
            ),
            validator_options.to_vec(),
        )
        .prompt();

        match ans {
            Ok(name) if name == STR_ALL => {
                for validator in validator_map.values() {
                    display_validator(validator);
                }
            }
            Ok(name) if name == STR_EXIT => break,
            Ok(name) => {
                if let Some(validator) = validator_map.get(name) {
                    display_validator(validator);
                }
            }
            Err(err) => {
                println!("Error: {err}");
                break;
            }
        }
    }
    print_divider(&format!("Validators"));
}

fn examine_objects(genesis: &UnsignedGenesis) {
    let object_options: Vec<&str> = vec![STR_SOMA, STR_STAKED_SOMA, STR_OTHER, STR_EXIT];

    loop {
        let ans = Select::new(
            "Select one object category to examine ('Exit' to return to Main):",
            object_options.clone(),
        )
        .prompt();

        match ans {
            Ok(name) if name == STR_EXIT => break,
            Ok(name) if name == STR_SOMA => {
                for object in genesis.objects() {
                    if object.as_coin().is_some() {
                        println!("ID: {}", object.id());
                        println!("Owner: {:?}", object.owner());
                        println!();
                    }
                }
                print_divider("Soma");
            }
            Ok(name) if name == STR_STAKED_SOMA => {
                for object in genesis.objects() {
                    if object.as_staked_soma().is_some() {
                        println!("ID: {}", object.id());
                        println!("Owner: {:?}", object.owner());
                        println!();
                    }
                }
                print_divider("StakedSoma");
            }
            Ok(name) if name == STR_OTHER => {
                for object in genesis.objects() {
                    if object.as_coin().is_none() && object.as_staked_soma().is_none() {
                        println!("ID: {}", object.id());
                        println!("Type: {:?}", object.type_());
                        println!("Owner: {:?}", object.owner());
                        println!();
                    }
                }
                print_divider("Other");
            }
            Ok(_) => (),
            Err(err) => {
                println!("Error: {err}");
                break;
            }
        }
    }
    print_divider("Objects");
}

fn examine_soma_distribution(soma_distribution: &BTreeMap<String, BTreeMap<String, (&str, u64)>>) {
    let mut total_soma: u64 = 0;

    for (owner, coins) in soma_distribution {
        let mut amount_sum: u64 = 0;
        for (_name, (_type, value)) in coins {
            amount_sum += value;
        }
        total_soma += amount_sum;

        println!("Owner: {:?}", owner);
        println!(
            "Total Amount: {} SHANNONS or {} SOMA",
            amount_sum,
            amount_sum / SHANNONS_PER_SOMA
        );
        println!("{:#?}\n", coins);
    }

    println!(
        "Total Supply Accounted: {} SHANNONS or {} SOMA\n",
        total_soma,
        total_soma / SHANNONS_PER_SOMA
    );
    print_divider("Soma Distribution");
}

fn display_validator(validator: &Validator) {
    println!("Validator Address: {}", validator.metadata.soma_address);
    println!("Protocol Key: {:?}", validator.metadata.protocol_pubkey);
    println!("Network Key: {:?}", validator.metadata.network_pubkey);
    println!("Worker Key: {:?}", validator.metadata.worker_pubkey);
    println!("Network Address: {}", validator.metadata.net_address);
    println!("P2P Address: {}", validator.metadata.p2p_address);
    println!("Primary Address: {}", validator.metadata.primary_address);

    println!("Voting Power: {}", validator.voting_power);
    println!("Commission Rate: {}", validator.commission_rate);
    println!("Staking Pool ID: {}", validator.staking_pool.id);
    println!("Staking Pool Soma Balance: {}", validator.staking_pool.soma_balance);
    println!("Next Epoch Stake: {}", validator.next_epoch_stake);
    print_divider(&validator.metadata.soma_address.to_string());
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn print_divider(title: &str) {
    let title = format!("End of {title}");
    let divider_length = 80;
    let left_divider_length = 10;
    let divider_op = "-";
    let divider = divider_op.repeat(divider_length);
    let left_divider = divider_op.repeat(left_divider_length);
    let title_len = title.len().min(divider_length - left_divider_length * 2);
    let margin_length = (divider_length - left_divider_length * 2 - title_len) / 2;
    let margin = " ".repeat(margin_length);
    println!();
    println!("{divider}");
    println!("{left_divider}{margin}{title}{margin}{left_divider}");
    println!("{divider}");
    println!();
}
