use crate::base::SomaAddress;
use crate::effects::TransactionEffects;
use crate::object::Object;
use crate::object::Owner;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BalanceChange {
    /// Owner of the balance change
    pub address: SomaAddress,

    /// The amount indicate the balance value changes.
    ///
    /// A negative amount means spending coin value and positive means receiving coin value.
    pub amount: i128,
}

fn coins(objects: &[Object]) -> impl Iterator<Item = (&SomaAddress, u64)> + '_ {
    objects.iter().filter_map(|object| {
        let address = match object.owner() {
            Owner::AddressOwner(soma_address) => soma_address,
            Owner::Shared { .. } | Owner::Immutable => return None,
        };
        let balance = object.as_coin()?;
        Some((address, balance))
    })
}

pub fn derive_balance_changes(
    _effects: &TransactionEffects,
    input_objects: &[Object],
    output_objects: &[Object],
) -> Vec<BalanceChange> {
    // 1. subtract all input coins
    let balances = coins(input_objects).fold(
        std::collections::BTreeMap::<_, i128>::new(),
        |mut acc, (address, balance)| {
            *acc.entry(address).or_default() -= balance as i128;
            acc
        },
    );

    // 2. add all mutated/output coins
    let balances = coins(output_objects).fold(balances, |mut acc, (address, balance)| {
        *acc.entry(address).or_default() += balance as i128;
        acc
    });

    balances
        .into_iter()
        .filter_map(|(address, amount)| {
            if amount == 0 {
                return None;
            }

            Some(BalanceChange {
                address: *address,
                amount,
            })
        })
        .collect()
}
