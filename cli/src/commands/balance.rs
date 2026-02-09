use anyhow::Result;
use futures::TryStreamExt;
use rpc::types::ObjectType;
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore as _;

use crate::response::BalanceOutput;

/// Execute the balance command
pub async fn execute(
    context: &WalletContext,
    address: Option<KeyIdentity>,
    with_coins: bool,
) -> Result<BalanceOutput> {
    let address = match address {
        Some(key_id) => context.config.keystore.get_by_identity(&key_id)?,
        None => {
            context.config.active_address.ok_or_else(|| anyhow::anyhow!("No active address set"))?
        }
    };

    let client = context.get_client().await?;

    let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
    request.owner = Some(address.to_string());
    request.object_type = Some(ObjectType::Coin.into());
    request.page_size = Some(1000);
    request.read_mask = Some(FieldMask::from_paths(["object_id", "version", "digest", "contents"]));

    let stream = client.list_owned_objects(request).await;
    tokio::pin!(stream);

    let mut total_balance: u128 = 0;
    let mut coin_details = Vec::new();

    while let Some(obj) = stream.try_next().await? {
        if let Some(coin) = obj.as_coin() {
            total_balance += coin as u128;
            if with_coins {
                coin_details.push((obj.id(), coin));
            }
        }
    }

    Ok(BalanceOutput {
        address,
        total_balance,
        coin_count: if with_coins {
            coin_details.len()
        } else {
            // Count coins even if not showing details
            let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
            request.owner = Some(address.to_string());
            request.object_type = Some(ObjectType::Coin.into());
            request.page_size = Some(1000);
            request.read_mask = Some(FieldMask::from_paths(["object_id"]));

            let stream = client.list_owned_objects(request).await;
            tokio::pin!(stream);

            let mut count = 0;
            while stream.try_next().await?.is_some() {
                count += 1;
            }
            count
        },
        coins: if with_coins { Some(coin_details) } else { None },
    })
}
