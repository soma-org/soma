use anyhow::{Result, anyhow};
use clap::Parser;
use futures::TryStreamExt;
use rpc::types::ObjectType;
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore as _;
use types::object::ObjectID;

use crate::response::{ClientCommandResponse, GasCoinsOutput, ObjectOutput, ObjectsOutput};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum ObjectsCommand {
    /// Get a specific object by ID
    #[clap(name = "get")]
    Get {
        /// Object ID to fetch
        object_id: ObjectID,
        /// Return BCS serialized data
        #[clap(long)]
        bcs: bool,
    },

    /// List all objects owned by an address
    #[clap(name = "list")]
    List {
        /// Owner address (defaults to active address)
        owner: Option<KeyIdentity>,
    },

    /// List gas coins owned by an address
    #[clap(name = "gas")]
    Gas {
        /// Owner address (defaults to active address)
        owner: Option<KeyIdentity>,
    },
}

/// Execute the objects command
pub async fn execute(
    context: &mut WalletContext,
    cmd: ObjectsCommand,
) -> Result<ClientCommandResponse> {
    match cmd {
        ObjectsCommand::Get { object_id, bcs } => {
            let client = context.get_client().await?;
            let object = client
                .get_object(object_id)
                .await
                .map_err(|e| anyhow!("Failed to get object: {}", e.message()))?;

            Ok(ClientCommandResponse::Object(ObjectOutput::from_object(&object, bcs)))
        }

        ObjectsCommand::List { owner } => {
            let address = match owner {
                Some(key_id) => context.config.keystore.get_by_identity(&key_id)?,
                None => context.active_address()?,
            };
            let client = context.get_client().await?;

            let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
            request.owner = Some(address.to_string());
            request.page_size = Some(100);
            request.read_mask = Some(FieldMask::from_paths([
                "object_id",
                "version",
                "digest",
                "object_type",
                "owner",
                "previous_transaction",
                "contents",
            ]));

            let stream = client.list_owned_objects(request).await;
            tokio::pin!(stream);

            let mut objects = Vec::new();
            while let Some(obj) = stream.try_next().await? {
                objects.push(ObjectOutput::from_object(&obj, false));
            }

            Ok(ClientCommandResponse::Objects(ObjectsOutput { address, objects }))
        }

        ObjectsCommand::Gas { owner } => {
            let address = match owner {
                Some(key_id) => context.config.keystore.get_by_identity(&key_id)?,
                None => context.active_address()?,
            };

            let gas_objects = context.get_all_gas_objects_owned_by_address(address).await?;

            // Fetch full objects to get balances
            let client = context.get_client().await?;
            let mut coins = Vec::new();
            for obj_ref in gas_objects {
                if let Ok(obj) = client.get_object(obj_ref.0).await
                    && let Some(balance) = obj.as_coin()
                {
                    coins.push((obj_ref, balance));
                }
            }

            Ok(ClientCommandResponse::Gas(GasCoinsOutput { address, coins }))
        }
    }
}
