// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma object {get, list, transfer}` — all operations on individual
//! on-chain objects. Fungible balance transfers live under
//! `soma transfer` instead.

use anyhow::{Result, anyhow};
use clap::Parser;
use futures::TryStreamExt;
use rpc::utils::field::{FieldMask, FieldMaskUtil};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use soma_keys::keystore::AccountKeystore as _;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::{TxProcessingArgs, execute_or_serialize};
use crate::response::{ClientCommandResponse, ObjectOutput, ObjectsOutput};

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum ObjectCommand {
    /// Get a specific object by ID.
    Get {
        /// Object ID to fetch.
        object_id: ObjectID,
        /// Return BCS serialized data.
        #[clap(long)]
        bcs: bool,
    },

    /// List all objects owned by an address.
    List {
        /// Owner address (defaults to active address).
        owner: Option<KeyIdentity>,
    },

    /// Transfer an object to a recipient.
    #[clap(after_help = "\
EXAMPLES:
    soma object transfer 0xOBJECT_ID 0xRECIPIENT
    soma object transfer 0xOBJECT_ID alice")]
    Transfer {
        /// Object ID to transfer.
        object_id: ObjectID,
        /// Recipient address or alias.
        recipient: KeyIdentity,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

pub async fn execute(
    context: &mut WalletContext,
    cmd: ObjectCommand,
) -> Result<ClientCommandResponse> {
    match cmd {
        ObjectCommand::Get { object_id, bcs } => {
            let client = context.get_client().await?;
            let object = client
                .get_object(object_id)
                .await
                .map_err(|e| anyhow!("Failed to get object: {}", e.message()))?;

            Ok(ClientCommandResponse::Object(ObjectOutput::from_object(&object, bcs)))
        }

        ObjectCommand::List { owner } => {
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

        ObjectCommand::Transfer { object_id, recipient, tx_args } => {
            let sender = context.get_object_owner(&object_id).await?;
            let recipient_address = context.get_identity_address(Some(recipient))?;
            let client = context.get_client().await?;

            let object = client
                .get_object(object_id)
                .await
                .map_err(|e| anyhow!("Failed to get object: {}", e))?;
            let object_ref = object.compute_object_reference();

            let kind = TransactionKind::TransferObjects {
                objects: vec![object_ref],
                recipient: recipient_address,
            };

            execute_or_serialize(context, sender, kind, tx_args).await
        }
    }
}
