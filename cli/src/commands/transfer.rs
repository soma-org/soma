// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma transfer {soma, usdc}` — fungible balance transfers.
//!
//! Object transfers (the non-fungible kind) live under `soma object
//! transfer` instead, since the object is the primary subject of the
//! action.

use anyhow::Result;
use clap::Parser;
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::CoinType;
use types::transaction::{BalanceTransferArgs, TransactionKind};

use crate::client_commands::{TxProcessingArgs, execute_or_serialize};
use crate::response::ClientCommandResponse;
use crate::soma_amount::SomaAmount;
use crate::usdc_amount::UsdcAmount;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum TransferCommand {
    /// Transfer SOMA to a recipient.
    #[clap(after_help = "\
EXAMPLES:
    soma transfer soma 10 0x1234...5678
    soma transfer soma 0.5 alice")]
    Soma {
        /// Amount in SOMA (e.g. `10`, `0.5`).
        amount: SomaAmount,
        /// Recipient address or alias.
        recipient: KeyIdentity,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Transfer USDC to a recipient.
    #[clap(after_help = "\
EXAMPLES:
    soma transfer usdc 1.50 0x1234...5678
    soma transfer usdc 100 alice")]
    Usdc {
        /// Amount in USDC (e.g. `1.50`).
        amount: UsdcAmount,
        /// Recipient address or alias.
        recipient: KeyIdentity,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

impl TransferCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ClientCommandResponse> {
        let (coin_type, amount_base_units, recipient, tx_args) = match self {
            Self::Soma { amount, recipient, tx_args } => {
                (CoinType::Soma, amount.shannons(), recipient, tx_args)
            }
            Self::Usdc { amount, recipient, tx_args } => {
                (CoinType::Usdc, amount.microdollars(), recipient, tx_args)
            }
        };

        let sender = context.active_address()?;
        let recipient_address = context.get_identity_address(Some(recipient))?;

        let kind = TransactionKind::BalanceTransfer(BalanceTransferArgs {
            coin_type,
            transfers: vec![(recipient_address, amount_base_units)],
        });

        execute_or_serialize(context, sender, kind, tx_args).await
    }
}
