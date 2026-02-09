use std::{collections::BTreeSet, path::Path};

use types::{
    effects::TransactionEffectsAPI as _,
    error::{SomaError, SomaResult},
    full_checkpoint_content::{Checkpoint, CheckpointData, ExecutedTransaction, ObjectSet},
    storage::object_store::ObjectStore,
};

use crate::{
    cache::TransactionCacheRead,
    checkpoints::checkpoint_executor::{CheckpointExecutionData, CheckpointTransactionData},
};

pub(crate) fn store_checkpoint_locally(
    path: impl AsRef<Path>,
    checkpoint_data: &CheckpointData,
) -> SomaResult {
    let path = path.as_ref();
    let file_name = format!("{}.chk", checkpoint_data.checkpoint_summary.sequence_number);

    std::fs::create_dir_all(path).map_err(|err| {
        SomaError::FileIOError(format!("failed to save full checkpoint content locally {:?}", err))
    })?;

    bcs::to_bytes(&checkpoint_data)
        .map_err(|_| SomaError::TransactionSerializationError {
            error: "failed to serialize full checkpoint content".to_string(),
        }) // Map the first error
        .and_then(|blob| {
            std::fs::write(path.join(file_name), blob).map_err(|_| {
                SomaError::FileIOError("failed to save full checkpoint content locally".to_string())
            })
        })?;

    Ok(())
}

pub(crate) fn load_checkpoint(
    ckpt_data: &CheckpointExecutionData,
    ckpt_tx_data: &CheckpointTransactionData,
    object_store: &dyn ObjectStore,
    transaction_cache_reader: &dyn TransactionCacheRead,
) -> SomaResult<Checkpoint> {
    let mut transactions = Vec::with_capacity(ckpt_tx_data.transactions.len());
    for (tx, fx) in ckpt_tx_data.transactions.iter().zip(ckpt_tx_data.effects.iter()) {
        let transaction = ExecutedTransaction {
            transaction: tx.transaction_data().clone(),
            signatures: tx.tx_signatures().to_vec(),
            effects: fx.clone(),
        };
        transactions.push(transaction);
    }

    let object_set = {
        let refs = transactions
            .iter()
            .flat_map(|tx| types::storage::get_transaction_object_set(&tx.transaction, &tx.effects))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let objects = object_store.multi_get_objects_by_key(&refs);

        let mut object_set = ObjectSet::default();
        for (idx, object) in objects.into_iter().enumerate() {
            object_set.insert(object.ok_or_else(|| {
                types::storage::storage_error::Error::custom(format!(
                    "unabled to load object {:?}",
                    refs[idx]
                ))
            })?);
        }
        object_set
    };
    let checkpoint = Checkpoint {
        summary: ckpt_data.checkpoint.clone().into(),
        contents: ckpt_data.checkpoint_contents.clone(),
        transactions,
        object_set,
    };
    Ok(checkpoint)
}
