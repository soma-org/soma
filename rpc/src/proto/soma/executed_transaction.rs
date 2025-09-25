use super::*;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;

impl Merge<&ExecutedTransaction> for ExecutedTransaction {
    fn merge(&mut self, source: &ExecutedTransaction, mask: &FieldMaskTree) {
        let ExecutedTransaction {
            digest,
            transaction,
            signatures,
            effects,
            commit,
            timestamp,
            balance_changes,
            input_objects,
            output_objects,
            shard,
        } = source;

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if let Some(submask) = mask.subtree(Self::TRANSACTION_FIELD.name) {
            self.transaction = transaction
                .as_ref()
                .map(|t| Transaction::merge_from(t, &submask));
        }

        if let Some(submask) = mask.subtree(Self::SIGNATURES_FIELD.name) {
            self.signatures = signatures
                .iter()
                .map(|s| UserSignature::merge_from(s, &submask))
                .collect();
        }

        if let Some(submask) = mask.subtree(Self::EFFECTS_FIELD.name) {
            self.effects = effects
                .as_ref()
                .map(|e| TransactionEffects::merge_from(e, &submask));
        }

        if mask.contains(Self::COMMIT_FIELD.name) {
            self.commit = *commit;
        }

        if mask.contains(Self::TIMESTAMP_FIELD.name) {
            self.timestamp = *timestamp;
        }

        if mask.contains(Self::BALANCE_CHANGES_FIELD.name) {
            self.balance_changes = balance_changes.clone();
        }

        if mask.contains(Self::SHARD_FIELD.name) {
            self.shard = shard.clone();
        }

        if let Some(submask) = mask.subtree(Self::INPUT_OBJECTS_FIELD.name) {
            self.input_objects = input_objects
                .iter() // Changed from into_iter() to iter()
                .map(|object| {
                    let mut result = Object::default();
                    result.merge(object, &submask); // Use merge instead of merge_from
                    result
                })
                .collect();
        }

        if let Some(submask) = mask.subtree(Self::OUTPUT_OBJECTS_FIELD.name) {
            self.output_objects = output_objects
                .iter() // Changed from into_iter() to iter()
                .map(|object| {
                    let mut result = Object::default();
                    result.merge(object, &submask); // Use merge instead of merge_from
                    result
                })
                .collect();
        }
    }
}
