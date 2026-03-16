DROP TRIGGER IF EXISTS soma_tx_details_notify ON soma_tx_details;
DROP FUNCTION IF EXISTS notify_new_transaction();

DROP TRIGGER IF EXISTS soma_targets_notify_filled ON soma_targets;
DROP FUNCTION IF EXISTS notify_target_filled();

DROP TRIGGER IF EXISTS cp_sequence_numbers_notify ON cp_sequence_numbers;
DROP FUNCTION IF EXISTS notify_new_checkpoint();

DROP TRIGGER IF EXISTS kv_epoch_starts_notify ON kv_epoch_starts;
DROP FUNCTION IF EXISTS notify_new_epoch();
