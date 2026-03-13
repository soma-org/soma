DROP INDEX IF EXISTS soma_targets_winning_model_owner;
DROP INDEX IF EXISTS soma_targets_fill_epoch;
DROP INDEX IF EXISTS soma_targets_winning_model_id;
DROP INDEX IF EXISTS soma_targets_submitter;

ALTER TABLE soma_targets
    DROP COLUMN IF EXISTS winning_distance_score,
    DROP COLUMN IF EXISTS winning_loss_score,
    DROP COLUMN IF EXISTS winning_model_owner,
    DROP COLUMN IF EXISTS fill_epoch,
    DROP COLUMN IF EXISTS distance_threshold,
    DROP COLUMN IF EXISTS model_ids_json,
    DROP COLUMN IF EXISTS winning_data_url,
    DROP COLUMN IF EXISTS winning_data_checksum,
    DROP COLUMN IF EXISTS winning_data_size;
