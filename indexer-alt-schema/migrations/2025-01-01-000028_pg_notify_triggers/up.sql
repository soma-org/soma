-- Trigger functions that send Postgres NOTIFY events for GraphQL subscriptions.
-- The GraphQL server LISTENs on these channels and broadcasts to WebSocket clients.

-- New transaction inserted
CREATE OR REPLACE FUNCTION notify_new_transaction() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_transaction', json_build_object(
        'tx_sequence_number', NEW.tx_sequence_number,
        'kind', NEW.kind,
        'sender', encode(NEW.sender, 'hex'),
        'epoch', NEW.epoch,
        'timestamp_ms', NEW.timestamp_ms
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER soma_tx_details_notify
    AFTER INSERT ON soma_tx_details
    FOR EACH ROW EXECUTE FUNCTION notify_new_transaction();

-- Target status change (specifically when status = 'filled')
CREATE OR REPLACE FUNCTION notify_target_filled() RETURNS trigger AS $$
BEGIN
    IF NEW.status = 'filled' THEN
        PERFORM pg_notify('target_filled', json_build_object(
            'target_id', encode(NEW.target_id, 'hex'),
            'epoch', NEW.epoch,
            'fill_epoch', NEW.fill_epoch,
            'winning_model_id', encode(COALESCE(NEW.winning_model_id, ''::bytea), 'hex'),
            'submitter', encode(COALESCE(NEW.submitter, ''::bytea), 'hex'),
            'reward_pool', NEW.reward_pool
        )::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER soma_targets_notify_filled
    AFTER INSERT ON soma_targets
    FOR EACH ROW EXECUTE FUNCTION notify_target_filled();

-- New checkpoint
CREATE OR REPLACE FUNCTION notify_new_checkpoint() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_checkpoint', json_build_object(
        'cp_sequence_number', NEW.cp_sequence_number,
        'tx_lo', NEW.tx_lo,
        'epoch', NEW.epoch
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER cp_sequence_numbers_notify
    AFTER INSERT ON cp_sequence_numbers
    FOR EACH ROW EXECUTE FUNCTION notify_new_checkpoint();

-- New epoch (fires when kv_epoch_starts gets a new row)
CREATE OR REPLACE FUNCTION notify_new_epoch() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_epoch', json_build_object(
        'epoch', NEW.epoch,
        'start_timestamp_ms', NEW.start_timestamp_ms,
        'protocol_version', NEW.protocol_version
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER kv_epoch_starts_notify
    AFTER INSERT ON kv_epoch_starts
    FOR EACH ROW EXECUTE FUNCTION notify_new_epoch();
