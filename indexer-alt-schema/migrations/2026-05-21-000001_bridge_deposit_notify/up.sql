-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

-- NOTIFY trigger for the `new_bridge_deposit` GraphQL subscription channel.
-- See `notify_new_transaction` in the initial-schema migration for the
-- pattern; the GraphQL server's `spawn_pg_listener` LISTENs on this channel
-- and fans the payload out to per-recipient subscribers.

CREATE OR REPLACE FUNCTION notify_new_bridge_deposit() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('new_bridge_deposit', json_build_object(
        'tx_sequence_number', NEW.tx_sequence_number,
        'cp_sequence_number', NEW.cp_sequence_number,
        'recipient', encode(NEW.recipient, 'hex'),
        'amount', NEW.amount,
        'nonce', NEW.nonce,
        'eth_tx_hash', encode(NEW.eth_tx_hash, 'hex'),
        'timestamp_ms', NEW.timestamp_ms
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER soma_bridge_deposits_notify
    AFTER INSERT ON soma_bridge_deposits
    FOR EACH ROW EXECUTE FUNCTION notify_new_bridge_deposit();
