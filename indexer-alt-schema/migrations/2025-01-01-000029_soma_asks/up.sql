CREATE TABLE soma_asks (
    ask_id                  BYTEA   NOT NULL,
    cp_sequence_number      BIGINT  NOT NULL,
    buyer                   BYTEA   NOT NULL,
    task_digest             BYTEA   NOT NULL,
    max_price_per_bid       BIGINT  NOT NULL,
    num_bids_wanted         INTEGER NOT NULL,
    timeout_ms              BIGINT  NOT NULL,
    created_at_ms           BIGINT  NOT NULL,
    status                  TEXT    NOT NULL,
    accepted_bid_count      INTEGER NOT NULL,
    PRIMARY KEY (ask_id, cp_sequence_number)
);

CREATE INDEX idx_asks_buyer ON soma_asks (buyer);
CREATE INDEX idx_asks_status ON soma_asks (status);
CREATE INDEX idx_asks_created_at ON soma_asks (created_at_ms);
