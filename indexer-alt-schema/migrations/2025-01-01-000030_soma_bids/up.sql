CREATE TABLE soma_bids (
    bid_id                  BYTEA   NOT NULL,
    cp_sequence_number      BIGINT  NOT NULL,
    ask_id                  BYTEA   NOT NULL,
    seller                  BYTEA   NOT NULL,
    price                   BIGINT  NOT NULL,
    response_digest         BYTEA   NOT NULL,
    created_at_ms           BIGINT  NOT NULL,
    status                  TEXT    NOT NULL,
    PRIMARY KEY (bid_id, cp_sequence_number)
);

CREATE INDEX idx_bids_seller ON soma_bids (seller);
CREATE INDEX idx_bids_ask_id ON soma_bids (ask_id);
CREATE INDEX idx_bids_status ON soma_bids (status);
