CREATE TABLE soma_settlements (
    settlement_id           BYTEA   NOT NULL,
    cp_sequence_number      BIGINT  NOT NULL,
    ask_id                  BYTEA   NOT NULL,
    bid_id                  BYTEA   NOT NULL,
    buyer                   BYTEA   NOT NULL,
    seller                  BYTEA   NOT NULL,
    amount                  BIGINT  NOT NULL,
    task_digest             BYTEA   NOT NULL,
    response_digest         BYTEA   NOT NULL,
    settled_at_ms           BIGINT  NOT NULL,
    seller_rating           TEXT    NOT NULL,
    rating_deadline_ms      BIGINT  NOT NULL,
    PRIMARY KEY (settlement_id, cp_sequence_number)
);

CREATE INDEX idx_settlements_buyer ON soma_settlements (buyer);
CREATE INDEX idx_settlements_seller ON soma_settlements (seller);
CREATE INDEX idx_settlements_task_digest ON soma_settlements (task_digest);
CREATE INDEX idx_settlements_settled_at ON soma_settlements (settled_at_ms);
