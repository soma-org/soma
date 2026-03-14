CREATE TABLE soma_target_reports (
    target_id          BYTEA  NOT NULL,
    cp_sequence_number BIGINT NOT NULL,
    reporter           BYTEA  NOT NULL,
    PRIMARY KEY (target_id, cp_sequence_number, reporter)
);

CREATE INDEX soma_target_reports_reporter
    ON soma_target_reports (reporter, cp_sequence_number DESC);
