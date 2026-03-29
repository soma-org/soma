CREATE TABLE soma_epoch_state (
    epoch                           BIGINT PRIMARY KEY,
    emission_balance                BIGINT NOT NULL,
    emission_per_epoch              BIGINT NOT NULL,
    distribution_counter            BIGINT NOT NULL,
    period_length                   BIGINT NOT NULL,
    decrease_rate                   INT    NOT NULL,
    protocol_fund_balance           BIGINT NOT NULL,
    safe_mode                       BOOL   NOT NULL,
    safe_mode_accumulated_fees      BIGINT NOT NULL,
    safe_mode_accumulated_emissions BIGINT NOT NULL
);
