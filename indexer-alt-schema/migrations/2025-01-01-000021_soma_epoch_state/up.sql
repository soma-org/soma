CREATE TABLE soma_epoch_state (
    epoch                           BIGINT PRIMARY KEY,
    emission_balance                BIGINT NOT NULL,
    emission_per_epoch              BIGINT NOT NULL,
    distance_threshold              FLOAT8 NOT NULL,
    targets_generated_this_epoch    BIGINT NOT NULL,
    hits_this_epoch                 BIGINT NOT NULL,
    hits_ema                        BIGINT NOT NULL,
    reward_per_target               BIGINT NOT NULL,
    safe_mode                       BOOL   NOT NULL,
    safe_mode_accumulated_fees      BIGINT NOT NULL,
    safe_mode_accumulated_emissions BIGINT NOT NULL
);
