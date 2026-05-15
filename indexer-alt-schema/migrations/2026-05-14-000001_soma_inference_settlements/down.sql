-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

DROP INDEX IF EXISTS idx_soma_inference_settlements_payer_ts;
DROP INDEX IF EXISTS idx_soma_inference_settlements_payee_ts;
DROP INDEX IF EXISTS idx_soma_inference_settlements_model_ts;
DROP TABLE IF EXISTS soma_inference_settlements;
