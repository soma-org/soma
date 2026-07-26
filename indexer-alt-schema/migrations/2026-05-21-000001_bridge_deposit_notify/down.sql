-- Copyright (c) Soma Contributors
-- SPDX-License-Identifier: Apache-2.0

DROP TRIGGER IF EXISTS soma_bridge_deposits_notify ON soma_bridge_deposits;
DROP FUNCTION IF EXISTS notify_new_bridge_deposit();
