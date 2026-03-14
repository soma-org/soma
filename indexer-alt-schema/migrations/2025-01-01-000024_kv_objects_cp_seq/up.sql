-- Add cp_sequence_number to kv_objects for checkpoint-range-based pruning.
ALTER TABLE kv_objects ADD COLUMN cp_sequence_number BIGINT NOT NULL DEFAULT 0;
CREATE INDEX kv_objects_cp_seq ON kv_objects (cp_sequence_number);
