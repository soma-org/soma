DROP INDEX IF EXISTS kv_objects_cp_seq;
ALTER TABLE kv_objects DROP COLUMN IF EXISTS cp_sequence_number;
