use crate::object::{Object, ObjectID, Version};

use super::{storage_error::Result, ObjectKey};

pub trait ObjectStore {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>>;

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>>;

    fn multi_get_objects(&self, object_ids: &[ObjectID]) -> Result<Vec<Option<Object>>> {
        object_ids
            .iter()
            .map(|digest| self.get_object(digest))
            .collect::<Result<Vec<_>, _>>()
    }

    fn multi_get_objects_by_key(&self, object_keys: &[ObjectKey]) -> Result<Vec<Option<Object>>> {
        object_keys
            .iter()
            .map(|k| self.get_object_by_key(&k.0, k.1))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl ObjectStore for &[Object] {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>> {
        Ok(self.iter().find(|o| o.id() == *object_id).cloned())
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>> {
        Ok(self
            .iter()
            .find(|o| o.id() == *object_id && o.version() == version)
            .cloned())
    }
}
