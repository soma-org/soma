use crate::{
    base::SomaAddress,
    object::{Object, ObjectID, ObjectRef, ObjectType, Version},
};

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

    // TODO: Remove the following function after RPC + Indexer is implemented
    fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectRef>>;
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

    fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectRef>> {
        let mut result = Vec::new();

        for object in self.iter() {
            // Check if this is a coin owned by the specified address
            if let Some(owner_address) = object.get_single_owner() {
                if owner_address == address && *object.data.object_type() == ObjectType::Coin {
                    let obj_ref = object.compute_object_reference();
                    result.push(obj_ref);

                    // Apply limit if specified
                    if let Some(lim) = limit {
                        if result.len() >= lim {
                            break;
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

impl<T: ObjectStore + ?Sized> ObjectStore for &T {
    fn get_object(&self, object_id: &ObjectID) -> Result<Option<Object>> {
        (*self).get_object(object_id)
    }

    fn get_object_by_key(&self, object_id: &ObjectID, version: Version) -> Result<Option<Object>> {
        (*self).get_object_by_key(object_id, version)
    }

    fn multi_get_objects(&self, object_ids: &[ObjectID]) -> Result<Vec<Option<Object>>> {
        (*self).multi_get_objects(object_ids)
    }

    fn multi_get_objects_by_key(&self, object_keys: &[ObjectKey]) -> Result<Vec<Option<Object>>> {
        (*self).multi_get_objects_by_key(object_keys)
    }

    fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> Result<Vec<ObjectRef>> {
        // Delegate to the inner implementation
        (*self).get_gas_objects_owned_by_address(address, limit)
    }
}
