use std::str::FromStr;

use super::*;
use crate::{
    proto::TryFromProtoError,
    utils::{field::FieldMaskTree, merge::Merge},
};
use tap::Pipe;

//
// Object
//
// From domain type to protobuf

impl Merge<&Object> for Object {
    fn merge(&mut self, source: &Object, mask: &FieldMaskTree) {
        let Object {
            object_id,
            version,
            digest,
            owner,
            object_type,
            contents,
            previous_transaction,
        } = source;

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = digest.clone();
        }

        if mask.contains(Self::OBJECT_ID_FIELD.name) {
            self.object_id = object_id.clone();
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = *version;
        }

        if mask.contains(Self::OWNER_FIELD.name) {
            self.owner = owner.clone();
        }

        if mask.contains(Self::PREVIOUS_TRANSACTION_FIELD.name) {
            self.previous_transaction = previous_transaction.clone();
        }

        if mask.contains(Self::OBJECT_TYPE_FIELD.name) {
            self.object_type = object_type.clone();
        }

        if mask.contains(Self::CONTENTS_FIELD.name) {
            self.contents = contents.clone();
        }
    }
}

impl Merge<crate::types::Object> for Object {
    fn merge(&mut self, source: crate::types::Object, mask: &FieldMaskTree) {
        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::OBJECT_ID_FIELD.name) {
            self.object_id = Some(source.object_id.to_string());
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = Some(source.version());
        }

        if mask.contains(Self::OWNER_FIELD.name) {
            self.owner = Some((*source.owner()).into());
        }

        if mask.contains(Self::PREVIOUS_TRANSACTION_FIELD.name) {
            self.previous_transaction = Some(source.previous_transaction.to_string());
        }

        if mask.contains(Self::OBJECT_TYPE_FIELD.name) {
            self.object_type = Some(source.object_type.into());
        }

        if mask.contains(Self::CONTENTS_FIELD.name) {
            // Get the contents without the ID prefix
            let contents = source.contents.to_vec();
            self.contents = Some(contents.into());
        }
    }
}

// Also implement From for the initial conversion
impl From<crate::types::Object> for Object {
    fn from(value: crate::types::Object) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

// Helper to create from source with mask
// impl Object {
//     pub fn merge_from(source: crate::types::Object, mask: &FieldMaskTree) -> Self {
//         let mut result = Self::default();
//         result.merge(source, mask);
//         result
//     }

//     pub fn merge_from_types(source: types::object::Object, mask: &FieldMaskTree) -> Self {
//         let mut result = Self::default();
//         result.merge(source, mask);
//         result
//     }
// }

// From protobuf to domain type
impl TryFrom<&Object> for crate::types::Object {
    type Error = TryFromProtoError;

    fn try_from(value: &Object) -> Result<Self, Self::Error> {
        // Otherwise construct from individual fields
        let object_id = value
            .object_id
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("object_id"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("object_id", e))?;

        let version = value.version.ok_or_else(|| TryFromProtoError::missing("version"))?
            as crate::types::Version;

        let object_type = value
            .object_type
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("object_type"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("object_type", e))?;

        let owner =
            value.owner.as_ref().ok_or_else(|| TryFromProtoError::missing("owner"))?.try_into()?;

        let previous_transaction = value
            .previous_transaction
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("previous_transaction"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid("previous_transaction", e))?;

        let contents =
            value.contents.clone().ok_or_else(|| TryFromProtoError::missing("contents"))?.into();

        Ok(crate::types::Object::new(
            object_id,
            version,
            object_type,
            owner,
            previous_transaction,
            contents,
        ))
    }
}

// ObjectType conversions
impl From<crate::types::ObjectType> for String {
    fn from(value: crate::types::ObjectType) -> Self {
        match value {
            crate::types::ObjectType::SystemState => "SystemState".to_string(),
            crate::types::ObjectType::Coin => "Coin".to_string(),
            crate::types::ObjectType::StakedSoma => "StakedSoma".to_string(),
            crate::types::ObjectType::Target => "Target".to_string(),
            crate::types::ObjectType::Submission => "Submission".to_string(),
            crate::types::ObjectType::Challenge => "Challenge".to_string(),
        }
    }
}

impl FromStr for crate::types::ObjectType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SystemState" => Ok(Self::SystemState),
            "Coin" => Ok(Self::Coin),
            "StakedSoma" => Ok(Self::StakedSoma),
            "Target" => Ok(Self::Target),
            "Submission" => Ok(Self::Submission),
            "Challenge" => Ok(Self::Challenge),
            _ => Err(format!("Unknown object type: {}", s)),
        }
    }
}

//
// ObjectReference
//

impl From<crate::types::ObjectReference> for ObjectReference {
    fn from(value: crate::types::ObjectReference) -> Self {
        let (object_id, version, digest) = value.into_parts();
        Self {
            object_id: Some(object_id.to_string()),
            version: Some(version),
            digest: Some(digest.to_string()),
        }
    }
}

impl TryFrom<&ObjectReference> for crate::types::ObjectReference {
    type Error = TryFromProtoError;

    fn try_from(value: &ObjectReference) -> Result<Self, Self::Error> {
        let object_id = value
            .object_id
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("object_id"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(ObjectReference::OBJECT_ID_FIELD, e))?;

        let version = value.version.ok_or_else(|| TryFromProtoError::missing("version"))?;

        let digest = value
            .digest
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("digest"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(ObjectReference::DIGEST_FIELD, e))?;

        Ok(Self::new(object_id, version, digest))
    }
}

//
// Owner
//

impl From<crate::types::Owner> for Owner {
    fn from(value: crate::types::Owner) -> Self {
        use crate::types::Owner::*;
        use owner::OwnerKind;

        let mut message = Self::default();

        let kind = match value {
            Address(address) => {
                message.address = Some(address.to_string());
                OwnerKind::Address
            }
            Shared(version) => {
                message.version = Some(version);
                OwnerKind::Shared
            }
            Immutable => OwnerKind::Immutable,
            _ => OwnerKind::Unknown,
        };

        message.set_kind(kind);
        message
    }
}

impl TryFrom<&Owner> for crate::types::Owner {
    type Error = TryFromProtoError;

    fn try_from(value: &Owner) -> Result<Self, Self::Error> {
        use owner::OwnerKind;

        match value.kind() {
            OwnerKind::Unknown => {
                return Err(TryFromProtoError::invalid(Owner::KIND_FIELD, "unknown OwnerKind"));
            }
            OwnerKind::Address => Self::Address(
                value
                    .address()
                    .parse()
                    .map_err(|e| TryFromProtoError::invalid(Owner::ADDRESS_FIELD, e))?,
            ),

            OwnerKind::Shared => Self::Shared(value.version()),
            OwnerKind::Immutable => Self::Immutable,
        }
        .pipe(Ok)
    }
}

impl Merge<&ObjectSet> for ObjectSet {
    fn merge(&mut self, source: &ObjectSet, mask: &FieldMaskTree) {
        if let Some(submask) = mask.subtree(Self::OBJECTS_FIELD) {
            self.objects = source
                .objects()
                .iter()
                .map(|object| Object::merge_from(object, &submask))
                .collect();
        }
    }
}

impl ObjectSet {
    // Sorts the objects in this set by the key `(object_id, version)`
    #[doc(hidden)]
    pub fn sort_objects(&mut self) {
        self.objects_mut().sort_by(|a, b| {
            let a = (a.object_id(), a.version());
            let b = (b.object_id(), b.version());
            a.cmp(&b)
        });
    }

    // Performs a binary search on the contained object set searching for the specified
    // (object_id, version). This function assumes that both the `object_id` and `version` fields
    // are set for all contained objects.
    pub fn binary_search<'a>(
        &'a self,
        object_id: &crate::types::Address,
        version: u64,
    ) -> Option<&'a Object> {
        let object_id = object_id.to_string();
        let seek = (object_id.as_str(), version);
        self.objects()
            .binary_search_by(|object| {
                let probe = (object.object_id(), object.version());
                probe.cmp(&seek)
            })
            .ok()
            .and_then(|found| self.objects().get(found))
    }
}
