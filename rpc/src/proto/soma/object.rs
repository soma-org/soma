use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

//
// Object
//

pub const PACKAGE_TYPE: &str = "package";

impl From<crate::types::Object> for Object {
    fn from(value: crate::types::Object) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<&Object> for Object {
    fn merge(&mut self, source: &Object, mask: &FieldMaskTree) {
        let Object {
            bcs,
            object_id,
            version,
            digest,
            owner,
            object_type,
            contents,
            previous_transaction,
            json,
            balance,
        } = source;

        if mask.contains(Self::BCS_FIELD.name) {
            self.bcs = bcs.clone();
        }

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

        if mask.contains(Self::PACKAGE_FIELD.name) {
            self.package = package.clone();
        }

        if mask.contains(Self::JSON_FIELD.name) {
            self.json = json.clone();
        }

        if mask.contains(Self::BALANCE_FIELD) {
            self.balance = *balance;
        }
    }
}

impl Merge<crate::types::Object> for Object {
    fn merge(&mut self, source: crate::types::Object, mask: &FieldMaskTree) {
        if mask.contains(Self::BCS_FIELD.name) {
            let mut bcs = Bcs::serialize(&source).unwrap();
            bcs.name = Some("Object".to_owned());
            self.bcs = Some(bcs);
        }

        if mask.contains(Self::DIGEST_FIELD.name) {
            self.digest = Some(source.digest().to_string());
        }

        if mask.contains(Self::OBJECT_ID_FIELD.name) {
            self.object_id = Some(source.object_id().to_string());
        }

        if mask.contains(Self::VERSION_FIELD.name) {
            self.version = Some(source.version());
        }

        if mask.contains(Self::OWNER_FIELD.name) {
            self.owner = Some(source.owner().to_owned().into());
        }

        if mask.contains(Self::PREVIOUS_TRANSACTION_FIELD.name) {
            self.previous_transaction = Some(source.previous_transaction().to_string());
        }

        match source.data() {
            crate::types::ObjectData::Struct(move_struct) => {
                self.merge(move_struct, mask);
            }
            crate::types::ObjectData::Package(move_package) => {
                self.merge(move_package, mask);
            }
        }
    }
}

impl Merge<&crate::types::MoveStruct> for Object {
    fn merge(&mut self, source: &crate::types::MoveStruct, mask: &FieldMaskTree) {
        if mask.contains(Self::OBJECT_TYPE_FIELD.name) {
            self.object_type = Some(source.object_type().to_string());
        }

        if mask.contains(Self::HAS_PUBLIC_TRANSFER_FIELD.name) {
            self.has_public_transfer = Some(source.has_public_transfer());
        }

        if mask.contains(Self::CONTENTS_FIELD.name) {
            self.contents = Some(Bcs {
                name: Some(source.object_type().to_string()),
                value: Some(source.contents().to_vec().into()),
            });
        }
    }
}

impl Merge<&crate::types::MovePackage> for Object {
    fn merge(&mut self, source: &crate::types::MovePackage, mask: &FieldMaskTree) {
        if mask.contains(Self::OBJECT_TYPE_FIELD.name) {
            self.object_type = Some(PACKAGE_TYPE.to_owned());
        }

        if mask.contains(Self::PACKAGE_FIELD.name) {
            self.package = Some(Package {
                modules: source
                    .modules
                    .iter()
                    .map(|(name, contents)| Module {
                        name: Some(name.to_string()),
                        contents: Some(contents.clone().into()),
                        ..Default::default()
                    })
                    .collect(),
                type_origins: source
                    .type_origin_table
                    .clone()
                    .into_iter()
                    .map(Into::into)
                    .collect(),
                linkage: source
                    .linkage_table
                    .iter()
                    .map(
                        |(
                            original_id,
                            crate::types::UpgradeInfo {
                                upgraded_id,
                                upgraded_version,
                            },
                        )| {
                            Linkage {
                                original_id: Some(original_id.to_string()),
                                upgraded_id: Some(upgraded_id.to_string()),
                                upgraded_version: Some(*upgraded_version),
                            }
                        },
                    )
                    .collect(),

                ..Default::default()
            })
        }
    }
}

#[allow(clippy::result_large_err)]
fn try_extract_struct(value: &Object) -> Result<crate::types::MoveStruct, TryFromProtoError> {
    let version = value
        .version
        .ok_or_else(|| TryFromProtoError::missing("version"))?;

    let object_type = value
        .object_type()
        .parse()
        .map_err(|e| TryFromProtoError::invalid(Object::OBJECT_TYPE_FIELD, e))?;

    let has_public_transfer = value
        .has_public_transfer
        .ok_or_else(|| TryFromProtoError::missing("has_public_transfer"))?;
    let contents = value
        .contents
        .as_ref()
        .ok_or_else(|| TryFromProtoError::missing("contents"))?
        .value()
        .to_vec();

    crate::types::MoveStruct::new(object_type, has_public_transfer, version, contents).ok_or_else(
        || TryFromProtoError::invalid(Object::CONTENTS_FIELD, "contents missing object_id"),
    )
}

impl TryFrom<&Object> for crate::types::Object {
    type Error = TryFromProtoError;

    fn try_from(value: &Object) -> Result<Self, Self::Error> {
        let owner = value
            .owner
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("owner"))?
            .try_into()?;

        let previous_transaction = value
            .previous_transaction
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("previous_transaction"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(Object::PREVIOUS_TRANSACTION_FIELD, e))?;

        let object_data = if value.object_type() == PACKAGE_TYPE {
            // Package
            crate::types::ObjectData::Package(try_extract_package(value)?)
        } else {
            // Struct
            crate::types::ObjectData::Struct(try_extract_struct(value)?)
        };

        Ok(Self::new(object_data, owner, previous_transaction))
    }
}

//
// GenesisObject
//

impl From<crate::types::GenesisObject> for Object {
    fn from(value: crate::types::GenesisObject) -> Self {
        let mut message = Self {
            object_id: Some(value.object_id().to_string()),
            version: Some(value.version()),
            owner: Some(value.owner().to_owned().into()),
            ..Default::default()
        };

        match value.data() {
            crate::types::ObjectData::Struct(move_struct) => {
                message.merge(move_struct, &FieldMaskTree::new_wildcard());
            }
            crate::types::ObjectData::Package(move_package) => {
                message.merge(move_package, &FieldMaskTree::new_wildcard());
            }
        }

        message
    }
}

impl TryFrom<&Object> for crate::types::GenesisObject {
    type Error = TryFromProtoError;

    fn try_from(value: &Object) -> Result<Self, Self::Error> {
        let object_data = if value.object_type() == PACKAGE_TYPE {
            // Package
            crate::types::ObjectData::Package(try_extract_package(value)?)
        } else {
            // Struct
            crate::types::ObjectData::Struct(try_extract_struct(value)?)
        };

        let owner = value
            .owner
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("owner"))?
            .try_into()?;

        Ok(Self::new(object_data, owner))
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

        let version = value
            .version
            .ok_or_else(|| TryFromProtoError::missing("version"))?;

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
                return Err(TryFromProtoError::invalid(
                    Owner::KIND_FIELD,
                    "unknown OwnerKind",
                ));
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
