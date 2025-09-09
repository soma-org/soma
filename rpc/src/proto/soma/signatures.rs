use super::*;
use crate::proto::TryFromProtoError;
use crate::utils::field::FieldMaskTree;
use crate::utils::merge::Merge;
use tap::Pipe;

//
// ValidatorAggregatedSignature
//

impl From<crate::types::ValidatorAggregatedSignature> for ValidatorAggregatedSignature {
    fn from(value: crate::types::ValidatorAggregatedSignature) -> Self {
        Self {
            epoch: Some(value.epoch),
            signature: Some(value.signature.as_bytes().to_vec().into()),
            bitmap: value.bitmap.iter().collect(),
        }
    }
}

impl TryFrom<&ValidatorAggregatedSignature> for crate::types::ValidatorAggregatedSignature {
    type Error = TryFromProtoError;

    fn try_from(value: &ValidatorAggregatedSignature) -> Result<Self, Self::Error> {
        let epoch = value
            .epoch
            .ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        let signature = value
            .signature
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("signature"))?
            .as_ref()
            .pipe(crate::types::Bls12381Signature::from_bytes)
            .map_err(|e| {
                TryFromProtoError::invalid(ValidatorAggregatedSignature::SIGNATURE_FIELD, e)
            })?;
        let bitmap = value.bitmap.iter().copied().collect();

        Ok(Self {
            epoch,
            signature,
            bitmap,
        })
    }
}

//
// ValidatorCommitteeMember
//

impl From<crate::types::ValidatorCommitteeMember> for ValidatorCommitteeMember {
    fn from(value: crate::types::ValidatorCommitteeMember) -> Self {
        Self {
            public_key: Some(value.public_key.as_bytes().to_vec().into()),
            weight: Some(value.stake),
        }
    }
}

impl TryFrom<&ValidatorCommitteeMember> for crate::types::ValidatorCommitteeMember {
    type Error = TryFromProtoError;

    fn try_from(
        ValidatorCommitteeMember { public_key, weight }: &ValidatorCommitteeMember,
    ) -> Result<Self, Self::Error> {
        let public_key = public_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("public_key"))?
            .as_ref()
            .pipe(crate::types::Bls12381PublicKey::from_bytes)
            .map_err(|e| {
                TryFromProtoError::invalid(ValidatorCommitteeMember::PUBLIC_KEY_FIELD, e)
            })?;
        let stake = weight.ok_or_else(|| TryFromProtoError::missing("weight"))?;
        Ok(Self { public_key, stake })
    }
}

//
// ValidatorCommittee
//

impl From<crate::types::ValidatorCommittee> for ValidatorCommittee {
    fn from(value: crate::types::ValidatorCommittee) -> Self {
        Self {
            epoch: Some(value.epoch),
            members: value.members.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<&ValidatorCommittee> for crate::types::ValidatorCommittee {
    type Error = TryFromProtoError;

    fn try_from(value: &ValidatorCommittee) -> Result<Self, Self::Error> {
        let epoch = value
            .epoch
            .ok_or_else(|| TryFromProtoError::missing("epoch"))?;
        Ok(Self {
            epoch,
            members: value
                .members
                .iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

//
// SignatureScheme
//

impl TryFrom<&SignatureScheme> for crate::types::SignatureScheme {
    type Error = TryFromProtoError;

    fn try_from(value: &SignatureScheme) -> Result<Self, Self::Error> {
        use SignatureScheme::*;

        match value {
            Ed25519 => Self::Ed25519,
            Bls12381 => Self::Bls12381,
        }
        .pipe(Ok)
    }
}

//
// SimpleSignature
//

impl From<crate::types::SimpleSignature> for SimpleSignature {
    fn from(value: crate::types::SimpleSignature) -> Self {
        let (signature, public_key) = match &value {
            crate::types::SimpleSignature::Ed25519 {
                signature,
                public_key,
            } => (signature.as_bytes(), public_key.as_bytes()),
            _ => return Self::default(),
        };

        Self {
            scheme: Some(value.scheme().to_u8().into()),
            signature: Some(signature.to_vec().into()),
            public_key: Some(public_key.to_vec().into()),
        }
    }
}

impl TryFrom<&SimpleSignature> for crate::types::SimpleSignature {
    type Error = TryFromProtoError;

    fn try_from(value: &SimpleSignature) -> Result<Self, Self::Error> {
        use crate::types::Ed25519PublicKey;
        use crate::types::Ed25519Signature;
        use SignatureScheme;

        let scheme = value
            .scheme
            .ok_or_else(|| TryFromProtoError::missing("scheme"))?
            .pipe(SignatureScheme::try_from)
            .map_err(|e| TryFromProtoError::invalid(SimpleSignature::SCHEME_FIELD, e))?;
        let signature = value
            .signature
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("signature"))?;
        let public_key = value
            .public_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("public_key"))?;

        match scheme {
            SignatureScheme::Ed25519 => Self::Ed25519 {
                signature: Ed25519Signature::from_bytes(signature)
                    .map_err(|e| TryFromProtoError::invalid(SimpleSignature::SIGNATURE_FIELD, e))?,
                public_key: Ed25519PublicKey::from_bytes(public_key).map_err(|e| {
                    TryFromProtoError::invalid(SimpleSignature::PUBLIC_KEY_FIELD, e)
                })?,
            },
            SignatureScheme::Bls12381 => {
                return Err(TryFromProtoError::invalid(
                    SimpleSignature::SCHEME_FIELD,
                    "invalid or unknown signature scheme",
                ));
            }
        }
        .pipe(Ok)
    }
}

//
// UserSignature
//

impl From<crate::types::UserSignature> for UserSignature {
    fn from(value: crate::types::UserSignature) -> Self {
        Self::merge_from(value, &FieldMaskTree::new_wildcard())
    }
}

impl Merge<crate::types::UserSignature> for UserSignature {
    fn merge(&mut self, source: crate::types::UserSignature, mask: &FieldMaskTree) {
        use crate::types::UserSignature::*;
        use user_signature::Signature;

        if mask.contains(Self::BCS_FIELD.name) {
            self.bcs = Some(Bcs {
                name: Some("UserSignatureBytes".to_owned()),
                value: Some(source.to_bytes().into()),
            });
        }

        if mask.contains(Self::SCHEME_FIELD.name) {
            self.scheme = Some(source.scheme().to_u8().into());
        }

        match source {
            Simple(simple) => {
                if mask.contains(Self::SIMPLE_FIELD.name) {
                    self.signature = Some(Signature::Simple(simple.into()));
                }
            }
            _ => {}
        }
    }
}

impl Merge<&UserSignature> for UserSignature {
    fn merge(&mut self, source: &UserSignature, mask: &FieldMaskTree) {
        use user_signature::Signature;

        let UserSignature {
            bcs,
            scheme,
            signature,
        } = source;

        if mask.contains(Self::BCS_FIELD.name) {
            self.bcs = bcs.clone();
        }

        if mask.contains(Self::SCHEME_FIELD.name) {
            self.scheme = *scheme;
        }

        if matches!(signature, Some(Signature::Simple(_))) && mask.contains(Self::SIMPLE_FIELD.name)
        {
            self.signature = signature.clone();
        }
    }
}

impl TryFrom<&UserSignature> for crate::types::UserSignature {
    type Error = TryFromProtoError;

    fn try_from(value: &UserSignature) -> Result<Self, Self::Error> {
        use user_signature::Signature;

        if let Some(bcs) = &value.bcs {
            if let Ok(sig) = Self::from_bytes(bcs.value()) {
                return Ok(sig);
            } else {
                return bcs
                    .deserialize()
                    .map_err(|e| TryFromProtoError::invalid(UserSignature::BCS_FIELD, e));
            }
        }

        let _scheme = value
            .scheme
            .ok_or_else(|| TryFromProtoError::missing("scheme"))?
            .pipe(SignatureScheme::try_from)
            .map_err(|e| TryFromProtoError::invalid(UserSignature::SCHEME_FIELD, e));

        match &value.signature {
            Some(Signature::Simple(simple)) => Self::Simple(simple.try_into()?),
            None => {
                return Err(TryFromProtoError::invalid(
                    "signature",
                    "invalid or unknown signature scheme",
                ));
            }
        }
        .pipe(Ok)
    }
}
