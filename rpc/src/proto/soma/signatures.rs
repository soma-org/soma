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
            authority_key: Some(value.authority_key.into()),
            weight: Some(value.stake),
            network_metadata: Some(value.network_metadata.into()),
        }
    }
}

impl TryFrom<&ValidatorCommitteeMember> for crate::types::ValidatorCommitteeMember {
    type Error = TryFromProtoError;

    fn try_from(
        ValidatorCommitteeMember {
            authority_key,
            weight,
            network_metadata,
        }: &ValidatorCommitteeMember,
    ) -> Result<Self, Self::Error> {
        let authority_key = authority_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("authority_key"))?
            .as_ref()
            .to_vec();

        let stake = weight.ok_or_else(|| TryFromProtoError::missing("weight"))?;

        let network_metadata = network_metadata
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("network_metadata"))?
            .try_into()?;

        Ok(Self {
            authority_key,
            stake,
            network_metadata,
        })
    }
}

//
// ValidatorNetworkMetadata
//

impl From<crate::types::ValidatorNetworkMetadata> for ValidatorNetworkMetadata {
    fn from(value: crate::types::ValidatorNetworkMetadata) -> Self {
        Self {
            consensus_address: Some(value.consensus_address),
            hostname: Some(value.hostname),
            protocol_key: Some(value.protocol_key.into()),
            network_key: Some(value.network_key.into()),
        }
    }
}

impl TryFrom<&ValidatorNetworkMetadata> for crate::types::ValidatorNetworkMetadata {
    type Error = TryFromProtoError;

    fn try_from(
        ValidatorNetworkMetadata {
            consensus_address,
            hostname,
            protocol_key,
            network_key,
        }: &ValidatorNetworkMetadata,
    ) -> Result<Self, Self::Error> {
        let consensus_address = consensus_address
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("consensus_address"))?
            .clone();

        let hostname = hostname
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("hostname"))?
            .clone();

        let protocol_key = protocol_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("protocol_key"))?
            .as_ref()
            .to_vec();

        let network_key = network_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("network_key"))?
            .as_ref()
            .to_vec();

        Ok(Self {
            consensus_address,
            hostname,
            protocol_key,
            network_key,
        })
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

impl TryFrom<ValidatorCommittee> for crate::types::ValidatorCommittee {
    type Error = TryFromProtoError;

    fn try_from(value: ValidatorCommittee) -> Result<Self, Self::Error> {
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

        let UserSignature { scheme, signature } = source;

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
