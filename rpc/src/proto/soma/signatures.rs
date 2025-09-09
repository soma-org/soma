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
// CircomG1
//

impl From<crate::types::CircomG1> for CircomG1 {
    fn from(value: crate::types::CircomG1) -> Self {
        let [e0, e1, e2] = value.0;

        Self {
            e0: Some(e0.to_string()),
            e1: Some(e1.to_string()),
            e2: Some(e2.to_string()),
        }
    }
}

impl TryFrom<&CircomG1> for crate::types::CircomG1 {
    type Error = TryFromProtoError;

    fn try_from(value: &CircomG1) -> Result<Self, Self::Error> {
        let e0 = value
            .e0
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e0"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG1::E0_FIELD, e))?;
        let e1 = value
            .e1
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e1"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG1::E1_FIELD, e))?;
        let e2 = value
            .e2
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e2"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG1::E2_FIELD, e))?;

        Ok(Self([e0, e1, e2]))
    }
}

//
// CircomG2
//

impl From<crate::types::CircomG2> for CircomG2 {
    fn from(value: crate::types::CircomG2) -> Self {
        let [[e00, e01], [e10, e11], [e20, e21]] = value.0;

        Self {
            e00: Some(e00.to_string()),
            e01: Some(e01.to_string()),
            e10: Some(e10.to_string()),
            e11: Some(e11.to_string()),
            e20: Some(e20.to_string()),
            e21: Some(e21.to_string()),
        }
    }
}

impl TryFrom<&CircomG2> for crate::types::CircomG2 {
    type Error = TryFromProtoError;

    fn try_from(value: &CircomG2) -> Result<Self, Self::Error> {
        let e00 = value
            .e00
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e00"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E00_FIELD, e))?;
        let e01 = value
            .e01
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e01"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E01_FIELD, e))?;

        let e10 = value
            .e10
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e10"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E10_FIELD, e))?;
        let e11 = value
            .e11
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e11"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E11_FIELD, e))?;

        let e20 = value
            .e20
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e20"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E20_FIELD, e))?;
        let e21 = value
            .e21
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("e21"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(CircomG2::E21_FIELD, e))?;

        Ok(Self([[e00, e01], [e10, e11], [e20, e21]]))
    }
}

//
// ZkLoginClaim
//

impl From<crate::types::ZkLoginClaim> for ZkLoginClaim {
    fn from(
        crate::types::ZkLoginClaim { value, index_mod_4 }: crate::types::ZkLoginClaim,
    ) -> Self {
        Self {
            value: Some(value),
            index_mod_4: Some(index_mod_4.into()),
        }
    }
}

impl TryFrom<&ZkLoginClaim> for crate::types::ZkLoginClaim {
    type Error = TryFromProtoError;

    fn try_from(ZkLoginClaim { value, index_mod_4 }: &ZkLoginClaim) -> Result<Self, Self::Error> {
        let value = value
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("value"))?
            .into();
        let index_mod_4 = index_mod_4
            .ok_or_else(|| TryFromProtoError::missing("index_mod_4"))?
            .try_into()
            .map_err(|e| TryFromProtoError::invalid(ZkLoginClaim::INDEX_MOD_4_FIELD, e))?;

        Ok(Self { value, index_mod_4 })
    }
}

//
// ZkLoginProof
//

impl From<crate::types::ZkLoginProof> for ZkLoginProof {
    fn from(value: crate::types::ZkLoginProof) -> Self {
        Self {
            a: Some(value.a.into()),
            b: Some(value.b.into()),
            c: Some(value.c.into()),
        }
    }
}

impl TryFrom<&ZkLoginProof> for crate::types::ZkLoginProof {
    type Error = TryFromProtoError;

    fn try_from(value: &ZkLoginProof) -> Result<Self, Self::Error> {
        let a = value
            .a
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("a"))?
            .try_into()?;
        let b = value
            .b
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("b"))?
            .try_into()?;
        let c = value
            .c
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("c"))?
            .try_into()?;

        Ok(Self { a, b, c })
    }
}

//
// ZkLoginInputs
//

impl From<crate::types::ZkLoginInputs> for ZkLoginInputs {
    fn from(value: crate::types::ZkLoginInputs) -> Self {
        Self {
            proof_points: Some(value.proof_points().clone().into()),
            iss_base64_details: Some(value.iss_base64_details().clone().into()),
            header_base64: Some(value.header_base64().into()),
            address_seed: Some(value.address_seed().to_string()),
        }
    }
}

impl TryFrom<&ZkLoginInputs> for crate::types::ZkLoginInputs {
    type Error = TryFromProtoError;

    fn try_from(value: &ZkLoginInputs) -> Result<Self, Self::Error> {
        let proof_points = value
            .proof_points
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("proof_points"))?
            .try_into()?;
        let iss_base64_details = value
            .iss_base64_details
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("iss_base64_details"))?
            .try_into()?;
        let header_base64 = value
            .header_base64
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("header_base64"))?
            .into();
        let address_seed = value
            .address_seed
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("address_seed"))?
            .parse()
            .map_err(|e| TryFromProtoError::invalid(ZkLoginInputs::ADDRESS_SEED_FIELD, e))?;

        Self::new(
            proof_points,
            iss_base64_details,
            header_base64,
            address_seed,
        )
        .map_err(|e| TryFromProtoError::invalid("inputs", e))
    }
}

//
// ZkLoginAuthenticator
//

impl From<crate::types::ZkLoginAuthenticator> for ZkLoginAuthenticator {
    fn from(value: crate::types::ZkLoginAuthenticator) -> Self {
        Self {
            inputs: Some(value.inputs.into()),
            max_epoch: Some(value.max_epoch),
            signature: Some(value.signature.into()),
        }
    }
}

impl TryFrom<&ZkLoginAuthenticator> for crate::types::ZkLoginAuthenticator {
    type Error = TryFromProtoError;

    fn try_from(value: &ZkLoginAuthenticator) -> Result<Self, Self::Error> {
        let inputs = value
            .inputs
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("inputs"))?
            .try_into()?;
        let max_epoch = value
            .max_epoch
            .ok_or_else(|| TryFromProtoError::missing("max_epoch"))?;
        let signature = value
            .signature
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("signature"))?
            .try_into()?;

        Ok(Self {
            inputs,
            max_epoch,
            signature,
        })
    }
}

//
// ZkLoginPublicIdentifier
//

impl From<&crate::types::ZkLoginPublicIdentifier> for ZkLoginPublicIdentifier {
    fn from(value: &crate::types::ZkLoginPublicIdentifier) -> Self {
        Self {
            iss: Some(value.iss().to_owned()),
            address_seed: Some(value.address_seed().to_string()),
        }
    }
}

impl TryFrom<&ZkLoginPublicIdentifier> for crate::types::ZkLoginPublicIdentifier {
    type Error = TryFromProtoError;

    fn try_from(value: &ZkLoginPublicIdentifier) -> Result<Self, Self::Error> {
        let iss = value
            .iss
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("iss"))?
            .into();
        let address_seed = value
            .address_seed
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("address_seed"))?
            .parse()
            .map_err(|e| {
                TryFromProtoError::invalid(ZkLoginPublicIdentifier::ADDRESS_SEED_FIELD, e)
            })?;

        Self::new(iss, address_seed)
            .ok_or_else(|| {
                TryFromProtoError::invalid(ZkLoginPublicIdentifier::ISS_FIELD, "invalid iss")
            })?
            .pipe(Ok)
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
            Secp256k1 => Self::Secp256k1,
            Secp256r1 => Self::Secp256r1,
            Multisig => Self::Multisig,
            Bls12381 => Self::Bls12381,
            Zklogin => Self::ZkLogin,
            Passkey => Self::Passkey,
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
            crate::types::SimpleSignature::Secp256k1 {
                signature,
                public_key,
            } => (signature.as_bytes(), public_key.as_bytes()),
            crate::types::SimpleSignature::Secp256r1 {
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
        use SignatureScheme;
        use crate::types::Ed25519PublicKey;
        use crate::types::Ed25519Signature;
        use crate::types::Secp256k1PublicKey;
        use crate::types::Secp256k1Signature;
        use crate::types::Secp256r1PublicKey;
        use crate::types::Secp256r1Signature;

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
            SignatureScheme::Secp256k1 => Self::Secp256k1 {
                signature: Secp256k1Signature::from_bytes(signature)
                    .map_err(|e| TryFromProtoError::invalid(SimpleSignature::SIGNATURE_FIELD, e))?,
                public_key: Secp256k1PublicKey::from_bytes(public_key).map_err(|e| {
                    TryFromProtoError::invalid(SimpleSignature::PUBLIC_KEY_FIELD, e)
                })?,
            },
            SignatureScheme::Secp256r1 => Self::Secp256r1 {
                signature: Secp256r1Signature::from_bytes(signature)
                    .map_err(|e| TryFromProtoError::invalid(SimpleSignature::SIGNATURE_FIELD, e))?,
                public_key: Secp256r1PublicKey::from_bytes(public_key).map_err(|e| {
                    TryFromProtoError::invalid(SimpleSignature::PUBLIC_KEY_FIELD, e)
                })?,
            },
            SignatureScheme::Multisig
            | SignatureScheme::Bls12381
            | SignatureScheme::Zklogin
            | SignatureScheme::Passkey => {
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
// PasskeyAuthenticator
//

impl From<crate::types::PasskeyAuthenticator> for PasskeyAuthenticator {
    fn from(value: crate::types::PasskeyAuthenticator) -> Self {
        Self {
            authenticator_data: Some(value.authenticator_data().to_vec().into()),
            client_data_json: Some(value.client_data_json().to_owned()),
            signature: Some(value.signature().into()),
        }
    }
}

impl TryFrom<&PasskeyAuthenticator> for crate::types::PasskeyAuthenticator {
    type Error = TryFromProtoError;

    fn try_from(value: &PasskeyAuthenticator) -> Result<Self, Self::Error> {
        let authenticator_data = value
            .authenticator_data
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("authenticator_data"))?
            .to_vec();
        let client_data_json = value
            .client_data_json
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("client_data_json"))?
            .into();

        let signature = value
            .signature
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("signature"))?
            .try_into()?;

        Self::new(authenticator_data, client_data_json, signature).ok_or_else(|| {
            TryFromProtoError::invalid(
                PasskeyAuthenticator::CLIENT_DATA_JSON_FIELD,
                "invalid passkey",
            )
        })
    }
}

//
// MultisigMemberPublicKey
//

impl From<&crate::types::MultisigMemberPublicKey> for MultisigMemberPublicKey {
    fn from(value: &crate::types::MultisigMemberPublicKey) -> Self {
        use crate::types::MultisigMemberPublicKey::*;

        let mut message = Self::default();

        let scheme = match value {
            Ed25519(public_key) => {
                message.public_key = Some(public_key.as_bytes().to_vec().into());
                SignatureScheme::Ed25519
            }
            Secp256k1(public_key) => {
                message.public_key = Some(public_key.as_bytes().to_vec().into());
                SignatureScheme::Secp256k1
            }
            Secp256r1(public_key) => {
                message.public_key = Some(public_key.as_bytes().to_vec().into());
                SignatureScheme::Secp256r1
            }
            ZkLogin(zklogin_id) => {
                message.zklogin = Some(zklogin_id.into());
                SignatureScheme::Zklogin
            }
            Passkey(public_key) => {
                message.public_key = Some(public_key.inner().as_bytes().to_vec().into());
                SignatureScheme::Passkey
            }
            _ => return Self::default(),
        };

        message.set_scheme(scheme);
        message
    }
}

impl TryFrom<&MultisigMemberPublicKey> for crate::types::MultisigMemberPublicKey {
    type Error = TryFromProtoError;

    fn try_from(value: &MultisigMemberPublicKey) -> Result<Self, Self::Error> {
        use crate::types::Ed25519PublicKey;
        use crate::types::Secp256k1PublicKey;
        use crate::types::Secp256r1PublicKey;

        match value.scheme() {
            SignatureScheme::Ed25519 => Self::Ed25519(
                Ed25519PublicKey::from_bytes(value.public_key()).map_err(|e| {
                    TryFromProtoError::invalid(MultisigMemberPublicKey::PUBLIC_KEY_FIELD, e)
                })?,
            ),
            SignatureScheme::Secp256k1 => {
                Self::Secp256k1(Secp256k1PublicKey::from_bytes(value.public_key()).map_err(
                    |e| TryFromProtoError::invalid(MultisigMemberPublicKey::PUBLIC_KEY_FIELD, e),
                )?)
            }
            SignatureScheme::Secp256r1 => {
                Self::Secp256r1(Secp256r1PublicKey::from_bytes(value.public_key()).map_err(
                    |e| TryFromProtoError::invalid(MultisigMemberPublicKey::PUBLIC_KEY_FIELD, e),
                )?)
            }
            SignatureScheme::Zklogin => Self::ZkLogin(
                value
                    .zklogin
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("zklogin"))?
                    .try_into()?,
            ),
            SignatureScheme::Passkey => Self::Passkey(crate::types::PasskeyPublicKey::new(
                Secp256r1PublicKey::from_bytes(value.public_key()).map_err(|e| {
                    TryFromProtoError::invalid(MultisigMemberPublicKey::PUBLIC_KEY_FIELD, e)
                })?,
            )),
            SignatureScheme::Multisig | SignatureScheme::Bls12381 => {
                return Err(TryFromProtoError::invalid(
                    MultisigMemberPublicKey::SCHEME_FIELD,
                    "invalid MultisigMemberPublicKey scheme",
                ));
            }
        }
        .pipe(Ok)
    }
}

//
// MultisigMember
//

impl From<&crate::types::MultisigMember> for MultisigMember {
    fn from(value: &crate::types::MultisigMember) -> Self {
        Self {
            public_key: Some(value.public_key().into()),
            weight: Some(value.weight().into()),
        }
    }
}

impl TryFrom<&MultisigMember> for crate::types::MultisigMember {
    type Error = TryFromProtoError;

    fn try_from(value: &MultisigMember) -> Result<Self, Self::Error> {
        let public_key = value
            .public_key
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("public_key"))?
            .try_into()?;
        let weight = value
            .weight
            .ok_or_else(|| TryFromProtoError::missing("weight"))?
            .try_into()
            .map_err(|e| TryFromProtoError::invalid(MultisigMember::WEIGHT_FIELD, e))?;

        Ok(Self::new(public_key, weight))
    }
}

//
// MultisigCommittee
//

impl From<&crate::types::MultisigCommittee> for MultisigCommittee {
    fn from(value: &crate::types::MultisigCommittee) -> Self {
        Self {
            members: value.members().iter().map(Into::into).collect(),
            threshold: Some(value.threshold().into()),
        }
    }
}

impl TryFrom<&MultisigCommittee> for crate::types::MultisigCommittee {
    type Error = TryFromProtoError;

    fn try_from(value: &MultisigCommittee) -> Result<Self, Self::Error> {
        let members = value
            .members
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;
        let threshold = value
            .threshold
            .ok_or_else(|| TryFromProtoError::missing("threshold"))?
            .try_into()
            .map_err(|e| TryFromProtoError::invalid(MultisigCommittee::THRESHOLD_FIELD, e))?;

        Ok(Self::new(members, threshold))
    }
}

//
// MultisigMemberSignature
//

impl From<&crate::types::MultisigMemberSignature> for MultisigMemberSignature {
    fn from(value: &crate::types::MultisigMemberSignature) -> Self {
        use crate::types::MultisigMemberSignature::*;

        let mut message = Self::default();

        let scheme = match value {
            Ed25519(signature) => {
                message.signature = Some(signature.as_bytes().to_vec().into());
                SignatureScheme::Ed25519
            }
            Secp256k1(signature) => {
                message.signature = Some(signature.as_bytes().to_vec().into());
                SignatureScheme::Secp256k1
            }
            Secp256r1(signature) => {
                message.signature = Some(signature.as_bytes().to_vec().into());
                SignatureScheme::Secp256r1
            }
            ZkLogin(zklogin_id) => {
                message.zklogin = Some((**zklogin_id).clone().into());
                SignatureScheme::Zklogin
            }
            Passkey(p) => {
                message.passkey = Some(p.clone().into());
                SignatureScheme::Passkey
            }
            _ => return Self::default(),
        };

        message.set_scheme(scheme);
        message
    }
}

impl TryFrom<&MultisigMemberSignature> for crate::types::MultisigMemberSignature {
    type Error = TryFromProtoError;

    fn try_from(value: &MultisigMemberSignature) -> Result<Self, Self::Error> {
        use crate::types::Ed25519Signature;
        use crate::types::Secp256k1Signature;
        use crate::types::Secp256r1Signature;

        match value.scheme() {
            SignatureScheme::Ed25519 => Self::Ed25519(
                Ed25519Signature::from_bytes(value.signature()).map_err(|e| {
                    TryFromProtoError::invalid(MultisigMemberSignature::SIGNATURE_FIELD, e)
                })?,
            ),
            SignatureScheme::Secp256k1 => Self::Secp256k1(
                Secp256k1Signature::from_bytes(value.signature()).map_err(|e| {
                    TryFromProtoError::invalid(MultisigMemberSignature::SIGNATURE_FIELD, e)
                })?,
            ),
            SignatureScheme::Secp256r1 => Self::Secp256r1(
                Secp256r1Signature::from_bytes(value.signature()).map_err(|e| {
                    TryFromProtoError::invalid(MultisigMemberSignature::SIGNATURE_FIELD, e)
                })?,
            ),
            SignatureScheme::Zklogin => Self::ZkLogin(Box::new(
                value
                    .zklogin
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("zklogin"))?
                    .try_into()?,
            )),
            SignatureScheme::Passkey => Self::Passkey(
                value
                    .passkey
                    .as_ref()
                    .ok_or_else(|| TryFromProtoError::missing("passkey"))?
                    .try_into()?,
            ),
            SignatureScheme::Multisig | SignatureScheme::Bls12381 => {
                return Err(TryFromProtoError::invalid(
                    MultisigMemberSignature::SCHEME_FIELD,
                    "invalid MultisigMemberSignature scheme",
                ));
            }
        }
        .pipe(Ok)
    }
}

//
// MultisigAggregatedSignature
//

impl From<&crate::types::MultisigAggregatedSignature> for MultisigAggregatedSignature {
    fn from(value: &crate::types::MultisigAggregatedSignature) -> Self {
        Self {
            signatures: value.signatures().iter().map(Into::into).collect(),
            bitmap: Some(value.bitmap().into()),
            legacy_bitmap: value
                .legacy_bitmap()
                .map(|roaring| roaring.iter().collect())
                .unwrap_or_default(),
            committee: Some(value.committee().into()),
        }
    }
}

impl TryFrom<&MultisigAggregatedSignature> for crate::types::MultisigAggregatedSignature {
    type Error = TryFromProtoError;

    fn try_from(value: &MultisigAggregatedSignature) -> Result<Self, Self::Error> {
        let signatures = value
            .signatures
            .iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;
        let bitmap = value
            .bitmap
            .ok_or_else(|| TryFromProtoError::missing("bitmap"))?
            .try_into()
            .map_err(|e| {
                TryFromProtoError::invalid(MultisigAggregatedSignature::BITMAP_FIELD, e)
            })?;
        let committee = value
            .committee
            .as_ref()
            .ok_or_else(|| TryFromProtoError::missing("committee"))?
            .try_into()?;

        let mut signature = Self::new(committee, signatures, bitmap);

        if !value.legacy_bitmap.is_empty() {
            let legacy_bitmap = value
                .legacy_bitmap
                .iter()
                .copied()
                .collect::<crate::types::Bitmap>();
            signature.with_legacy_bitmap(legacy_bitmap);
        }

        Ok(signature)
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
            Multisig(ref multisig) => {
                if mask.contains(Self::MULTISIG_FIELD.name) {
                    self.signature = Some(Signature::Multisig(multisig.into()));
                }
            }
            ZkLogin(zklogin) => {
                if mask.contains(Self::ZKLOGIN_FIELD.name) {
                    self.signature = Some(Signature::Zklogin((*zklogin).into()));
                }
            }
            Passkey(passkey) => {
                if mask.contains(Self::PASSKEY_FIELD.name) {
                    self.signature = Some(Signature::Passkey(passkey.into()));
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
            || matches!(signature, Some(Signature::Multisig(_)))
                && mask.contains(Self::MULTISIG_FIELD.name)
            || matches!(signature, Some(Signature::Zklogin(_)))
                && mask.contains(Self::ZKLOGIN_FIELD.name)
            || matches!(signature, Some(Signature::Passkey(_)))
                && mask.contains(Self::PASSKEY_FIELD.name)
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
            Some(Signature::Multisig(multisig)) => Self::Multisig(multisig.try_into()?),
            Some(Signature::Zklogin(zklogin)) => Self::ZkLogin(Box::new(zklogin.try_into()?)),
            Some(Signature::Passkey(passkey)) => Self::Passkey(passkey.try_into()?),
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
