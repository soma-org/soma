use super::MultisigAggregatedSignature;
use crate::types::{Ed25519PublicKey, Ed25519Signature};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UserSignature {
    Simple(SimpleSignature),
    Multisig(MultisigAggregatedSignature),
}

impl UserSignature {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        match self {
            UserSignature::Simple(simple) => simple.scheme(),
            UserSignature::Multisig(_) => SignatureScheme::Multisig,
        }
    }
}

impl UserSignature {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            UserSignature::Simple(s) => s.to_bytes(),
            UserSignature::Multisig(m) => m.to_bytes(),
        }
    }

    pub fn to_base64(&self) -> String {
        use base64ct::Encoding;

        base64ct::Base64::encode_string(&self.to_bytes())
    }

    fn from_serialized_bytes<T: AsRef<[u8]>, E: serde::de::Error>(bytes: T) -> Result<Self, E> {
        let bytes = bytes.as_ref();

        let flag = SignatureScheme::from_byte(
            *bytes
                .first()
                .ok_or_else(|| serde::de::Error::custom("missing signature scheme flag"))?,
        )
        .map_err(serde::de::Error::custom)?;
        match flag {
            SignatureScheme::Ed25519 => {
                let simple = SimpleSignature::from_serialized_bytes(bytes)?;
                Ok(Self::Simple(simple))
            }
            SignatureScheme::Multisig => {
                let multisig = MultisigAggregatedSignature::from_serialized_bytes(bytes)?;
                Ok(Self::Multisig(multisig))
            }
            SignatureScheme::Bls12381 => {
                Err(serde::de::Error::custom("bls not supported for user signatures"))
            }
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bcs::Error> {
        Self::from_serialized_bytes(bytes)
    }

    pub fn from_base64(s: &str) -> Result<Self, bcs::Error> {
        use base64ct::Encoding;
        use serde::de::Error;

        let bytes = base64ct::Base64::decode_vec(s).map_err(bcs::Error::custom)?;
        Self::from_bytes(&bytes)
    }
}

#[derive(serde_derive::Serialize)]
#[serde(tag = "scheme", rename_all = "lowercase")]
enum ReadableUserSignatureRef<'a> {
    Ed25519 { signature: &'a Ed25519Signature, public_key: &'a Ed25519PublicKey },
    Multisig(&'a MultisigAggregatedSignature),
}

#[derive(serde_derive::Deserialize)]
#[serde(tag = "scheme", rename_all = "lowercase")]
#[serde(rename = "UserSignature")]
enum ReadableUserSignature {
    Ed25519 { signature: Ed25519Signature, public_key: Ed25519PublicKey },
    Multisig(MultisigAggregatedSignature),
}

impl serde::Serialize for UserSignature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            let readable = match self {
                UserSignature::Simple(SimpleSignature::Ed25519 { signature, public_key }) => {
                    ReadableUserSignatureRef::Ed25519 { signature, public_key }
                }
                UserSignature::Multisig(multisig) => ReadableUserSignatureRef::Multisig(multisig),
            };
            readable.serialize(serializer)
        } else {
            match self {
                UserSignature::Simple(simple) => simple.serialize(serializer),
                UserSignature::Multisig(multisig) => multisig.serialize(serializer),
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for UserSignature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let readable = ReadableUserSignature::deserialize(deserializer)?;
            Ok(match readable {
                ReadableUserSignature::Ed25519 { signature, public_key } => {
                    Self::Simple(SimpleSignature::Ed25519 { signature, public_key })
                }
                ReadableUserSignature::Multisig(multisig) => Self::Multisig(multisig),
            })
        } else {
            use serde_with::DeserializeAs;

            let bytes: std::borrow::Cow<'de, [u8]> =
                serde_with::Bytes::deserialize_as(deserializer)?;
            Self::from_serialized_bytes(bytes)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimpleSignature {
    Ed25519 { signature: Ed25519Signature, public_key: Ed25519PublicKey },
}

impl SimpleSignature {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        match self {
            SimpleSignature::Ed25519 { .. } => SignatureScheme::Ed25519,
        }
    }
}

impl SimpleSignature {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            SimpleSignature::Ed25519 { signature, public_key } => {
                buf.push(SignatureScheme::Ed25519 as u8);
                buf.extend_from_slice(signature.as_ref());
                buf.extend_from_slice(public_key.as_ref());
            }
        }

        buf
    }

    fn from_serialized_bytes<T: AsRef<[u8]>, E: serde::de::Error>(bytes: T) -> Result<Self, E> {
        let bytes = bytes.as_ref();
        let flag = SignatureScheme::from_byte(
            *bytes
                .first()
                .ok_or_else(|| serde::de::Error::custom("missing signature scheme flag"))?,
        )
        .map_err(serde::de::Error::custom)?;
        match flag {
            SignatureScheme::Ed25519 => {
                let expected_length = 1 + Ed25519Signature::LENGTH + Ed25519PublicKey::LENGTH;

                if bytes.len() != expected_length {
                    return Err(serde::de::Error::custom("invalid ed25519 signature"));
                }

                let mut signature = [0; Ed25519Signature::LENGTH];
                signature.copy_from_slice(&bytes[1..(1 + Ed25519Signature::LENGTH)]);

                let mut public_key = [0; Ed25519PublicKey::LENGTH];
                public_key.copy_from_slice(&bytes[(1 + Ed25519Signature::LENGTH)..]);

                Ok(SimpleSignature::Ed25519 {
                    signature: Ed25519Signature::new(signature),
                    public_key: Ed25519PublicKey::new(public_key),
                })
            }
            SignatureScheme::Bls12381 | SignatureScheme::Multisig => {
                Err(serde::de::Error::custom("invalid signature scheme"))
            }
        }
    }
}

impl serde::Serialize for SimpleSignature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde_derive::Serialize)]
        #[serde(tag = "scheme")]
        #[serde(rename_all = "lowercase")]
        enum Sig<'a> {
            Ed25519 { signature: &'a Ed25519Signature, public_key: &'a Ed25519PublicKey },
        }

        if serializer.is_human_readable() {
            let sig = match self {
                SimpleSignature::Ed25519 { signature, public_key } => {
                    Sig::Ed25519 { signature, public_key }
                }
            };

            sig.serialize(serializer)
        } else {
            match self {
                SimpleSignature::Ed25519 { signature, public_key } => {
                    let mut buf = [0; 1 + Ed25519Signature::LENGTH + Ed25519PublicKey::LENGTH];
                    buf[0] = SignatureScheme::Ed25519 as u8;
                    buf[1..(1 + Ed25519Signature::LENGTH)].copy_from_slice(signature.as_ref());
                    buf[(1 + Ed25519Signature::LENGTH)..].copy_from_slice(public_key.as_ref());

                    serializer.serialize_bytes(&buf)
                }
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for SimpleSignature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde_derive::Deserialize)]
        #[serde(tag = "scheme")]
        #[serde(rename_all = "lowercase")]
        enum Sig {
            Ed25519 { signature: Ed25519Signature, public_key: Ed25519PublicKey },
        }

        if deserializer.is_human_readable() {
            let sig = Sig::deserialize(deserializer)?;
            Ok(match sig {
                Sig::Ed25519 { signature, public_key } => {
                    SimpleSignature::Ed25519 { signature, public_key }
                }
            })
        } else {
            let bytes: std::borrow::Cow<'de, [u8]> = std::borrow::Cow::deserialize(deserializer)?;
            Self::from_serialized_bytes(bytes)
        }
    }
}

/// Flag use to disambiguate the signature schemes supported by Sui.
///
/// # BCS
///
/// The BCS serialized form for this type is defined by the following ABNF:
///
/// ```text
/// signature-scheme = ed25519-flag /
///                    multisig-flag / bls-flag
/// ed25519-flag     = %x00
/// multisig-flag    = %x02
/// bls-flag         = %x01
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
#[repr(u8)]
pub enum SignatureScheme {
    Ed25519 = 0x00,
    Bls12381 = 0x01, // This is currently not supported for user addresses
    Multisig = 0x02,
}

impl SignatureScheme {
    /// Return the name of this signature scheme
    pub fn name(self) -> &'static str {
        match self {
            SignatureScheme::Ed25519 => "ed25519",
            SignatureScheme::Bls12381 => "bls12381",
            SignatureScheme::Multisig => "multisig",
        }
    }

    /// Try constructing from a byte flag
    pub fn from_byte(flag: u8) -> Result<Self, InvalidSignatureScheme> {
        match flag {
            0x00 => Ok(Self::Ed25519),
            0x01 => Ok(Self::Bls12381),
            0x02 => Ok(Self::Multisig),
            invalid => Err(InvalidSignatureScheme(invalid)),
        }
    }

    /// Convert to a byte flag
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl Ed25519PublicKey {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        SignatureScheme::Ed25519
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct InvalidSignatureScheme(u8);

impl std::fmt::Display for InvalidSignatureScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid signature scheme: {:02x}", self.0)
    }
}
