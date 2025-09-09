#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UserSignature {
    Simple(SimpleSignature),
}

impl UserSignature {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        match self {
            UserSignature::Simple(simple) => simple.scheme(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "proptest", derive(test_strategy::Arbitrary))]
#[non_exhaustive]
pub enum SimpleSignature {
    Ed25519 {
        signature: Ed25519Signature,
        public_key: Ed25519PublicKey,
    },
    Secp256k1 {
        signature: Secp256k1Signature,
        public_key: Secp256k1PublicKey,
    },
    Secp256r1 {
        signature: Secp256r1Signature,
        public_key: Secp256r1PublicKey,
    },
}

impl SimpleSignature {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        match self {
            SimpleSignature::Ed25519 { .. } => SignatureScheme::Ed25519,
            SimpleSignature::Secp256k1 { .. } => SignatureScheme::Secp256k1,
            SimpleSignature::Secp256r1 { .. } => SignatureScheme::Secp256r1,
        }
    }
}

impl SimpleSignature {
    #[cfg(feature = "serde")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "serde")))]
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            SimpleSignature::Ed25519 {
                signature,
                public_key,
            } => {
                buf.push(SignatureScheme::Ed25519 as u8);
                buf.extend_from_slice(signature.as_ref());
                buf.extend_from_slice(public_key.as_ref());
            }
            SimpleSignature::Secp256k1 {
                signature,
                public_key,
            } => {
                buf.push(SignatureScheme::Secp256k1 as u8);
                buf.extend_from_slice(signature.as_ref());
                buf.extend_from_slice(public_key.as_ref());
            }
            SimpleSignature::Secp256r1 {
                signature,
                public_key,
            } => {
                buf.push(SignatureScheme::Secp256r1 as u8);
                buf.extend_from_slice(signature.as_ref());
                buf.extend_from_slice(public_key.as_ref());
            }
        }

        buf
    }

    #[cfg(feature = "serde")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "serde")))]
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
            SignatureScheme::Secp256k1 => {
                let expected_length = 1 + Secp256k1Signature::LENGTH + Secp256k1PublicKey::LENGTH;

                if bytes.len() != expected_length {
                    return Err(serde::de::Error::custom("invalid secp25k1 signature"));
                }

                let mut signature = [0; Secp256k1Signature::LENGTH];
                signature.copy_from_slice(&bytes[1..(1 + Secp256k1Signature::LENGTH)]);

                let mut public_key = [0; Secp256k1PublicKey::LENGTH];
                public_key.copy_from_slice(&bytes[(1 + Secp256k1Signature::LENGTH)..]);

                Ok(SimpleSignature::Secp256k1 {
                    signature: Secp256k1Signature::new(signature),
                    public_key: Secp256k1PublicKey::new(public_key),
                })
            }
            SignatureScheme::Secp256r1 => {
                let expected_length = 1 + Secp256r1Signature::LENGTH + Secp256r1PublicKey::LENGTH;

                if bytes.len() != expected_length {
                    return Err(serde::de::Error::custom("invalid secp25r1 signature"));
                }

                let mut signature = [0; Secp256r1Signature::LENGTH];
                signature.copy_from_slice(&bytes[1..(1 + Secp256r1Signature::LENGTH)]);

                let mut public_key = [0; Secp256r1PublicKey::LENGTH];
                public_key.copy_from_slice(&bytes[(1 + Secp256r1Signature::LENGTH)..]);

                Ok(SimpleSignature::Secp256r1 {
                    signature: Secp256r1Signature::new(signature),
                    public_key: Secp256r1PublicKey::new(public_key),
                })
            }
            SignatureScheme::Multisig
            | SignatureScheme::Bls12381
            | SignatureScheme::ZkLogin
            | SignatureScheme::Passkey => Err(serde::de::Error::custom("invalid signature scheme")),
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "serde")))]
impl serde::Serialize for SimpleSignature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde_derive::Serialize)]
        #[serde(tag = "scheme")]
        #[serde(rename_all = "lowercase")]
        enum Sig<'a> {
            Ed25519 {
                signature: &'a Ed25519Signature,
                public_key: &'a Ed25519PublicKey,
            },
            Secp256k1 {
                signature: &'a Secp256k1Signature,
                public_key: &'a Secp256k1PublicKey,
            },
            Secp256r1 {
                signature: &'a Secp256r1Signature,
                public_key: &'a Secp256r1PublicKey,
            },
        }

        if serializer.is_human_readable() {
            let sig = match self {
                SimpleSignature::Ed25519 {
                    signature,
                    public_key,
                } => Sig::Ed25519 {
                    signature,
                    public_key,
                },
                SimpleSignature::Secp256k1 {
                    signature,
                    public_key,
                } => Sig::Secp256k1 {
                    signature,
                    public_key,
                },
                SimpleSignature::Secp256r1 {
                    signature,
                    public_key,
                } => Sig::Secp256r1 {
                    signature,
                    public_key,
                },
            };

            sig.serialize(serializer)
        } else {
            match self {
                SimpleSignature::Ed25519 {
                    signature,
                    public_key,
                } => {
                    let mut buf = [0; 1 + Ed25519Signature::LENGTH + Ed25519PublicKey::LENGTH];
                    buf[0] = SignatureScheme::Ed25519 as u8;
                    buf[1..(1 + Ed25519Signature::LENGTH)].copy_from_slice(signature.as_ref());
                    buf[(1 + Ed25519Signature::LENGTH)..].copy_from_slice(public_key.as_ref());

                    serializer.serialize_bytes(&buf)
                }
                SimpleSignature::Secp256k1 {
                    signature,
                    public_key,
                } => {
                    let mut buf = [0; 1 + Secp256k1Signature::LENGTH + Secp256k1PublicKey::LENGTH];
                    buf[0] = SignatureScheme::Secp256k1 as u8;
                    buf[1..(1 + Secp256k1Signature::LENGTH)].copy_from_slice(signature.as_ref());
                    buf[(1 + Secp256k1Signature::LENGTH)..].copy_from_slice(public_key.as_ref());

                    serializer.serialize_bytes(&buf)
                }
                SimpleSignature::Secp256r1 {
                    signature,
                    public_key,
                } => {
                    let mut buf = [0; 1 + Secp256r1Signature::LENGTH + Secp256r1PublicKey::LENGTH];
                    buf[0] = SignatureScheme::Secp256r1 as u8;
                    buf[1..(1 + Secp256r1Signature::LENGTH)].copy_from_slice(signature.as_ref());
                    buf[(1 + Secp256r1Signature::LENGTH)..].copy_from_slice(public_key.as_ref());

                    serializer.serialize_bytes(&buf)
                }
            }
        }
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for SimpleSignature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde_derive::Deserialize)]
        #[serde(tag = "scheme")]
        #[serde(rename_all = "lowercase")]
        enum Sig {
            Ed25519 {
                signature: Ed25519Signature,
                public_key: Ed25519PublicKey,
            },
            Secp256k1 {
                signature: Secp256k1Signature,
                public_key: Secp256k1PublicKey,
            },
            Secp256r1 {
                signature: Secp256r1Signature,
                public_key: Secp256r1PublicKey,
            },
        }

        if deserializer.is_human_readable() {
            let sig = Sig::deserialize(deserializer)?;
            Ok(match sig {
                Sig::Ed25519 {
                    signature,
                    public_key,
                } => SimpleSignature::Ed25519 {
                    signature,
                    public_key,
                },
                Sig::Secp256k1 {
                    signature,
                    public_key,
                } => SimpleSignature::Secp256k1 {
                    signature,
                    public_key,
                },
                Sig::Secp256r1 {
                    signature,
                    public_key,
                } => SimpleSignature::Secp256r1 {
                    signature,
                    public_key,
                },
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
/// signature-scheme = ed25519-flag / secp256k1-flag / secp256r1-flag /
///                    multisig-flag / bls-flag / zklogin-flag / passkey-flag
/// ed25519-flag     = %x00
/// secp256k1-flag   = %x01
/// secp256r1-flag   = %x02
/// multisig-flag    = %x03
/// bls-flag         = %x04
/// zklogin-flag     = %x05
/// passkey-flag     = %x06
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "proptest", derive(test_strategy::Arbitrary))]
#[non_exhaustive]
#[repr(u8)]
pub enum SignatureScheme {
    Ed25519 = 0x00,
    Secp256k1 = 0x01,
    Secp256r1 = 0x02,
    Multisig = 0x03,
    Bls12381 = 0x04, // This is currently not supported for user addresses
    ZkLogin = 0x05,
    Passkey = 0x06,
}

impl SignatureScheme {
    /// Return the name of this signature scheme
    pub fn name(self) -> &'static str {
        match self {
            SignatureScheme::Ed25519 => "ed25519",
            SignatureScheme::Secp256k1 => "secp256k1",
            SignatureScheme::Secp256r1 => "secp256r1",
            SignatureScheme::Multisig => "multisig",
            SignatureScheme::Bls12381 => "bls12381",
            SignatureScheme::ZkLogin => "zklogin",
            SignatureScheme::Passkey => "passkey",
        }
    }

    /// Try constructing from a byte flag
    pub fn from_byte(flag: u8) -> Result<Self, InvalidSignatureScheme> {
        match flag {
            0x00 => Ok(Self::Ed25519),
            0x01 => Ok(Self::Secp256k1),
            0x02 => Ok(Self::Secp256r1),
            0x03 => Ok(Self::Multisig),
            0x04 => Ok(Self::Bls12381),
            0x05 => Ok(Self::ZkLogin),
            0x06 => Ok(Self::Passkey),
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

impl Secp256k1PublicKey {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        SignatureScheme::Secp256k1
    }
}

impl Secp256r1PublicKey {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        SignatureScheme::Secp256r1
    }
}

impl super::ZkLoginPublicIdentifier {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        SignatureScheme::ZkLogin
    }
}

impl super::PasskeyPublicKey {
    /// Return the flag for this signature scheme
    pub fn scheme(&self) -> SignatureScheme {
        SignatureScheme::Passkey
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct InvalidSignatureScheme(u8);

impl std::fmt::Display for InvalidSignatureScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid signature scheme: {:02x}", self.0)
    }
}
