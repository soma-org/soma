// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use anyhow::anyhow;
use fastcrypto::encoding::{Encoding, Hex};
use fastcrypto::traits::EncodeDecodeBase64;
use types::crypto::{AuthorityKeyPair, NetworkKeyPair, SomaKeyPair};

/// Write Base64 encoded `flag || privkey` to file.
pub fn write_keypair_to_file<P: AsRef<std::path::Path>>(
    keypair: &SomaKeyPair,
    path: P,
) -> anyhow::Result<()> {
    let contents = keypair.encode_base64();
    std::fs::write(path, contents)?;
    Ok(())
}

/// Write Base64 encoded `privkey` to file.
pub fn write_authority_keypair_to_file<P: AsRef<std::path::Path>>(
    keypair: &AuthorityKeyPair,
    path: P,
) -> anyhow::Result<()> {
    let contents = keypair.encode_base64();
    std::fs::write(path, contents)?;
    Ok(())
}

/// Read from file as Base64 encoded `privkey` and return a AuthorityKeyPair.
pub fn read_authority_keypair_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<AuthorityKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    AuthorityKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}

/// Read from file as Base64 encoded `flag || privkey` and return a SomaKeypair.
pub fn read_keypair_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<SomaKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    SomaKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}

/// Read from file as Base64 encoded `flag || privkey` and return a NetworkKeyPair.
pub fn read_network_keypair_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<NetworkKeyPair> {
    let kp = read_keypair_from_file(path)?;
    match kp {
        SomaKeyPair::Ed25519(kp) => Ok(NetworkKeyPair::new(kp)),
    }
}

/// Read a SomaKeyPair from a file. The content could be any of the following:
/// - Base64 encoded `flag || privkey` for ECDSA key
/// - Base64 encoded `privkey` for Raw key
/// - Bech32 encoded private key prefixed with `somaprivkey`
/// - Hex encoded `privkey` for Raw key
pub fn read_key(path: &PathBuf) -> Result<SomaKeyPair, anyhow::Error> {
    if !path.exists() {
        return Err(anyhow::anyhow!("Key file not found at path: {:?}", path));
    }
    let file_contents = std::fs::read_to_string(path)?;
    let contents = file_contents.as_str().trim();

    // Try base64 encoded SomaKeyPair `flag || privkey`
    if let Ok(key) = SomaKeyPair::decode_base64(contents) {
        return Ok(key);
    }

    // Try Bech32 encoded 33-byte `flag || private key` starting with `somaprivkey`A prefix.
    // This is the format of a private key exported from SOMA Wallet or soma.keystore.
    if let Ok(key) = SomaKeyPair::decode(contents) {
        return Ok(key);
    }

    Err(anyhow!("Error decoding key from {:?}", path))
}
