use crate::crypto::NetworkKeyPair;
use certgen::SelfSignedCertificate;
use fastcrypto::{ed25519::Ed25519PublicKey, traits::ToFromBytes};
use rustls::{pki_types::CertificateDer, ClientConfig};
use std::collections::BTreeSet;
use tokio_rustls::rustls::ServerConfig;
use verifier::{Allower, ClientCertVerifier, ServerCertVerifier};

mod acceptor;
mod certgen;
pub mod verifier;

// Checks if the public key from a TLS certificate belongs to one of the validators.
#[derive(Debug)]
pub struct AllowedPublicKeys {
    keys: BTreeSet<Ed25519PublicKey>,
}

impl AllowedPublicKeys {
    pub fn new(keys: BTreeSet<Ed25519PublicKey>) -> Self {
        // let keys = context
        //     .committee
        //     .authorities()
        //     .map(|(_, authority)| authority.network_key.clone().into_inner())
        //     .collect();
        Self { keys }
    }
}

impl Allower for AllowedPublicKeys {
    fn allowed(&self, key: &Ed25519PublicKey) -> bool {
        self.keys.contains(key)
    }
}

pub fn create_rustls_server_config(
    allower: impl Allower + 'static,
    certificate_server_name: String,
    network_keypair: NetworkKeyPair,
) -> ServerConfig {
    // let allower = AllowedPublicKeys::new(context);
    let self_signed_cert = SelfSignedCertificate::new(
        network_keypair.private_key().into_inner(),
        &certificate_server_name,
    );
    let verifier = ClientCertVerifier::new(allower, certificate_server_name);

    let tls_cert = self_signed_cert.rustls_certificate();
    let tls_private_key = self_signed_cert.rustls_private_key();
    let mut tls_config = verifier
        .rustls_server_config(vec![tls_cert], tls_private_key)
        .unwrap_or_else(|e| panic!("failed to create TLS server config: {:?}", e));
    tls_config.alpn_protocols = vec![b"h2".to_vec()];
    tls_config
}

pub fn create_rustls_client_config(
    target_public_key: Ed25519PublicKey,
    certificate_server_name: String,
    network_keypair: NetworkKeyPair,
) -> ClientConfig {
    // let target_public_key = context
    //     .committee
    //     .authority(target)
    //     .network_key
    //     .clone()
    //     .into_inner();
    let self_signed_cert = SelfSignedCertificate::new(
        network_keypair.private_key().into_inner(),
        &certificate_server_name,
    );
    let tls_cert = self_signed_cert.rustls_certificate();
    let tls_private_key = self_signed_cert.rustls_private_key();
    let mut tls_config = ServerCertVerifier::new(target_public_key, certificate_server_name)
        .rustls_client_config(vec![tls_cert], tls_private_key)
        .unwrap_or_else(|e| panic!("Failed to create TLS client config: {:?}", e));
    tls_config.alpn_protocols = vec![];
    tls_config
}

// tests

/// Extracts the public key from a certificate.
pub fn public_key_from_certificate(
    certificate: &CertificateDer,
) -> Result<Ed25519PublicKey, rustls::Error> {
    use x509_parser::{certificate::X509Certificate, prelude::FromDer};

    let cert = X509Certificate::from_der(certificate.as_ref())
        .map_err(|e| rustls::Error::General(e.to_string()))?;
    let spki = cert.1.public_key();
    let public_key_bytes =
        <ed25519::pkcs8::PublicKeyBytes as ed25519::pkcs8::DecodePublicKey>::from_public_key_der(
            spki.raw,
        )
        .map_err(|e| rustls::Error::General(format!("invalid ed25519 public key: {e}")))?;

    let public_key = Ed25519PublicKey::from_bytes(public_key_bytes.as_ref())
        .map_err(|e| rustls::Error::General(format!("invalid ed25519 public key: {e}")))?;
    Ok(public_key)
}

#[cfg(test)]
mod tests {
    const VALIDATOR_SERVER_NAME: &str = "soma";

    use super::*;
    use acceptor::{TlsAcceptor, TlsConnectionInfo};
    use fastcrypto::ed25519::Ed25519KeyPair;
    use fastcrypto::traits::KeyPair;
    use rustls::client::danger::ServerCertVerifier as _;
    use rustls::pki_types::ServerName;
    use rustls::pki_types::UnixTime;
    use rustls::server::danger::ClientCertVerifier as _;
    use verifier::{AllowAll, HashSetAllow};

    #[test]
    fn verify_allowall() {
        let mut rng = rand::thread_rng();
        let allowed = Ed25519KeyPair::generate(&mut rng);
        let disallowed = Ed25519KeyPair::generate(&mut rng);
        let random_cert_bob = SelfSignedCertificate::new(allowed.private(), VALIDATOR_SERVER_NAME);
        let random_cert_alice =
            SelfSignedCertificate::new(disallowed.private(), VALIDATOR_SERVER_NAME);

        let verifier = ClientCertVerifier::new(AllowAll, VALIDATOR_SERVER_NAME.to_string());

        // The bob passes validation
        verifier
            .verify_client_cert(&random_cert_bob.rustls_certificate(), &[], UnixTime::now())
            .unwrap();

        // The alice passes validation
        verifier
            .verify_client_cert(
                &random_cert_alice.rustls_certificate(),
                &[],
                UnixTime::now(),
            )
            .unwrap();
    }

    #[test]
    fn verify_server_cert() {
        let mut rng = rand::thread_rng();
        let allowed = Ed25519KeyPair::generate(&mut rng);
        let disallowed = Ed25519KeyPair::generate(&mut rng);
        let allowed_public_key = allowed.public().to_owned();
        let random_cert_bob = SelfSignedCertificate::new(allowed.private(), VALIDATOR_SERVER_NAME);
        let random_cert_alice =
            SelfSignedCertificate::new(disallowed.private(), VALIDATOR_SERVER_NAME);

        let verifier =
            ServerCertVerifier::new(allowed_public_key, VALIDATOR_SERVER_NAME.to_string());

        // The bob passes validation
        verifier
            .verify_server_cert(
                &random_cert_bob.rustls_certificate(),
                &[],
                &ServerName::try_from("example.com").unwrap(),
                &[],
                UnixTime::now(),
            )
            .unwrap();

        // The alice does not pass validation
        let err = verifier
            .verify_server_cert(
                &random_cert_alice.rustls_certificate(),
                &[],
                &ServerName::try_from("example.com").unwrap(),
                &[],
                UnixTime::now(),
            )
            .unwrap_err();
        assert!(
            matches!(err, rustls::Error::General(_)),
            "Actual error: {err:?}"
        );
    }

    #[test]
    fn verify_hashset() {
        let mut rng = rand::thread_rng();
        let allowed = Ed25519KeyPair::generate(&mut rng);
        let disallowed = Ed25519KeyPair::generate(&mut rng);

        let allowed_public_key = allowed.public().to_owned();
        let allowed_cert = SelfSignedCertificate::new(allowed.private(), VALIDATOR_SERVER_NAME);

        let disallowed_cert =
            SelfSignedCertificate::new(disallowed.private(), VALIDATOR_SERVER_NAME);

        let mut allowlist = HashSetAllow::new();
        let verifier =
            ClientCertVerifier::new(allowlist.clone(), VALIDATOR_SERVER_NAME.to_string());

        // Add our public key to the allower
        allowlist
            .inner_mut()
            .write()
            .unwrap()
            .insert(allowed_public_key);

        // The allowed cert passes validation
        verifier
            .verify_client_cert(&allowed_cert.rustls_certificate(), &[], UnixTime::now())
            .unwrap();

        // The disallowed cert fails validation
        let err = verifier
            .verify_client_cert(&disallowed_cert.rustls_certificate(), &[], UnixTime::now())
            .unwrap_err();
        assert!(
            matches!(err, rustls::Error::General(_)),
            "Actual error: {err:?}"
        );

        // After removing the allowed public key from the set it now fails validation
        allowlist.inner_mut().write().unwrap().clear();
        let err = verifier
            .verify_client_cert(&allowed_cert.rustls_certificate(), &[], UnixTime::now())
            .unwrap_err();
        assert!(
            matches!(err, rustls::Error::General(_)),
            "Actual error: {err:?}"
        );
    }

    #[test]
    fn invalid_server_name() {
        let mut rng = rand::thread_rng();
        let keypair = Ed25519KeyPair::generate(&mut rng);
        let public_key = keypair.public().to_owned();
        let cert = SelfSignedCertificate::new(keypair.private(), "not-soma");

        let mut allowlist = HashSetAllow::new();
        let client_verifier =
            ClientCertVerifier::new(allowlist.clone(), VALIDATOR_SERVER_NAME.to_string());

        // Add our public key to the allower
        allowlist
            .inner_mut()
            .write()
            .unwrap()
            .insert(public_key.clone());

        // Allowed public key but the server-name in the cert is not the required
        let err = client_verifier
            .verify_client_cert(&cert.rustls_certificate(), &[], UnixTime::now())
            .unwrap_err();
        assert_eq!(
            err,
            rustls::Error::InvalidCertificate(rustls::CertificateError::NotValidForName),
            "Actual error: {err:?}"
        );

        let server_verifier =
            ServerCertVerifier::new(public_key, VALIDATOR_SERVER_NAME.to_string());

        // Allowed public key but the server-name in the cert is not the required
        let err = server_verifier
            .verify_server_cert(
                &cert.rustls_certificate(),
                &[],
                &ServerName::try_from("example.com").unwrap(),
                &[],
                UnixTime::now(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            rustls::Error::InvalidCertificate(rustls::CertificateError::NotValidForName),
            "Actual error: {err:?}"
        );
    }

    #[tokio::test]
    async fn axum_acceptor() {
        use fastcrypto::ed25519::Ed25519KeyPair;
        use fastcrypto::traits::KeyPair;

        let mut rng = rand::thread_rng();
        let client_keypair = Ed25519KeyPair::generate(&mut rng);
        let client_public_key = client_keypair.public().to_owned();
        let client_certificate =
            SelfSignedCertificate::new(client_keypair.private(), VALIDATOR_SERVER_NAME);
        let server_keypair = Ed25519KeyPair::generate(&mut rng);
        let server_certificate = SelfSignedCertificate::new(server_keypair.private(), "localhost");

        let client = reqwest::Client::builder()
            .add_root_certificate(server_certificate.reqwest_certificate())
            .identity(client_certificate.reqwest_identity())
            .https_only(true)
            .build()
            .unwrap();

        let mut allowlist = HashSetAllow::new();
        let tls_config =
            ClientCertVerifier::new(allowlist.clone(), VALIDATOR_SERVER_NAME.to_string())
                .rustls_server_config(
                    vec![server_certificate.rustls_certificate()],
                    server_certificate.rustls_private_key(),
                )
                .unwrap();

        async fn handler(tls_info: axum::Extension<TlsConnectionInfo>) -> String {
            tls_info.public_key().unwrap().to_string()
        }

        let app = axum::Router::new().route("/", axum::routing::get(handler));
        let listener = std::net::TcpListener::bind("localhost:0").unwrap();
        let server_address = listener.local_addr().unwrap();
        let acceptor = TlsAcceptor::new(tls_config);
        let _server = tokio::spawn(async move {
            axum_server::Server::from_tcp(listener)
                .acceptor(acceptor)
                .serve(app.into_make_service())
                .await
                .unwrap()
        });

        let server_url = format!("https://localhost:{}", server_address.port());
        // Client request is rejected because it isn't in the allowlist
        client.get(&server_url).send().await.unwrap_err();

        // Insert the client's public key into the allowlist and verify the request is successful
        allowlist
            .inner_mut()
            .write()
            .unwrap()
            .insert(client_public_key.clone());

        let res = client.get(&server_url).send().await.unwrap();
        let body = res.text().await.unwrap();
        assert_eq!(client_public_key.to_string(), body);
    }
}
