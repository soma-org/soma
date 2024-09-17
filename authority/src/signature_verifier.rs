use std::{sync::Arc, time::Duration};

use futures::{future::Either, pin_mut};
use itertools::izip;
use nonempty::NonEmpty;
use parking_lot::{Mutex, MutexGuard};
use tokio::{runtime::Handle, sync::oneshot, time::timeout};
use tracing::info;
use types::crypto::{AuthoritySignInfoTrait, VerificationObligation};
use types::envelope::Message;
use types::transaction::verify_sender_signed_data_message_signatures;
use types::{
    committee::{Committee, EpochId},
    error::{SomaError, SomaResult},
    intent::Intent,
    transaction::{CertifiedTransaction, SenderSignedData, VerifiedCertificate},
};
// Maximum amount of time we wait for a batch to fill up before verifying a partial batch.
const BATCH_TIMEOUT_MS: Duration = Duration::from_millis(10);

// Maximum size of batch to verify. Increasing this value will slightly improve CPU utilization
// (batching starts to hit steeply diminishing marginal returns around batch sizes of 16), at the
// cost of slightly increasing latency (BATCH_TIMEOUT_MS will be hit more frequently if system is
// not heavily loaded).
const MAX_BATCH_SIZE: usize = 8;

type Sender = oneshot::Sender<SomaResult<VerifiedCertificate>>;

struct CertBuffer {
    certs: Vec<CertifiedTransaction>,
    senders: Vec<Sender>,
    id: u64,
}

impl CertBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            certs: Vec::with_capacity(capacity),
            senders: Vec::with_capacity(capacity),
            id: 0,
        }
    }

    // Function consumes MutexGuard, therefore releasing the lock after mem swap is done
    fn take_and_replace(mut guard: MutexGuard<'_, Self>) -> Self {
        let this = &mut *guard;
        let mut new = CertBuffer::new(this.capacity());
        new.id = this.id + 1;
        std::mem::swap(&mut new, this);
        new
    }

    fn capacity(&self) -> usize {
        debug_assert_eq!(self.certs.capacity(), self.senders.capacity());
        self.certs.capacity()
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.certs.len(), self.senders.len());
        self.certs.len()
    }

    fn push(&mut self, tx: Sender, cert: CertifiedTransaction) {
        self.senders.push(tx);
        self.certs.push(cert);
    }
}

/// Verifies signatures in ways that faster than verifying each signature individually.
/// - BLS signatures - caching and batch verification.
/// - User signed data - caching.
pub struct SignatureVerifier {
    committee: Arc<Committee>,
    // certificate_cache: VerifiedDigestCache<CertificateDigest>,
    // signed_data_cache: VerifiedDigestCache<SenderSignedDataDigest>,
    queue: Mutex<CertBuffer>,
}

impl SignatureVerifier {
    pub fn new_with_batch_size(committee: Arc<Committee>, batch_size: usize) -> Self {
        Self {
            committee,
            queue: Mutex::new(CertBuffer::new(batch_size)),
        }
    }

    pub fn new(committee: Arc<Committee>) -> Self {
        Self::new_with_batch_size(committee, MAX_BATCH_SIZE)
    }

    /// Verifies all certs, returns Ok only if all are valid.
    pub fn verify_certs(&self, certs: Vec<CertifiedTransaction>) -> SomaResult {
        let certs: Vec<_> = certs.into_iter().collect();

        // Verify only the user sigs of certificates that were not cached already, since whenever we
        // insert a certificate into the cache, it is already verified.
        for cert in &certs {
            self.verify_tx(cert.data())?;
        }
        batch_verify(&self.committee, &certs)?;
        Ok(())
    }

    /// Verifies one cert asynchronously, in a batch.
    pub async fn verify_cert(&self, cert: CertifiedTransaction) -> SomaResult<VerifiedCertificate> {
        let cert_digest = cert.certificate_digest();
        self.verify_tx(cert.data())?;
        self.verify_cert_skip_cache(cert).await
    }

    pub async fn multi_verify_certs(
        &self,
        certs: Vec<CertifiedTransaction>,
    ) -> Vec<SomaResult<VerifiedCertificate>> {
        // TODO: We could do better by pushing the all of `certs` into the verification queue at once,
        // but that's significantly more complex.
        let mut futures = Vec::with_capacity(certs.len());
        for cert in certs {
            futures.push(self.verify_cert(cert));
        }
        futures::future::join_all(futures).await
    }

    /// exposed as a public method for the benchmarks
    pub async fn verify_cert_skip_cache(
        &self,
        cert: CertifiedTransaction,
    ) -> SomaResult<VerifiedCertificate> {
        // this is the only innocent error we are likely to encounter - filter it before we poison
        // a whole batch.
        // TODO: Verify Epoch
        // if cert.auth_sig().epoch != self.committee.epoch() {
        //     return Err(SomaError::WrongEpoch {
        //         expected_epoch: self.committee.epoch(),
        //         actual_epoch: cert.auth_sig().epoch,
        //     });
        // }

        self.verify_cert_inner(cert).await
    }

    async fn verify_cert_inner(
        &self,
        cert: CertifiedTransaction,
    ) -> SomaResult<VerifiedCertificate> {
        // Cancellation safety: we use parking_lot locks, which cannot be held across awaits.
        // Therefore once the queue has been taken by a thread, it is guaranteed to process the
        // queue and send all results before the future can be cancelled by the caller.
        let (tx, rx) = oneshot::channel();
        pin_mut!(rx);

        let prev_id_or_buffer = {
            let mut queue = self.queue.lock();
            queue.push(tx, cert);
            if queue.len() == queue.capacity() {
                Either::Right(CertBuffer::take_and_replace(queue))
            } else {
                Either::Left(queue.id)
            }
        };
        let prev_id = match prev_id_or_buffer {
            Either::Left(prev_id) => prev_id,
            Either::Right(buffer) => {
                self.process_queue(buffer).await;
                // unwrap ok - process_queue will have sent the result already
                return rx.try_recv().unwrap();
            }
        };

        if let Ok(res) = timeout(BATCH_TIMEOUT_MS, &mut rx).await {
            // unwrap ok - tx cannot have been dropped without sending a result.
            return res.unwrap();
        }

        let buffer = {
            let queue = self.queue.lock();
            // check if another thread took the queue while we were re-acquiring lock.
            if prev_id == queue.id {
                debug_assert_ne!(queue.len(), queue.capacity());
                Some(CertBuffer::take_and_replace(queue))
            } else {
                None
            }
        };

        if let Some(buffer) = buffer {
            self.process_queue(buffer).await;
            // unwrap ok - process_queue will have sent the result already
            return rx.try_recv().unwrap();
        }

        // unwrap ok - another thread took the queue while we were re-acquiring the lock and is
        // guaranteed to process the queue immediately.
        rx.await.unwrap()
    }

    async fn process_queue(&self, buffer: CertBuffer) {
        let committee = self.committee.clone();
        Handle::current()
            .spawn_blocking(move || Self::process_queue_sync(committee, buffer))
            .await
            .expect("Spawn blocking should not fail");
    }

    fn process_queue_sync(committee: Arc<Committee>, buffer: CertBuffer) {
        let results = batch_verify_certificates(&committee, &buffer.certs);
        izip!(
            results.into_iter(),
            buffer.certs.into_iter(),
            buffer.senders.into_iter(),
        )
        .for_each(|(result, cert, tx)| {
            tx.send(match result {
                Ok(()) => Ok(VerifiedCertificate::new_unchecked(cert)),
                Err(e) => Err(e),
            })
            .ok();
        });
    }

    pub fn verify_tx(&self, signed_tx: &SenderSignedData) -> SomaResult {
        verify_sender_signed_data_message_signatures(signed_tx, self.committee.epoch())
    }
}

/// Verifies certificates in batch mode, but returns a separate result for each cert.
pub fn batch_verify_certificates(
    committee: &Committee,
    certs: &[CertifiedTransaction],
) -> Vec<SomaResult> {
    // certs.data() is assumed to be verified already by the caller.
    match batch_verify(committee, certs) {
        Ok(_) => vec![Ok(()); certs.len()],

        // Verify one by one to find which certs were invalid.
        Err(_) if certs.len() > 1 => certs
            .iter()
            // TODO: verify_signature currently checks the tx sig as well, which might be cached
            // already.
            .map(|c| c.verify_signatures_authenticated(committee))
            .collect(),

        Err(e) => vec![Err(e)],
    }
}

fn batch_verify(committee: &Committee, certs: &[CertifiedTransaction]) -> SomaResult {
    let mut obligation = VerificationObligation::default();

    for cert in certs {
        let idx = obligation.add_message(cert.data(), cert.epoch(), Intent::soma_transaction());
        cert.auth_sig()
            .add_to_verification_obligation(committee, &mut obligation, idx)?;
    }

    obligation.verify_all()
}
