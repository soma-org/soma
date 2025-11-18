use std::{
    sync::{Arc, Weak},
    time::Duration,
};
use tokio::{
    sync::{mpsc::UnboundedReceiver, oneshot, Semaphore},
    time::sleep,
};
use tracing::{Instrument, error, error_span, info, trace, warn};
use types::execution::ExecutionOutput;

use crate::{authority::AuthorityState, execution_scheduler::PendingCertificate};

// Execution should not encounter permanent failures, so any failure can and needs
// to be retried.
pub const EXECUTION_MAX_ATTEMPTS: u32 = 10;
const EXECUTION_FAILURE_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const QUEUEING_DELAY_SAMPLING_RATIO: f64 = 0.05;
/// When a notification that a new pending transaction is received we activate
/// processing the transaction in a loop.
pub async fn execution_process(
    authority_state: Weak<AuthorityState>,
    mut rx_ready_certificates: UnboundedReceiver<PendingCertificate>,
    mut rx_execution_shutdown: oneshot::Receiver<()>,
) {
    info!("Starting pending certificates execution process.");

    // Rate limit concurrent executions to # of cpus.
    let limit = Arc::new(Semaphore::new(num_cpus::get()));

    // Loop whenever there is a signal that a new transactions is ready to process.
    loop {

        let certificate;
        let execution_env;
        let txn_ready_time;
        let _executing_guard;
        tokio::select! {
            result = rx_ready_certificates.recv() => {
                if let Some(pending_cert) = result {
                    certificate = pending_cert.certificate;
                    execution_env = pending_cert.execution_env;
                    txn_ready_time = pending_cert.stats.ready_time.unwrap();
                    _executing_guard = pending_cert.executing_guard;
                } else {
                    // Should only happen after the AuthorityState has shut down and tx_ready_certificate
                    // has been dropped by ExecutionScheduler.
                    info!("No more certificate will be received. Exiting executor ...");
                    return;
                };
            }
            _ = &mut rx_execution_shutdown => {
                info!("Shutdown signal received. Exiting executor ...");
                return;
            }
        };

        let authority = if let Some(authority) = authority_state.upgrade() {
            authority
        } else {
            // Terminate the execution if authority has already shutdown, even if there can be more
            // items in rx_ready_certificates.
            info!("Authority state has shutdown. Exiting ...");
            return;
        };

        // TODO: Ideally execution_driver should own a copy of epoch store and recreate each epoch.
        let epoch_store = authority.load_epoch_store_one_call_per_task();

        let digest = *certificate.digest();
        trace!(?digest, "Pending certificate execution activated.");

        if epoch_store.epoch() != certificate.epoch() {
            info!(
                ?digest,
                cur_epoch = epoch_store.epoch(),
                cert_epoch = certificate.epoch(),
                "Ignoring certificate from previous epoch."
            );
            continue;
        }

        let limit = limit.clone();
        // hold semaphore permit until task completes. unwrap ok because we never close
        // the semaphore in this context.
        let permit = limit.acquire_owned().await.unwrap();


        // Certificate execution can take significant time, so run it in a separate task.
        let epoch_store_clone = epoch_store.clone();
        tokio::spawn(epoch_store.within_alive_epoch(async move {
            let _guard = permit;
            if authority.is_tx_already_executed(&digest) {
                return;
            }


            match authority.try_execute_immediately(
                &certificate,
                execution_env,
                &epoch_store_clone,
            ).await {
                ExecutionOutput::Success(_) => {
                   
                }
                ExecutionOutput::EpochEnded => {
                    warn!("Could not execute transaction {digest:?} because validator is halted at epoch end. certificate={certificate:?}");
                }
                ExecutionOutput::Fatal(e) => {
                    panic!("Failed to execute certified transaction {digest:?}! error={e} certificate={certificate:?}");
                }
                ExecutionOutput::RetryLater => {
                    // Transaction will be retried later and auto-rescheduled, so we ignore it here
                
                }
            }
        }.instrument(error_span!("execution_driver", tx_digest = ?digest))));
    }
}