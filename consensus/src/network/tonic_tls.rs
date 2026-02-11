use types::consensus::context::Context;

pub(crate) fn certificate_server_name(context: &Context) -> String {
    "consensus_epoch".to_string()
    // TODO: figure out why this isn't working format!("consensus_epoch_{}", context.committee.epoch())
}
