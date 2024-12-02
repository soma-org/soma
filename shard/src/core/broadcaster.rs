use std::sync::Arc;

use crate::{networking::messaging::EncoderNetworkClient, types::context::EncoderContext};

pub (crate) struct Broadcaster<C: EncoderNetworkClient> {
    context: Arc<EncoderContext>,
    network_client: Arc<C>

}


impl<C: EncoderNetworkClient> Broadcaster<C> {

    pub(crate) fn new(
        context: Arc<EncoderContext>,
        network_client: Arc<C>,
    ) -> Self {
        Self {  
            context,
            network_client
        }
    }


    pub(crate) fn push_commits() {
        // ping everyone in the shard to get a signature
        // fabricate the certificate
        // ping everyone in the shard to notify them of the commit certificate
    }
}


// given some shard and a message, broadcast the message to the entire shard. Go ahead and 