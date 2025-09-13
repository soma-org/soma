# Encoder

The encoder crate contains most of the logic around participating in competitions to generate compressed representations.

For each piece of data and corresponding competition, a group of encoders are selected by stake-weighted random sampling. These encoders then progress through stages (commit, commit votes, reveal, report votes) to finally come to consensus on which representations are the "best".

The "best" representations are calculated using a fixed probe architecture that performs reconstruction on the original data in an unsupervised manner. The reconstruction loss is adjusted based on how many bytes are represented by each embedding (compression factor). Intelligence is ultimately how effective a model is at compressing the problem. The "best" representations have low loss and a high degree of compression.

# Important Files

The most important files are inside of the pipelines directory. The pipelines contain the systems core logic for progressing to handling messages and progressing the next stages.

The encoder specific types are useful to understand the contents of the messages, along with message specific verification.

The shard verifier contains code on authorizing work for generating an embedding using a transaction proof along with the random sampling.

Finally the internal/external services along with tonic implementations serve as the entry point for external messages to enter the pipelines after verification.


# Attack Resistance

Great care has been taken to secure the encoder network against attacks.

### Collusion Resistance

Collusion occurs when the probability of detection is low, and the reward is high. To combat collusion economically, the key is to adjust the system until the payoff function until the expected value of colluding costs more than the expected value of acting honestly. 

In the case of Soma, there are two ways to earn tokens: operating an encoder, and submitting data. One case of collusion is to make it profitable for encoders to fake the actual scores for a piece of data such that it is eligible for the data reward. We mitigate this paying out data reward from all the encoders in a shard except for the best encoder. This makes it unprofitable for a majority of the encoders to collude to create a specific data score.

Operating an encoder is not profitable unless the encoders are operating at a high stake efficiency. Win rates are tracked for encoders that participate in shards. 


### DDoS

There are multiple protections against DDoS attacks.

Encoders reject all communication from computers that do not have stake in the
network. In order to register as an accepted networking peer, the computer must
have an established TLS key as well as meet a minimum stake.

If an abusive peer meets the stake requirement, then the
encoder can quickly blacklist that peer locally. The abusive peer would then
need to unstake and restake funds to create a new identity which takes time.
Alternatively, the peer would need to have a lot of staked identities which
becomes expensive especially when the computational cost to blacklist an
identity is so cheap.

Furthermore, encoders can operate behind reverse proxies or cloud infrastructure
specially designed to handle DDoS attacks.

### Censorship

A malicious actor may want to censor a specific parties ability to create
embeddings. This can be achieved by either: operating malicious encoders and
causing an encoder shard to halt (more below on liveliness attacks), or
attempting to take down the encoders responsible for processing the data.

In the case of later, the shard members are hidden from the rest of the network.
The seed for shard selection is a combination of the block specific randomness combined with the raw metadata/nonce for a piece of data.

The metadata/nonce is kept secret and just referenced to using a hash of the
secret data by the transaction.

By combining both to create the seed for a shard, we are able to keep the selected shard a secret from the rest of the network yet still reveal to the shard members that the transaction and shard is valid. 


### Liveliness

Another means of attacking the network or censoring is a liveliness attack.

Shards are designed to make it probabilistically unlikely for a shard
to have a dishonest majority. However, it is possible to have a
dishonest MINORITY. If shard size minus the number of nodes needed to meet quorum are dishonest, it is possible to halt execution of a shard. 

To combat this, every honest encoder starts a timeout that is sufficiently long
e.g. waiting a full epoch. At this point, any encoders that have not submitted
data to that encoder are tallied.

Additionally, if a shard does not complete, the buyer is refunded allowing for a
view change with an entirely different shard.

Ultimately, the best way to protect against a 20% malicious stake is to
distribute tokens in such a way that maximizes decentralization and meritocracy.

### Malicious Buyers

To prove to an encoder that they should work on a piece of data, the full node must provide a cryptographic proof of transaction finality.

As long as the encoder knows the authorities for the epoch, the encoder can strongly verify that a transaction is valid, has funds, and the dataset metadata matches the transaction.

In the case of a dataset that is larger than expected, ill formatted, or does not match checksums the encoder stops work immediately. While the buyer will be refunded since the shard could not complete, they do not receive the full amount but rather lose some of the value due to network fees. 

Full nodes are expected to verify the validity of input data prior to submission to the encoders so any failure regarding the integrity of the data results in the full node being tallied and at worst blacklisted.

### Malicious Encoders

As long as the malicious encoders in a shard do not control 20% of the nodes,
then a shard can effectively remove the encoders from the shard that are faulty.

Another attack that encoders can use is to steal the embeddings generated by a
peer. To protect against copying embeddings, or using a peers embeddings to fine
tune output, the shard goes through a commit reveal scheme. Before revealing
their embeddings, an encoder verifies that their peers have also committed or
they have been removed from the shard. This stops unwanted copying.

