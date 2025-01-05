# Encoder

The encoder module houses all the code pertaining to communicating, downloading
data, and processing embeddings.

# Attack Resistance

Great care has been taken to secure the encoder network against attacks.

### DDoS

There are multiple protections against DDoS attacks.

Encoders reject all communication from computers that do not have stake in the
network. In order to register as an accepted networking peer, the computer must
have an established TLS key as well as meet a minimum stake.

In the case of an abusive peer that meets the stake requirement, then the
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
The seed for shard selection is a combination of a threshold signature for the
block containing the transaction and a dataset digest (hash). Threshold BLS
signatures offer extremely good randomness that is impossible to bias, but this
signature is public to the network.

The dataset hash is kept secret and just referenced to using a hash of the
secret data by the transaction.

By combining both the threshold signature and dataset hash we get an excellent
source of randomness, that selects a different shard for each dataset submitted,
but is kept secret from the rest of the network.

The client then shares the dataset / transaction proof with encoders which can
verify validity using the one way nature of hashing algorithms.

### Liveliness

Another means of attacking the network or censoring is a liveliness attack.
Shards are designed to make it probabilistically extremely unlikely for a shard
to have a dishonest majority. However, while unlikely, it is possible to have a
dishonest MINORITY. In the case of a dishonest minority that is GREATER than the
shard size minus the number of nodes needed to meet quorum, it is possible to
halt execution of a shard. The reason being that a quorum is required to remove
faulty shard members.

To combat this, every honest encoder starts a timeout that is sufficiently long
e.g. waiting a full epoch. At this point, any encoders that have not submitted
data to that encoder are tallied. Once 2f+1 stake has tallied that encoder, they
are slashed.

Additionally, if a shard does not complete, the buyer is refunded allowing for a
view change with an entirely different shard.

In order to probabilistically have control over a dishonest majority would
require controlling approximately 20% stake for a given modality. If the
dishonest stake acted maliciously frequently, they would eventually be slashed
due to tallies. If the dishonest stake was waiting to censor specific users,
they would have to be competitive at encoding or risk being slashed due to
performance. If the dishonest stake rotated identities, additional steps would
need to be taken to track where the malicious money was moved but due to the
transparency of the monetary system it would be possible to engineer a solution.

Ultimately, the best way to protect against a 20% malicious stake is to
distribute tokens in such a way that maximizes decentralization and meritocracy.

### Malicious Buyers

To prove to an encoder that they are a member of a shard, the client or client's
RPC must provide a cryptographic proof of transaction finality. As long as the
encoder has a valid view of the authorities for a given epoch, the encoder can
strongly verify that a transaction is valid, contains the funds, and that the
dataset digest matches the reference digest in the transaction. In the case of a
dataset that is larger than expected, ill formatted, or does not match checksums
the encoder stops work immediately. While the buyer will be refunded since the
shard could not complete, they do not receive the full amount but rather lose
some of the value due to network fees. The network fees can be tuned to make
this attack expensive.

TODO: Shard members need a way of communicating with the rest of their shard
that the input is bad and that is the reason why they are not submitting a
commit/reveal/etc such that they do not receive a tally.

Buyers may also selectively send inputs to certain encoders and not others. As
long as those encoders are honest, then when they commit, they will notify the
other shard members of the shard's existence and provide the oblivious member
with the input.

TODO: reconsider using a certificate to prove delivery to the shard members

### Malicious Encoders

As long as the malicious encoders in a shard do not control 20% of the nodes,
then a shard can effectively remove the encoders from the shard that are faulty.
To remove an encoder from a shard, the honest shard members must collect a
quorum number of removal signatures to certify the removal.

Another attack that encoders can use is to steal the embeddings generated by a
peer. To protect against copying embeddings, or using a peers embeddings to fine
tune output, the shard goes through a commit reveal scheme. Before revealing
their embeddings, an encoder verifies that their peers have also committed or
they have been removed from the shard. This stops unwanted copying.

### Encoder Collusion

While unwanted copying is impossible using the commit reveal scheme, collusion
is still possible if the colluding encoders opt-in. It is important to note that
collusion and collaboration are different. In the case of a shard, we want to
minimize encoders colluding to select a winner since that might result in
sub-optimal embeddings. However, using other encoders that are not shard members
to generate embeddings is encourage because it leads to better embeddings.

The first defense is that probe weights are locked in at every epoch. The probes
are what is used to evaluate the quality of embeddings, and there is no way to
predict what the shard will be beforehand. This makes the evaluation of
embeddings deterministic whether shard members collude or not.

TODO: consider the economics of incentives to improve performance or hinder
collusion
