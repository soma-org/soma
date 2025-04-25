# Encoder Messaging

The messaging module contains all of the code related to both client and server
messaging between encoders.

## Sharding

### High Level

Soma uses sharding to generate embeddings. Sharding is beneficial because it
allows the network to horizontally scale due to the way shards are selected.
Sharding also helps to decrease latency and redundant computations since only a
subset of encoders are used at any time.

The flow for a shard at a high level is comprised of a few stages:

Starting from a transaction submitted to the chain, a client submits a
transaction that specifies the modality (which set of encoders to use), the size
of dataset, and a digest (hash) of the details of the shard that are kept secret
from the rest of the network. The secret section contains the dataset digest,
and a set of primary/backup RPCs to receive the final embeddings on behalf of
the client. By keeping the dataset digest and RPCs secret, the client is
protected against censorship related attacks. At this point, only the client
knows who the shard members are.

The shard is selected using a combination of dataset digest and the blocks
threshold signature. Soma is unique in that blocks are signed using threshold
BLS signatures. Threshold signatures are an excellent source of randomness that
cannot be biased ahead of time. A client, or client's RPC then contacts the
shard members, proving to the shard members that their transaction of sufficient
funds has been finalized by the network and the threshold signature+dataset
digest result in the correct shard.

Shard members process the dataset, producing embeddings. These embeddings are
then encrypted using an AES key. The encrypted dataset + a download location are
committed to rest of the shard members. In a no failure setting, encoders after
receiving all commits reveal their encryption key. As commits come in, shard
members immediately start to download the encrypted data and corresponding probe
weights.

After a full reveal, shard members sync state and send a signed message
containing the scores for each member in the shard along with the finalized
embeddings. All endorsement scores should be identical due to deterministic
execution with the same state. Endorsements are aggregated together to create a
final signed endorsement that proves that the shard has reached quorum.

Once quorum has been reached, each encoder staggers a countdown to attempt
submitting the final endorsement back on chain and delivering to the specified
RPCs starting with the winner. Since the submission on-chain or delivery is
idempotent, it does not matter if two shard members submit at the same time. The
stagger is just to reduce redundant messaging.

After successful submission / delivery, the shard member notifies the rest of
the shard of its success, nullifying the remaining countdowns to submit and
deliver. At this point the shard has fully completed its lifecycle.

### Failure Cases

The process for resolving a faulty shard member is essentially just collecting a
quorum number of signatures to remove that shard member.

For both commits and reveals, an honest shard member will wait until a quorum
number of commits or reveals have been received and then will trigger a timeout
that is calculated based on how long it took to receive the quorum number of
commits. After the timeout is hit, then the shard member attempts to download
the commits/reveals from peers. If downloading from peers fails, the shard
member then broadcasts a signed message to remove the faulty peer. Shard members
collect the signatures until they can form a certificate with a quorum number of
signatures which results in the honest shard members removing the faulty
members.

The same approach is used for downloading encrypted data and probes, except
instead of using messaging to ask for commits and reveals, the shard member
attempts to directly download the data from a peers blob server (the download
location submitted in the honest shards commit message).

During state sync, shard members broadcast the entire set of removal
certificates. Any removal certificates received that a shard member did not know
about gets applied, and the process repeats until that shard member has received
a quorum number of messages matching their latest set of removal certificates.

While statistically difficult to achieve a dishonest majority, achieving a
dishonest minority is significantly easier. That means that liveliness attacks
are more attainable since all that is needed is to control more than the shard
size minus quorum members. However, to protect against this, each honest node
has a long timeout that if the shard was not able to progress, that node tallies
every shard member that did not commit/reveal/etc. Sustained liveliness attacks
would require substantial stake in the network and would eventually result in
slashing due to sufficient tallies. Specific attacks against an individual are
difficult to recognize at a network level, but in the case of a shard halting,
the buyer of the embeddings are reimbursed at which point they can retry their
dataset using an entirely new shard.
