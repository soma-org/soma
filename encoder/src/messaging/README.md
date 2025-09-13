# Encoder Messaging

The messaging module contains all the code related to both client and server messaging between encoders.

Soma uses sharding to generate representations. Sharding is beneficial because it allows the network to horizontally scale, decrease latency and reduce redundant computations since only a subset of encoders are used at any time.

Starting from a transaction submitted to the chain, a full node submits a transaction that specifies the size of dataset, and a digest (hash) of the details of the shard that are kept secret from the rest of the network. The secret section contains the dataset metadata and a nonce. By keeping the dataset digest and RPCs secret, the client is protected against censorship related attacks. At this point, only the full node knows who is in the shard. 

The shard is selected using a combination of the input data metadata, nonce, and block randomness (VDF). The full node then contacts the shard members, proving that the transaction and data are legitimate.

Shard members process the dataset, producing representations. The representations are scored by the individual encoder to create a submission which contains the self-reported score, the metadata about the representations, and additional useful information.

The hash of the submission is committed to other shard members. After collecting a quorum number of commits, the encoder starts a timeout. Once the timeout triggers or all encoders have committed the encoder produces a commit vote.


The commit vote is an explicit acceptance of a specific commit digest. Reject votes are implicit meaning that if there is no accept vote for that encoder, that indicates a reject vote. To finalize an encoder, that encoder must receive a quorum of accept votes for the same commit digest. To reject an encoder they must receive a quorum of reject votes OR be unable to form a quorum of accept votes with the remaining votes they have yet to receive.

Once every encoder in the shard has been finalized, it is safe to reveal. Encoders reveal their original submissions only after they locally deem it safe to do so. Since there is a chance that reveals are received from other encoders prior to finalizing the accepted commits, reveals are stored regardless of commit vote status.

After storing the reveal, the node attempts to look up all the accepted votes and cross compares the reveals with those accepted commits to ensure that they are valid. Once a quorum of valid reveals (matches a member of accepted commits and submission is valid) are received, the node starts a timeout. Once the timeout triggers or all valid reveals have been collected, the node starts evaluation.

Evaluation starts with the highest self reported score and independently verifies the submission score, summary embedding, etc. If everything is within an acceptable tolerance then the encoder generates and signs a report listing that submission as the winner. The report is broadcast to all encoders.

In the case that the encoder fails to verify the submission, the encoder tallies the submitter and moves to the second highest score.

Any encoders that are in the accepted commit, but has not revealed is automatically tallied.

After receiving a quorum number of matching reports, the encoder is able to form an aggregate signature which is used to prove both validity to the full node and the rest of the authorities!

## Flow verification notes

## Input

The input is sent from a full node to the encoders in the shard. The input contains a shard auth token which contains everything an encoder needs to prove validity of the data and corresponding transaction.

If the input passes all verification, the encoder downloads the data, generates representations for the data, and then computes it own score for the data. Finally the encoder sends the digest of their submission as a commit. 

## Commit

The commit messages are sent from encoders in the shard to other encoders in the shard. The verification step for a commit is minimal as long as the encoder is a valid shard member. The commit is saved to produce a commit vote later.

Once a quorum number of commits have been received, that node starts a timeout that waits a bit longer to accept any lagging nodes. After the timeout triggers or all encoders have committed, the encoder produces a commit vote that votes on which commits that encoder has seen.


## Commit Votes
Commit votes are validated to ensure each encoder inside the vote is a shard member and unique. Votes are then aggregated for specific commit digests until a quorum has been reached for each encoder. Accept votes are explicit, reject votes are implicit such that not voting for an encoder signals a commit vote. To finalize a rejection you need either a quorum of reject votes or it must be impossible to form a quorum on a single digest given the remaining votes.

After all encoders have been finalized as either accepted or rejected, it is safe for the encoders to reveal. At this point since full quorum was achieved, every encoder should agree on the accepted commits.

## Reveal

Receiving a reveal is saved with minimal verification besides checking shard inclusion. Note: failing early still stores the reveal since there might be slight message delays or unordered messages that result in a reveal arriving before the node locally has formed their accepted commits.

After adding the reveal, the accepted commit votes are looked up and cross compared counting the number of reveals match the accepted commit digests.


Once a quorum of valid reveals have been collected, the node starts a timeout. After the timeout hits or all valid reveals have been received the node starts evaluation.

## Evaluation (Pipeline)

Any encoders that are in the accepted commits but has not revealed are tallied. It is expected that all accepted commits will reveal. Furthermore as evaluation takes place, any inability to download data, or independently verify scores also results in tallying.

All the valid reveals are sorted by their self reported score. The highest score is processed first, downloading data, and independently verifying the submission. If this succeeds, that top score is selected as the winner and the encoder broadcasts a signature for that top score.

In the case that the top score fails, the encoder tallies the submitter, and moves to the next best score. This proceeds until a submission succeeds or all submissions are exhausted.

Note: the encoder will retry if a download fails or hardware issues occur. Progressing to the next score is a last step if retries fail. 

## Report Votes

Votes for winning submissions are collected and accepted commits. Once a quorum number of votes for the same report are collected, the signatures are aggregated to sign off on the final report. The aggregate signature of the report proves to the authorities and original submitter that the shard came to consensus regarding the winning representations. 


## Clean Up

After an aggregate signature is formed, or a long timeout, the encoder will clean up after the shard performing pruning / cancellation of cancellation tokens.

