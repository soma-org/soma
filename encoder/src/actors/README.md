# Actors

The implementation of actors is inspired by Alice Ryhl's (core contributor for Tokio) [blog on actors](https://ryhl.io/blog/actors-with-tokio/).

The actor pattern is a fundamental building block for concurrent systems where each actor is an independent unit that processes messages sequentially and maintains its own private state. Actors communicate strictly through message passing, eliminating the need for explicit locking or shared memory synchronization.

> A common use-case of actors is to assign the actor exclusive ownership of some resource you want to share, and then let other tasks access this resource indirectly by talking to the actor.

In the case of Soma encoders, there are many resources that must be shared effectively. These resource-limited operations are then controlled by a single actor, allowing the actor to maintain appropriate bounds. Take, for instance, limiting the number of concurrent data downloads that are active at any given period of time.

Actors are also used for orchestrating more complex flows. Take, for instance, the process of moving from a shard input to producing the embedding and finally going through the commit/reveal process. The background process of handling an input is controlled by an actor. The actor's use here allows for backpressure for each flow such that if processing shard input data by an encoder model is saturated, processing commits, which does not rely on the encoder model, can continue to operate.