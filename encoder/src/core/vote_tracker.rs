pub(crate) struct VoteTracker {
    // collect votes for unique slots in shard members
    // once you have collected (eval set - quorum) + 1 rejection votes the slot is finalized as rejected
    // once you have collected quorum implicit votes the slot is finalized as accepted
    //
    //
    // if you are a member of inference set, reveal once all slots are finalized
    // both should automatically add adjust the slot tracker so that producing a vote does not wait on ga
    // same with the vote tracker, a removed slot should be finalized as rejected in the reveal round

    // epoch, shard, round, [u16; size of shard]
    // epoch, shard, round, [u16; size of shard]
}
