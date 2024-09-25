# BIG TODOs:

```
1. Create macros to reduce a lot of the redundant code pertaining to digests, verified versions of structs,
and add the ability to add two functions one that takes a closure that verifies a block and returns an enum, and another that takes the verify enum, runs it, and then returns a verified struct.

2. Add TLS to the blob client to further protect, also add shard filtering, and a default keying in the url

3. Add rust dynamic library support to enable ML in rust as well as python.
```

# Sep 19
- fix modality index and modality committee to be more ergonomic
- split context
- split core / core threads
