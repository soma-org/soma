# Compression

Compression is used to reduce the size of data before being sent over the network. Since the size of the data has a tremendous impact on the time it takes to download the data over the network, often the time it takes to compress data is worth the tradeoff of improving the speed at which the data can be transmitted. 

The compressor trait creates a standard interface that simplifies switching compression algorithms or mocking tests. The default compression algorithm used is [ZStandard](https://facebook.github.io/zstd/).