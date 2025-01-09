# Encryption

Encryption is primarily for commit/reveal. Instead of just committing a hash of the data, the commit contains encrypted data. To reveal, the encryption key is released which allows for reconstructing the original contents. This allows for frontloading the download of large datasets over the network while still waiting for the reveal. 

The default encryption algorithm is Aes256Ctr64LE, or AES256, CTR mode with a 64-bit little endian counter.