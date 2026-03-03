"""Debug: diagnose safetensors header issues in scoring service.

Downloads a single encrypted model weights file from the chain manifest,
decrypts it locally, and inspects the safetensors header to identify
why the scoring service reports "wrong header length".

Usage:
    uv run soma-example-debug-safetensors
    # or with a specific chain:
    CHAIN=testnet uv run soma-example-debug-safetensors
"""

import asyncio
import os
import struct

import aiohttp

from soma_sdk import SomaClient


async def run():
    chain = os.environ.get("CHAIN", "testnet")
    print(f"Connecting to {chain}...")
    client = await SomaClient(chain=chain)

    # 1. Get an open target and its model manifests
    print("\n=== Step 1: Fetch target + model manifests ===")
    targets = await client.get_targets(status="open")
    target = next(iter(targets))
    print(f"Target: {target.id}")
    print(f"  model_ids: {target.model_ids}")

    manifests = await client.get_model_manifests(target)
    print(f"  Fetched {len(manifests)} manifest(s)")

    manifest = manifests[0]
    print("\n=== Step 2: Manifest details ===")
    print(f"  url:            {manifest.url}")
    print(f"  checksum:       {manifest.checksum}")
    print(f"  size:           {manifest.size}")
    print(f"  decryption_key: {manifest.decryption_key}")

    # 2. Download the raw bytes
    print("\n=== Step 3: Download raw bytes from URL ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(manifest.url) as response:
            if response.status != 200:
                print(f"  ERROR: HTTP {response.status}")
                return
            raw_bytes = await response.read()
    print(f"  Downloaded {len(raw_bytes)} bytes")
    print(f"  Expected size from manifest: {manifest.size}")
    if len(raw_bytes) != manifest.size:
        print(f"  WARNING: Size mismatch! Downloaded {len(raw_bytes)} vs manifest {manifest.size}")

    # 3. Verify checksum
    print("\n=== Step 4: Verify checksum ===")
    computed_checksum = SomaClient.commitment(raw_bytes)
    print(f"  Computed: {computed_checksum}")
    print(f"  Expected: {manifest.checksum}")
    if computed_checksum == manifest.checksum:
        print("  OK: checksums match (raw downloaded bytes = encrypted weights)")
    else:
        print("  MISMATCH: checksums differ!")

    # 4. Inspect raw bytes as safetensors (before decryption)
    print("\n=== Step 5: Inspect raw bytes (encrypted) as safetensors ===")
    inspect_safetensors(raw_bytes, label="encrypted")

    # 5. Decrypt and inspect
    if manifest.decryption_key:
        print("\n=== Step 6: Decrypt and inspect ===")
        decrypted = SomaClient.decrypt_weights(raw_bytes, manifest.decryption_key)
        print(f"  Decrypted {len(decrypted)} bytes")
        inspect_safetensors(decrypted, label="decrypted")

        # 6. Try loading with safetensors library
        print("\n=== Step 7: Try loading with safetensors Python library ===")
        try:
            from safetensors import safe_open
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
                f.write(decrypted)
                tmp_path = f.name

            try:
                with safe_open(tmp_path, framework="numpy") as f:
                    keys = f.keys()
                    print(f"  SUCCESS: Loaded {len(keys)} tensor(s)")
                    for k in sorted(keys)[:10]:
                        t = f.get_tensor(k)
                        print(f"    {k}: shape={t.shape} dtype={t.dtype}")
                    if len(keys) > 10:
                        print(f"    ... and {len(keys) - 10} more")
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"  FAILED to load safetensors: {e}")
        # 7. Simulate double-decryption (what happens on 2nd scoring call)
        print("\n=== Step 8: Simulate double-decryption (2nd scoring call) ===")
        print("  The scoring service caches the decrypted file at the same path.")
        print("  On a 2nd call, download is skipped (cache hit) but decrypt runs again.")
        double_decrypted = SomaClient.decrypt_weights(decrypted, manifest.decryption_key)
        print("  Applying CTR decrypt to already-decrypted bytes...")
        inspect_safetensors(double_decrypted, label="double-decrypted")
    else:
        print("\n=== Step 6: No decryption key available ===")
        print("  Cannot decrypt — the model may not have been revealed yet.")


def inspect_safetensors(data: bytes, label: str = ""):
    """Inspect the safetensors header structure of the given bytes."""
    prefix = f"  [{label}] " if label else "  "

    if len(data) < 8:
        print(f"{prefix}ERROR: Data too short ({len(data)} bytes), need at least 8 for header length")
        return

    # safetensors format: first 8 bytes = u64 LE = JSON header length
    header_len = struct.unpack("<Q", data[:8])[0]
    print(f"{prefix}Header length field (u64 LE): {header_len}")
    print(f"{prefix}  hex of first 16 bytes: {data[:16].hex()}")
    print(f"{prefix}  Total file size: {len(data)}")
    print(f"{prefix}  Remaining after 8-byte prefix: {len(data) - 8}")

    if header_len > len(data) - 8:
        print(f"{prefix}ERROR: Header length ({header_len}) exceeds available data ({len(data) - 8})")
        print(f"{prefix}  This is the 'wrong header length' error the Rust safetensors parser sees!")

        # Check if this looks like the header length is astronomically large
        # (typical sign of encrypted/corrupted data)
        if header_len > 100_000_000:
            print(f"{prefix}  Header length is unreasonably large — data is likely still encrypted or corrupted")
        return

    if header_len > 100_000_000:
        print(f"{prefix}WARNING: Header length is very large ({header_len} bytes), possibly corrupted")
        return

    # Try to parse the JSON header
    header_bytes = data[8:8 + header_len]
    try:
        import json
        header = json.loads(header_bytes)
        tensor_data_start = 8 + header_len
        tensor_data_len = len(data) - tensor_data_start
        print(f"{prefix}JSON header: valid ({header_len} bytes)")
        print(f"{prefix}Tensor data: {tensor_data_len} bytes (starts at offset {tensor_data_start})")

        # Show tensor names from header
        metadata = header.pop("__metadata__", {})
        tensor_names = sorted(header.keys())
        print(f"{prefix}Tensors: {len(tensor_names)}")
        for name in tensor_names[:10]:
            info = header[name]
            print(f"{prefix}  {name}: dtype={info.get('dtype')} shape={info.get('shape')} offsets={info.get('data_offsets')}")
        if len(tensor_names) > 10:
            print(f"{prefix}  ... and {len(tensor_names) - 10} more")
        if metadata:
            print(f"{prefix}Metadata: {metadata}")
    except json.JSONDecodeError as e:
        print(f"{prefix}ERROR: Header is not valid JSON: {e}")
        # Show first 200 bytes of the header for debugging
        preview = header_bytes[:200]
        print(f"{prefix}  Header preview (first 200 bytes): {preview}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
