#!/usr/bin/env python3
"""Verify the built .duckdb_extension is valid."""

import struct, sys, os

def main():
    ext_path = os.environ.get("EXT_PATH", "target/duckdb_ml.duckdb_extension")
    if not os.path.exists(ext_path):
        print(f"ERROR: {ext_path} not found. Run `make release` first.", file=sys.stderr)
        sys.exit(1)

    with open(ext_path, "rb") as f:
        data = f.read()

    if len(data) < 352:
        print(f"ERROR: file too small ({len(data)} bytes)", file=sys.stderr)
        sys.exit(1)

    # Read footer (last 352 bytes)
    footer = data[-352:]

    # Parse version (bytes 0-31, null-terminated)
    version = footer[:32].split(b'\x00')[0].decode()
    # Parse platform (bytes 32-63, null-terminated)
    platform = footer[32:64].split(b'\x00')[0].decode()
    # Parse ABI version (bytes 64-71, u64 LE)
    abi = struct.unpack('<Q', footer[64:72])[0]

    # Check SO has valid ELF header
    so_data = data[:-352]
    if so_data[:4] != b'\x7fELF':
        print("ERROR: .so part is not ELF", file=sys.stderr)
        sys.exit(1)

    # Check for expected symbols
    symbols_to_check = [
        b'ml_init',
        b'ml_predict',
        b'ml_train',
        b'ml_deploy',
        b'ml_compare',
        b'ml_snapshot',
        b'ml_predict_batch',
        b'duckdb_ml',
    ]
    for sym in symbols_to_check:
        if sym not in so_data:
            print(f"WARNING: symbol '{sym.decode()}' not found in binary", file=sys.stderr)

    print(f"✅ {ext_path} valid")
    print(f"   version:  {version}")
    print(f"   platform: {platform}")
    print(f"   abi:      {abi}")
    print(f"   so size:  {len(so_data):,} bytes")
    print(f"   total:    {len(data):,} bytes")


if __name__ == "__main__":
    main()
