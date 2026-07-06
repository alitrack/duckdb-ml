#!/usr/bin/env python3
"""Post-build: package the .so into a DuckDB loadable .duckdb_extension."""

import struct, sys, os, subprocess
from pathlib import Path

def detect_duckdb_version():
    """Get DuckDB version from system binary."""
    try:
        out = subprocess.check_output(["duckdb", "--version"], text=True)
        return out.strip().split()[0]  # "v1.5.2"
    except Exception:
        # Fallback: try to detect from libduckdb
        return "v1.5.2"


def build_footer(version, platform):
    """Build the 352-byte DuckDB extension metadata footer."""
    footer = bytearray()

    # Version string (32 bytes, null-padded)
    v = version.encode()[:31]
    footer.extend(v + b'\x00' * (32 - len(v)))

    # Platform string (32 bytes, null-padded)
    p = platform.encode()[:31]
    footer.extend(p + b'\x00' * (32 - len(p)))

    # Extension ABI version (8 bytes, u64 LE)
    footer.extend(struct.pack('<Q', 0))

    # Reserved (24 bytes)
    footer.extend(b'\x00' * 24)

    # Signature placeholder (256 bytes)
    footer.extend(b'\x00' * 256)

    assert len(footer) == 352
    return footer


def main():
    # Find the .so
    so_path = os.environ.get("SO_PATH")
    if not so_path:
        # Try common locations
        candidates = [
            "target/release/libduckdb_ml.so",
            "target/debug/libduckdb_ml.so",
        ]
        for c in candidates:
            if os.path.exists(c):
                so_path = c
                break
        if not so_path:
            # Search in deps
            for root, dirs, files in os.walk("target"):
                for f in files:
                    if f.startswith("libduckdb_ml") and f.endswith(".so"):
                        so_path = os.path.join(root, f)
                        break
                if so_path:
                    break

    if not so_path or not os.path.exists(so_path):
        print(f"Error: .so not found. Set SO_PATH.", file=sys.stderr)
        sys.exit(1)

    version = detect_duckdb_version()
    platform = "linux_amd64"

    out_path = os.environ.get("OUT_PATH", "target/duckdb_ml.duckdb_extension")

    with open(so_path, "rb") as f:
        data = f.read()

    footer = build_footer(version, platform)
    data += bytes(footer)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

    print(f"✅ {out_path}")
    print(f"   version:  {version}")
    print(f"   platform: {platform}")
    print(f"   size:     {len(data):,} bytes")


if __name__ == "__main__":
    main()
