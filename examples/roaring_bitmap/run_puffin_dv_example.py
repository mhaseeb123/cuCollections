#!/usr/bin/env python3
"""
Parse Puffin files to extract deletion-vector-v1 blob metadata, then invoke the
CUDA puffin_dv_example binary with the correct parameters.

Usage:
  python run_puffin_dv_example.py [--exe <path>] [--data-dir <dir>]

Defaults:
  --exe       ../../../build/examples/ROARING_BITMAP_PUFFIN_DV_EXAMPLE
  --data-dir  .  (directory containing .puffin files)
"""

import argparse
import json
import struct
import subprocess
import sys
import os

PUFFIN_MAGIC = b"\x50\x46\x41\x31"
DV_MAGIC = bytes([0xD1, 0xD3, 0x39, 0x64])


def parse_puffin_footer(filepath):
    """Read a Puffin file and return the list of BlobMetadata dicts."""
    with open(filepath, "rb") as f:
        data = f.read()

    if data[:4] != PUFFIN_MAGIC:
        raise ValueError(f"{filepath}: missing Puffin magic at start")
    if data[-4:] != PUFFIN_MAGIC:
        raise ValueError(f"{filepath}: missing Puffin magic at end")

    flags = struct.unpack_from("<i", data, len(data) - 8)[0]
    footer_payload_size = struct.unpack_from("<i", data, len(data) - 12)[0]

    footer_start = len(data) - 12 - footer_payload_size
    footer_magic = data[footer_start - 4 : footer_start]
    if footer_magic != PUFFIN_MAGIC:
        raise ValueError(f"{filepath}: missing footer magic")

    compressed = flags & 1
    if compressed:
        raise NotImplementedError("Compressed footer payload not supported")

    payload_bytes = data[footer_start : footer_start + footer_payload_size]
    metadata = json.loads(payload_bytes.decode("utf-8"))
    return metadata["blobs"]


def parse_dv_blob_keys(filepath, offset, length):
    """
    Read the deletion-vector-v1 blob from the file and return the list of
    (key, num_deleted_positions) tuples by parsing the 64-bit Roaring portable header.
    """
    with open(filepath, "rb") as f:
        f.seek(offset)
        blob = f.read(length)

    combined_length = struct.unpack(">I", blob[:4])[0]
    magic = blob[4:8]
    if magic != DV_MAGIC:
        raise ValueError("Invalid DV magic")

    vector = blob[8 : 4 + combined_length]

    num_buckets = struct.unpack_from("<Q", vector, 0)[0]
    pos = 8
    keys = []
    for _ in range(num_buckets):
        key = struct.unpack_from("<I", vector, pos)[0]
        pos += 4

        roaring_start = pos
        cookie_raw = struct.unpack_from("<I", vector, pos)[0]
        SERIAL_COOKIE = 12347
        SERIAL_COOKIE_NO_RUN = 12346
        NO_OFFSET_THRESHOLD = 4

        if (cookie_raw & 0xFFFF) == SERIAL_COOKIE:
            num_containers = ((cookie_raw >> 16) & 0xFFFF) + 1
            pos += 4
            run_bitmap_bytes = (num_containers + 7) // 8
            run_bitmap = vector[pos : pos + run_bitmap_bytes]
            pos += run_bitmap_bytes
            has_run = True
        elif cookie_raw == SERIAL_COOKIE_NO_RUN:
            pos += 4
            num_containers = struct.unpack_from("<I", vector, pos)[0]
            pos += 4
            run_bitmap = b"\x00" * ((num_containers + 7) // 8)
            has_run = False
        else:
            raise ValueError(f"Unknown roaring cookie: {cookie_raw:#x}")

        key_cards_start = pos
        cards = []
        for i in range(num_containers):
            _k = struct.unpack_from("<H", vector, pos)[0]
            pos += 2
            card_minus_1 = struct.unpack_from("<H", vector, pos)[0]
            pos += 2
            cards.append(card_minus_1 + 1)

        if (not has_run) or (num_containers >= NO_OFFSET_THRESHOLD):
            pos += num_containers * 4  # skip offset header

        total_keys = 0
        for i in range(num_containers):
            is_run = (run_bitmap[i // 8] >> (i % 8)) & 1
            card = cards[i]
            if is_run:
                num_runs = struct.unpack_from("<H", vector, pos)[0]
                pos += 2 + num_runs * 4
            elif card <= 4096:
                pos += card * 2
            else:
                pos += 8192
            total_keys += card

        keys.append((key, total_keys))

    return keys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_exe = os.path.join(script_dir, "..", "..", "build", "examples", "ROARING_BITMAP_PUFFIN_DV_EXAMPLE")

    parser = argparse.ArgumentParser(description="Run puffin deletion vector example")
    parser.add_argument("--exe", default=default_exe, help="Path to the CUDA example binary")
    parser.add_argument("--data-dir", default=script_dir, help="Directory containing .puffin files")
    args = parser.parse_args()

    if not os.path.isfile(args.exe):
        print(f"ERROR: Example binary not found at {args.exe}", file=sys.stderr)
        print("Build it first: cd ~/cucollections/build && cmake .. && make -j", file=sys.stderr)
        sys.exit(1)

    test_configs = {
        "dv_32bit.puffin": [
            # key=0, range [0, 1000)
            {"key": 0, "range_min": 0, "range_max": 1000},
        ],
        "dv_64bit.puffin": [
            # key=0, range [0, 1000)
            {"key": 0, "range_min": 0, "range_max": 1000},
            # key=1, range [0, 1000)
            {"key": 1, "range_min": 0, "range_max": 1000},
        ],
    }

    all_passed = True
    for filename, key_ranges in test_configs.items():
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"SKIP: {filepath} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print(f"{'='*60}")

        blobs = parse_puffin_footer(filepath)
        dv_blobs = [b for b in blobs if b["type"] == "deletion-vector-v1"]
        if not dv_blobs:
            print(f"  No deletion-vector-v1 blobs found in {filepath}")
            continue

        blob = dv_blobs[0]
        offset = blob["offset"]
        length = blob["length"]
        print(f"  Blob offset={offset}, length={length}")
        print(f"  Cardinality={blob['properties'].get('cardinality', '?')}")

        bucket_keys = parse_dv_blob_keys(filepath, offset, length)
        print(f"  Buckets: {bucket_keys}")

        cmd = [
            args.exe,
            filepath,
            str(offset),
            str(length),
            str(len(key_ranges)),
        ]
        for kr in key_ranges:
            cmd.extend([str(kr["key"]), str(kr["range_min"]), str(kr["range_max"])])

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            all_passed = False
        else:
            print(f"  OK")

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL PUFFIN DV TESTS PASSED")
    else:
        print("SOME PUFFIN DV TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
