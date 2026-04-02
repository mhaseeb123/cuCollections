#!/usr/bin/env python3
"""
Create two Puffin files containing deletion-vector-v1 blobs.

File 1 (32-bit only): deletion positions are all even indices in [0, 1000) — key == 0 only.
File 2 (64-bit):       deletion positions are all even indices in [0, 1000) (key == 0)
                        AND all even indices in [2^32, 2^32 + 1000) (key == 1).

The deletion-vector-v1 blob format (Iceberg spec):
  - 4 bytes big-endian: combined length of (magic + vector)
  - 4 bytes magic: 0xD1 0xD3 0x39 0x64
  - The vector: 64-bit Roaring portable format
      - uint64 LE: number of buckets (keys)
      - For each bucket:
          - uint32 LE: key (upper 32 bits)
          - 32-bit Roaring bitmap in portable format
  - 4 bytes big-endian: CRC-32 of (magic + vector)

Puffin file format:
  Magic("PFA1") | Blob_1 ... Blob_n | Footer
  Footer = Magic | FooterPayload | FooterPayloadSize(4B LE) | Flags(4B LE) | Magic
"""

import struct
import json
import zlib
import sys
import os


PUFFIN_MAGIC = b"\x50\x46\x41\x31"  # "PFA1"
DV_MAGIC = bytes([0xD1, 0xD3, 0x39, 0x64])

SERIAL_COOKIE_NO_RUNCONTAINER = 12346
SERIAL_COOKIE = 12347
NO_OFFSET_THRESHOLD = 4


def serialize_roaring32_array_container(values):
    """Serialize a sorted list of uint16 values as a Roaring array container."""
    buf = b""
    for v in sorted(values):
        buf += struct.pack("<H", v)
    return buf


def serialize_roaring32(values_u32):
    """
    Serialize a set of uint32 values into a 32-bit Roaring bitmap (portable format).
    Groups values by upper 16 bits (container key), uses array containers only.
    """
    containers = {}
    for v in values_u32:
        upper = (v >> 16) & 0xFFFF
        lower = v & 0xFFFF
        containers.setdefault(upper, set()).add(lower)

    keys_sorted = sorted(containers.keys())
    num_containers = len(keys_sorted)

    has_run = False
    use_serial_cookie = has_run  # no run containers

    buf = b""
    if use_serial_cookie:
        cookie = SERIAL_COOKIE | ((num_containers - 1) << 16)
        buf += struct.pack("<I", cookie)
        run_bitmap_bytes = (num_containers + 7) // 8
        buf += b"\x00" * run_bitmap_bytes
    else:
        buf += struct.pack("<I", SERIAL_COOKIE_NO_RUNCONTAINER)
        buf += struct.pack("<I", num_containers)

    for key in keys_sorted:
        card = len(containers[key])
        buf += struct.pack("<H", key)
        buf += struct.pack("<H", card - 1)

    if use_serial_cookie:
        run_bitmap_bytes = (num_containers + 7) // 8
        if num_containers >= NO_OFFSET_THRESHOLD:
            header_size = 4 + run_bitmap_bytes + 4 * num_containers + 4 * num_containers
        else:
            header_size = 4 + run_bitmap_bytes + 4 * num_containers
    else:
        header_size = 4 + 4 + 4 * num_containers + 4 * num_containers

    container_data_list = []
    for key in keys_sorted:
        vals = sorted(containers[key])
        container_data_list.append(serialize_roaring32_array_container(vals))

    if (not use_serial_cookie) or (num_containers >= NO_OFFSET_THRESHOLD):
        offset = header_size
        for cdata in container_data_list:
            buf += struct.pack("<I", offset)
            offset += len(cdata)

    for cdata in container_data_list:
        buf += cdata

    return buf


def serialize_deletion_vector_v1(positions_64bit):
    """
    Serialize a set of 64-bit positions into a deletion-vector-v1 blob.
    """
    buckets = {}
    for pos in positions_64bit:
        key = (pos >> 32) & 0xFFFFFFFF
        sub = pos & 0xFFFFFFFF
        buckets.setdefault(key, set()).add(sub)

    keys_sorted = sorted(buckets.keys())
    num_buckets = len(keys_sorted)

    vector_buf = b""
    vector_buf += struct.pack("<Q", num_buckets)
    for key in keys_sorted:
        vector_buf += struct.pack("<I", key)
        roaring_data = serialize_roaring32(buckets[key])
        vector_buf += roaring_data

    magic_and_vector = DV_MAGIC + vector_buf
    combined_length = len(magic_and_vector)
    crc = zlib.crc32(magic_and_vector) & 0xFFFFFFFF

    blob = b""
    blob += struct.pack(">I", combined_length)
    blob += magic_and_vector
    blob += struct.pack(">I", crc)
    return blob


def make_puffin_file(blobs_data, blob_metadatas):
    """
    Construct a Puffin file from blob data and metadata.
    """
    buf = PUFFIN_MAGIC

    offsets = []
    for bdata in blobs_data:
        offsets.append(len(buf))
        buf += bdata

    for i, meta in enumerate(blob_metadatas):
        meta["offset"] = offsets[i]
        meta["length"] = len(blobs_data[i])

    footer_payload_obj = {"blobs": blob_metadatas, "properties": {"created-by": "cuco-puffin-test"}}
    footer_payload = json.dumps(footer_payload_obj).encode("utf-8")
    footer_payload_size = len(footer_payload)

    buf += PUFFIN_MAGIC
    buf += footer_payload
    buf += struct.pack("<i", footer_payload_size)
    buf += struct.pack("<i", 0)  # flags: uncompressed
    buf += PUFFIN_MAGIC

    return buf


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(output_dir, exist_ok=True)

    # File 1: 32-bit only — even indices in [0, 1000)
    positions_32 = [i for i in range(0, 1000, 2)]
    blob_data_32 = serialize_deletion_vector_v1(positions_32)
    meta_32 = {
        "type": "deletion-vector-v1",
        "fields": [],
        "snapshot-id": -1,
        "sequence-number": -1,
        "offset": 0,
        "length": 0,
        "properties": {
            "referenced-data-file": "s3://bucket/table/data/file1.parquet",
            "cardinality": str(len(positions_32)),
        },
    }
    puffin_32 = make_puffin_file([blob_data_32], [meta_32])
    path_32 = os.path.join(output_dir, "dv_32bit.puffin")
    with open(path_32, "wb") as f:
        f.write(puffin_32)
    print(f"Wrote {path_32} ({len(puffin_32)} bytes)")

    # File 2: 64-bit — even indices in [0, 1000) AND [2^32, 2^32 + 1000)
    positions_64 = [i for i in range(0, 1000, 2)]
    positions_64 += [(1 << 32) + i for i in range(0, 1000, 2)]
    blob_data_64 = serialize_deletion_vector_v1(positions_64)
    meta_64 = {
        "type": "deletion-vector-v1",
        "fields": [],
        "snapshot-id": -1,
        "sequence-number": -1,
        "offset": 0,
        "length": 0,
        "properties": {
            "referenced-data-file": "s3://bucket/table/data/file2.parquet",
            "cardinality": str(len(positions_64)),
        },
    }
    puffin_64 = make_puffin_file([blob_data_64], [meta_64])
    path_64 = os.path.join(output_dir, "dv_64bit.puffin")
    with open(path_64, "wb") as f:
        f.write(puffin_64)
    print(f"Wrote {path_64} ({len(puffin_64)} bytes)")


if __name__ == "__main__":
    main()
