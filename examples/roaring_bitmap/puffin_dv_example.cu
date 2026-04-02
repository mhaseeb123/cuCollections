/*
 * Copyright (c) 2025 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file puffin_dv_example.cu
 * @brief Reads a deletion-vector-v1 blob from a Puffin file and queries it using a single
 *        cuco::experimental::roaring_bitmap<uint64_t> on the GPU.
 *
 * The deletion-vector-v1 vector payload (after stripping the 4-byte big-endian length,
 * 4-byte magic, and 4-byte CRC wrapper) is already in the 64-bit Roaring "portable" format
 * that roaring_bitmap<uint64_t> expects. This lets us query all keys in one shot.
 *
 * Usage:
 *   puffin_dv_example <puffin_file> <blob_offset> <blob_size>
 *                     <num_keys> <key0> <range_min0> <range_max0> [<key1> <range_min1> <range_max1> ...]
 *
 * For each key, the program queries all indices in [range_min, range_max) against the
 * 64-bit roaring bitmap (as (key << 32) | index) and verifies that all even indices are
 * marked as deleted and all odd indices are not.
 */

#include <cuco/roaring_bitmap.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

static constexpr cuda::std::uint8_t dv_magic[] = {0xD1, 0xD3, 0x39, 0x64};

struct key_range {
  cuda::std::uint32_t key;
  cuda::std::uint32_t range_min;
  cuda::std::uint32_t range_max;
};

cuda::std::uint32_t read_u32_be(cuda::std::byte const* p)
{
  auto const* u = reinterpret_cast<cuda::std::uint8_t const*>(p);
  return (static_cast<cuda::std::uint32_t>(u[0]) << 24) |
         (static_cast<cuda::std::uint32_t>(u[1]) << 16) |
         (static_cast<cuda::std::uint32_t>(u[2]) << 8) | static_cast<cuda::std::uint32_t>(u[3]);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 7) {
    std::cerr
      << "Usage: " << argv[0]
      << " <puffin_file> <blob_offset> <blob_size> <num_keys> <key0> <range_min0> <range_max0> ..."
      << std::endl;
    return 1;
  }

  std::string const puffin_path = argv[1];
  auto const blob_offset        = std::stoull(argv[2]);
  auto const blob_size          = std::stoull(argv[3]);
  auto const num_keys           = std::stoul(argv[4]);

  std::vector<key_range> ranges;
  for (cuda::std::uint32_t i = 0; i < num_keys; ++i) {
    int const base = 5 + i * 3;
    if (base + 2 >= argc) {
      std::cerr << "Not enough arguments for key " << i << std::endl;
      return 1;
    }
    key_range kr;
    kr.key       = std::stoul(argv[base]);
    kr.range_min = std::stoul(argv[base + 1]);
    kr.range_max = std::stoul(argv[base + 2]);
    ranges.push_back(kr);
  }

  // Read the blob from the puffin file
  std::ifstream file(puffin_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << puffin_path << std::endl;
    return 1;
  }

  std::vector<cuda::std::byte> blob_buf(blob_size);
  file.seekg(static_cast<std::streamoff>(blob_offset));
  file.read(reinterpret_cast<char*>(blob_buf.data()), static_cast<std::streamsize>(blob_size));
  file.close();

  // Parse the deletion-vector-v1 wrapper:
  //   [4B BE combined_length] [4B magic] [vector payload ...] [4B BE CRC]
  cuda::std::byte const* p              = blob_buf.data();
  cuda::std::uint32_t const combined_length = read_u32_be(p);
  p += 4;

  if (std::memcmp(p, dv_magic, 4) != 0) {
    std::cerr << "Invalid deletion vector magic" << std::endl;
    return 1;
  }
  p += 4;  // skip magic

  // `p` now points at the vector payload, which is already in the 64-bit Roaring
  // "portable" format: [num_buckets(u64 LE)] [key(u32 LE) + 32-bit roaring] × N
  // This is exactly what roaring_bitmap<uint64_t> expects.
  cuda::std::byte const* vector_payload = p;

  // Construct a single 64-bit roaring bitmap from the vector payload
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t> bitmap(vector_payload);

  std::cout << "Deletion vector: " << bitmap.size() << " deleted positions, " << bitmap.size_bytes()
            << " bytes" << std::endl;

  // Build all query keys across all key ranges, forming 64-bit positions: (key << 32) | index
  std::vector<cuda::std::uint64_t> all_keys_h;
  std::vector<cuda::std::size_t> range_offsets;  // start index into all_keys_h for each range
  for (auto const& kr : ranges) {
    range_offsets.push_back(all_keys_h.size());
    for (cuda::std::uint32_t idx = kr.range_min; idx < kr.range_max; ++idx) {
      cuda::std::uint64_t pos = (static_cast<cuda::std::uint64_t>(kr.key) << 32) | idx;
      all_keys_h.push_back(pos);
    }
  }
  range_offsets.push_back(all_keys_h.size());

  // Single bulk query for all keys at once
  thrust::device_vector<cuda::std::uint64_t> all_keys_d(all_keys_h.begin(), all_keys_h.end());
  thrust::device_vector<bool> all_results_d(all_keys_h.size(), false);

  bitmap.contains(all_keys_d.begin(), all_keys_d.end(), all_results_d.begin());

  thrust::host_vector<bool> all_results_h = all_results_d;

  // Verify results per key range
  bool all_ok = true;
  for (cuda::std::size_t r = 0; r < ranges.size(); ++r) {
    auto const& kr              = ranges[r];
    cuda::std::size_t const beg = range_offsets[r];
    cuda::std::size_t const end = range_offsets[r + 1];

    std::cout << "Verifying key=" << kr.key << " range=[" << kr.range_min << ", " << kr.range_max
              << ")" << std::endl;

    bool range_ok = true;
    for (cuda::std::size_t i = beg; i < end; ++i) {
      cuda::std::uint32_t idx = static_cast<cuda::std::uint32_t>(all_keys_h[i] & 0xFFFFFFFF);
      bool const expected     = (idx % 2 == 0);
      bool const actual       = all_results_h[i];
      if (expected != actual) {
        std::cerr << "  MISMATCH at key=" << kr.key << " index=" << idx << ": expected="
                  << std::boolalpha << expected << " got=" << actual << std::endl;
        range_ok = false;
      }
    }

    if (range_ok) {
      std::cout << "  PASS" << std::endl;
    } else {
      std::cerr << "  FAIL" << std::endl;
      all_ok = false;
    }
  }

  std::cout << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << std::endl;
  return all_ok ? 0 : 1;
}
