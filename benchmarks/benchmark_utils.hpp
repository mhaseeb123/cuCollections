/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cuco/detail/error.hpp>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

namespace cuco::benchmark {

template <typename Dist>
auto dist_from_state(nvbench::state const& state)
{
  if constexpr (std::is_same_v<Dist, cuco::utility::distribution::unique>) {
    return Dist{};
  } else if constexpr (std::is_same_v<Dist, cuco::utility::distribution::uniform>) {
    auto const multiplicity = state.get_int64("Multiplicity");
    return Dist{multiplicity};
  } else if constexpr (std::is_same_v<Dist, cuco::utility::distribution::gaussian>) {
    auto const skew = state.get_float64("Skew");
    return Dist{skew};
  } else {
    CUCO_FAIL("Unexpected distribution type");
  }
}

template <typename T, typename NewType>
struct rebind_hasher;

template <template <typename> class Template, typename OldType, typename NewType>
struct rebind_hasher<Template<OldType>, NewType> {
  using type = Template<NewType>;
};

template <typename T, typename NewType>
using rebind_hasher_t = typename rebind_hasher<T, NewType>::type;

}  // namespace cuco::benchmark

NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::unique, "UNIQUE", "distribution::unique");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::uniform,
                             "UNIFORM",
                             "distribution::uniform");
NVBENCH_DECLARE_TYPE_STRINGS(cuco::utility::distribution::gaussian,
                             "GAUSSIAN",
                             "distribution::gaussian");
