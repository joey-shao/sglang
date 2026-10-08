#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <climits>
#include <cstdint>

#ifndef ONLINE_EPLB_BLOCK_SIZE
#error "ONLINE_EPLB_BLOCK_SIZE must be provided by the online EPLB JIT wrapper"
#endif

namespace sglang {

static_assert(
    ONLINE_EPLB_BLOCK_SIZE >= device::kWarpThreads && ONLINE_EPLB_BLOCK_SIZE <= 1024 &&
    ONLINE_EPLB_BLOCK_SIZE % device::kWarpThreads == 0);
inline constexpr uint32_t kOnlineEplbNumWarps = ONLINE_EPLB_BLOCK_SIZE / device::kWarpThreads;

template <typename IdT>
__global__ void online_eplb_histogram_kernel(
    const IdT* __restrict__ ids, int32_t* __restrict__ block_counts, uint32_t n, uint32_t num_experts) {
  const uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int32_t bins[];

  for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    bins[e] = 0;
  }
  __syncthreads();

  if (flat < n) {
    const int64_t id = static_cast<int64_t>(ids[flat]);
    if (id >= 0 && id < static_cast<int64_t>(num_experts)) {
      atomicAdd(&bins[id], 1);
    }
  }
  __syncthreads();

  for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    block_counts[static_cast<int64_t>(blockIdx.x) * num_experts + e] = bins[e];
  }
}

__global__ void online_eplb_prefix_kernel(
    const int32_t* __restrict__ block_counts,
    int32_t* __restrict__ block_prefix,
    int32_t* __restrict__ local_counts,
    uint32_t blocks,
    uint32_t num_experts) {
  const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;

  if (e >= num_experts) {
    return;
  }

  int32_t acc = 0;
  for (uint32_t b = 0; b < blocks; ++b) {
    block_prefix[static_cast<int64_t>(b) * num_experts + e] = acc;
    acc += block_counts[static_cast<int64_t>(b) * num_experts + e];
  }
  local_counts[e] = acc;
}

template <typename IdT>
__global__ void online_eplb_remap_kernel(
    const IdT* __restrict__ ids,
    int64_t* __restrict__ out,
    const int32_t* __restrict__ block_prefix,
    const int32_t* __restrict__ source_prefix,
    const int32_t* __restrict__ quota_end,
    const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ instance_count,
    uint32_t n,
    uint32_t num_experts,
    uint32_t ep_size) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = device::get_lane_id();
  const uint32_t warp = threadIdx.x / device::kWarpThreads;
  const int64_t raw = i < n ? static_cast<int64_t>(ids[i]) : -1;
  const bool valid = raw >= 0 && raw < static_cast<int64_t>(num_experts);
  const uint32_t e = valid ? static_cast<uint32_t>(raw) : UINT32_MAX;
  extern __shared__ int32_t warp_counts[];

  for (uint32_t j = threadIdx.x; j < kOnlineEplbNumWarps * num_experts; j += blockDim.x)
    warp_counts[j] = 0;
  __syncthreads();

  // All 32 lanes participate, including invalid IDs and the partial-tile tail.
  // One leader writes the histogram for each (warp, expert) peer group.
  const uint32_t peers = __match_any_sync(0xffffffffu, e);
  if (valid && lane == static_cast<uint32_t>(__ffs(peers) - 1)) 
    warp_counts[warp * num_experts + e] = __popc(peers);
  __syncthreads();

  if (i >= n) return;
  if (!valid) {
    out[i] = -1;
    return;
  }

  const uint32_t tile = blockIdx.x;
  int32_t before = __popc(peers & ((1u << lane) - 1u));
  for (uint32_t w = 0; w < warp; ++w)
    before += warp_counts[w * num_experts + e];
  const int32_t ordinal = source_prefix[e] + block_prefix[static_cast<int64_t>(tile) * num_experts + e] + before;
  uint32_t instance = 0;
  for (uint32_t j = 0; j < instance_count[e]; ++j) {
    if (quota_end[static_cast<int64_t>(e) * ep_size + j] > ordinal) {
      instance = j;
      break;
    }
  }
  out[i] = static_cast<int64_t>(physical_ids[static_cast<int64_t>(e) * ep_size + instance]);
}

template <typename IdT>
struct OnlineEplbRouteKernel {
  static void histogram(tvm::ffi::TensorView ids, tvm::ffi::TensorView block_counts, uint32_t num_experts) {
    using namespace host;

    SymbolicSize n{"n"};
    SymbolicSize blocks{"blocks"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({n}).with_dtype<IdT>().with_device(device).verify(ids);
    TensorMatcher({blocks, static_cast<int64_t>(num_experts)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(block_counts);

    LaunchKernel(
        static_cast<uint32_t>(blocks.unwrap()), ONLINE_EPLB_BLOCK_SIZE, device.unwrap(), num_experts * sizeof(int32_t))(
        online_eplb_histogram_kernel<IdT>,
        static_cast<const IdT*>(ids.data_ptr()),
        static_cast<int32_t*>(block_counts.data_ptr()),
        static_cast<uint32_t>(n.unwrap()),
        num_experts);
  }

  static void prefix(
      tvm::ffi::TensorView block_counts,
      tvm::ffi::TensorView block_prefix,
      tvm::ffi::TensorView local_counts,
      uint32_t num_experts) {
    using namespace host;

    SymbolicSize blocks{"blocks"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({blocks, static_cast<int64_t>(num_experts)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(block_counts)
        .verify(block_prefix);
    TensorMatcher({static_cast<int64_t>(num_experts)}).with_dtype<int32_t>().with_device(device).verify(local_counts);

    LaunchKernel((num_experts + 127u) / 128u, 128, device.unwrap())(
        online_eplb_prefix_kernel,
        static_cast<const int32_t*>(block_counts.data_ptr()),
        static_cast<int32_t*>(block_prefix.data_ptr()),
        static_cast<int32_t*>(local_counts.data_ptr()),
        static_cast<uint32_t>(blocks.unwrap()),
        num_experts);
  }

  static void remap(
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView block_prefix,
      tvm::ffi::TensorView source_prefix,
      tvm::ffi::TensorView quota_end,
      tvm::ffi::TensorView physical_ids,
      tvm::ffi::TensorView instance_count,
      uint32_t num_experts,
      uint32_t ep_size) {
    using namespace host;

    SymbolicSize n{"n"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({n}).with_dtype<IdT>().with_device(device).verify(ids);
    TensorMatcher({n}).with_dtype<int64_t>().with_device(device).verify(out);
    TensorMatcher({static_cast<int64_t>(num_experts)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(source_prefix)
        .verify(instance_count);
    TensorMatcher({static_cast<int64_t>(num_experts), static_cast<int64_t>(ep_size)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(quota_end)
        .verify(physical_ids);

    const uint32_t count = static_cast<uint32_t>(n.unwrap());
    const uint32_t blocks = std::max<uint32_t>(1, (count + ONLINE_EPLB_BLOCK_SIZE - 1) / ONLINE_EPLB_BLOCK_SIZE);
    TensorMatcher({static_cast<int64_t>(blocks), static_cast<int64_t>(num_experts)})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(block_prefix);

    if (count == 0) return;

    LaunchKernel(blocks, ONLINE_EPLB_BLOCK_SIZE, device.unwrap(), kOnlineEplbNumWarps * num_experts * sizeof(int32_t))(
        online_eplb_remap_kernel<IdT>,
        static_cast<const IdT*>(ids.data_ptr()),
        static_cast<int64_t*>(out.data_ptr()),
        static_cast<const int32_t*>(block_prefix.data_ptr()),
        static_cast<const int32_t*>(source_prefix.data_ptr()),
        static_cast<const int32_t*>(quota_end.data_ptr()),
        static_cast<const int32_t*>(physical_ids.data_ptr()),
        static_cast<const int32_t*>(instance_count.data_ptr()),
        count,
        num_experts,
        ep_size);
  }
};

}  // namespace sglang
