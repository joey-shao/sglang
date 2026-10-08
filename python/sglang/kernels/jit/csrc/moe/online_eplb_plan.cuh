#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <climits>
#include <cstdint>

namespace sglang {

struct OnlineEplbPlanParams {
  const int32_t* counts;
  int32_t* mapping;
  int32_t* physical_ids;
  int32_t* quota_end;
  int32_t* instance_count;
  int32_t* source_prefix;
  uint32_t ep_size;
  uint32_t num_experts;
  uint32_t slots;
  uint32_t rank;
  uint32_t min_quota;
};

namespace online_eplb_plan {

struct Export {
  int32_t expert;
  int32_t target;
  int32_t quota;
};

struct Candidate {
  Export exports[64];
  int32_t count;
  int64_t max_load;
  int64_t moved;
};

struct Workspace {
  int32_t global_counts[512];
  int32_t expert_order[512];
  int32_t remaining[512];
  uint32_t occupied[512];
  int32_t source_order[8];
  int64_t initial_loads[8];
  int64_t loads[8];
  int64_t need[8];
  int64_t slack[8];
  int32_t used[8];
  Candidate candidate;
  Candidate best;
};

SGL_DEVICE bool export_less(const Export& a, const Export& b) {
  if (a.expert != b.expert) return a.expert < b.expert;
  if (a.target != b.target) return a.target < b.target;
  return a.quota < b.quota;
}

SGL_DEVICE bool better(const Candidate& candidate, const Candidate& best) {
  if (candidate.max_load != best.max_load) return candidate.max_load < best.max_load;
  if (candidate.count != best.count) return candidate.count < best.count;
  if (candidate.moved != best.moved) return candidate.moved < best.moved;
  // Canonical (expert, target, quota) tuples define the placement/quota lexicographic order.
  for (int32_t i = 0; i < candidate.count; ++i) {
    if (export_less(candidate.exports[i], best.exports[i])) return true;
    if (export_less(best.exports[i], candidate.exports[i])) return false;
  }
  return false;
}

SGL_DEVICE bool try_plan(const OnlineEplbPlanParams& p, Workspace& w, int64_t threshold) {
  auto& candidate = w.candidate;
  const uint32_t lane = device::get_lane_id();
  if (lane == 0) {
    candidate.count = 0;
    candidate.moved = 0;
  }
  for (uint32_t e = lane; e < p.num_experts; e += 32) {
    w.remaining[e] = w.global_counts[e];
    w.occupied[e] = 0;
  }
  for (uint32_t r = lane; r < p.ep_size; r += 32) {
    w.loads[r] = w.initial_loads[r];
    w.need[r] = w.loads[r] > threshold ? w.loads[r] - threshold : 0;
    w.slack[r] = w.loads[r] < threshold ? threshold - w.loads[r] : 0;
    w.used[r] = 0;
  }
  __syncwarp();
  const uint32_t masters = p.num_experts / p.ep_size;
  for (uint32_t sr = 0; sr < p.ep_size; ++sr) {
    const int32_t source = w.source_order[sr];
    for (uint32_t i = 0; i < masters && w.need[source] > 0; ++i) {
      const int32_t expert = w.expert_order[source * masters + i];
      while (w.need[source] > 0 && w.remaining[expert] >= static_cast<int32_t>(p.min_quota)) {
        // Each lane owns a rank candidate. The packed key preserves the
        // descending-slack / ascending-rank tie break without serial scanning.
        int64_t key = -1;
        if (lane < p.ep_size && lane != static_cast<uint32_t>(source) && w.used[lane] < static_cast<int32_t>(p.slots) &&
            !(w.occupied[expert] & (1u << lane)) && w.slack[lane] >= p.min_quota) {
          key = w.slack[lane] * p.ep_size + (p.ep_size - 1 - lane);
        }
        key = device::warp::reduce_max(key);
        const int32_t target = key < 0 ? -1 : p.ep_size - 1 - key % p.ep_size;
        if (target < 0) break;
        if (lane == 0) {
          const int64_t cap = w.slack[target] < w.remaining[expert] ? w.slack[target] : w.remaining[expert];
          const int64_t requested = w.need[source] > p.min_quota ? w.need[source] : p.min_quota;
          const int32_t quota = static_cast<int32_t>(cap < requested ? cap : requested);
          candidate.exports[candidate.count++] = {expert, target, quota};
          candidate.moved += quota;
          w.remaining[expert] -= quota;
          w.need[source] = w.need[source] > quota ? w.need[source] - quota : 0;
          w.slack[target] -= quota;
          w.loads[source] -= quota;
          w.loads[target] += quota;
          w.occupied[expert] |= 1u << target;
          ++w.used[target];
        }
        __syncwarp();
      }
    }
    if (w.need[source] > 0) return false;
  }
  if (lane == 0) {
    candidate.max_load = 0;
    for (uint32_t r = 0; r < p.ep_size; ++r) {
      if (w.loads[r] > candidate.max_load) candidate.max_load = w.loads[r];
    }
    for (int32_t i = 1; i < candidate.count; ++i) {
      const Export value = candidate.exports[i];
      int32_t j = i;
      while (j > 0 && export_less(value, candidate.exports[j - 1])) {
        candidate.exports[j] = candidate.exports[j - 1];
        --j;
      }
      candidate.exports[j] = value;
    }
  }
  __syncwarp();
  return true;
}

SGL_DEVICE void search(const OnlineEplbPlanParams& p, Workspace& w) {
  int64_t total = 0, upper = 0;
  const uint32_t lane = device::get_lane_id();
  for (uint32_t r = 0; r < p.ep_size; ++r) {
    total += w.initial_loads[r];
    if (w.initial_loads[r] > upper) upper = w.initial_loads[r];
  }
  assert(total <= INT32_MAX);
  if (lane == 0) {
    w.best.count = 0;
    w.best.moved = 0;
    w.best.max_load = upper;
  }
  __syncwarp();
  const int64_t lower = (total + p.ep_size - 1) / p.ep_size;
  if (total == 0 || lower == upper) return;
  const int64_t ideal = (101 * lower + 99) / 100;
  const int64_t first = ideal < upper - 1 ? ideal : upper - 1;
  if (try_plan(p, w, first)) {
    if (lane == 0 && better(w.candidate, w.best)) w.best = w.candidate;
    __syncwarp();
    return;
  }
  int64_t lo = first + 1, hi = upper - 1;
  for (uint32_t evaluation = 0; evaluation < 15 && lo <= hi; ++evaluation) {
    const int64_t threshold = lo + (hi - lo) / 2;
    if (try_plan(p, w, threshold)) {
      if (lane == 0 && better(w.candidate, w.best)) w.best = w.candidate;
      __syncwarp();
      hi = threshold - 1 < w.candidate.max_load - 1 ? threshold - 1 : w.candidate.max_load - 1;
    } else {
      lo = threshold + 1;
    }
  }
}

}  // namespace online_eplb_plan

__global__ void online_eplb_plan_kernel(const OnlineEplbPlanParams p) {
  __shared__ online_eplb_plan::Workspace w;
  const uint32_t masters = p.num_experts / p.ep_size;
  const uint32_t physical_per_rank = masters + p.slots;

  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    int64_t global = 0, prefix = 0;
    for (uint32_t r = 0; r < p.ep_size; ++r) {
      const int32_t count = p.counts[r * p.num_experts + e];
      global += count;
      if (r < p.rank) prefix += count;
    }
    w.global_counts[e] = static_cast<int32_t>(global);
    p.source_prefix[e] = static_cast<int32_t>(prefix);
  }
  __syncthreads();

  // Each expert independently computes its stable position in its owner's shard.
  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    const uint32_t base = e / masters * masters;
    uint32_t position = 0;
    for (uint32_t other = base; other < base + masters; ++other) {
      position +=
          w.global_counts[other] > w.global_counts[e] || (w.global_counts[other] == w.global_counts[e] && other < e);
    }
    w.expert_order[base + position] = static_cast<int32_t>(e);
  }
  if (threadIdx.x < p.ep_size) {
    const uint32_t r = threadIdx.x;
    int64_t load = 0;
    for (uint32_t e = r * masters; e < (r + 1) * masters; ++e)
      load += w.global_counts[e];
    w.initial_loads[r] = load;
  }
  __syncthreads();

  if (threadIdx.x < p.ep_size) {
    const uint32_t r = threadIdx.x;
    uint32_t position = 0;
    for (uint32_t other = 0; other < p.ep_size; ++other) {
      position +=
          w.initial_loads[other] > w.initial_loads[r] || (w.initial_loads[other] == w.initial_loads[r] && other < r);
    }
    w.source_order[position] = static_cast<int32_t>(r);
  }
  __syncthreads();

  if (threadIdx.x < 32) 
    online_eplb_plan::search(p, w);
  __syncthreads();

  for (uint32_t slot = threadIdx.x; slot < p.ep_size * p.slots; slot += blockDim.x)
    p.mapping[slot] = -1;
  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    int32_t master_quota = w.global_counts[e];
    for (int32_t i = 0; i < w.best.count; ++i) {
      if (w.best.exports[i].expert == e) master_quota -= w.best.exports[i].quota;
    }
    p.physical_ids[e * p.ep_size] = static_cast<int32_t>((e / masters) * physical_per_rank + e % masters);
    p.quota_end[e * p.ep_size] = master_quota;
    int32_t instance = 1, end = master_quota;
    for (int32_t i = 0; i < w.best.count; ++i) {
      const auto entry = w.best.exports[i];
      if (entry.expert != e) continue;
      int32_t slot = 0;
      for (int32_t j = 0; j < w.best.count; ++j) {
        slot += w.best.exports[j].target == entry.target && w.best.exports[j].expert < e;
      }
      // Mapping is written below, after the CTA-wide empty-slot initialization.
      end += entry.quota;
      p.physical_ids[e * p.ep_size + instance] = entry.target * physical_per_rank + masters + slot;
      p.quota_end[e * p.ep_size + instance++] = end;
    }
    p.instance_count[e] = instance;
    for (uint32_t j = instance; j < p.ep_size; ++j) {
      p.physical_ids[e * p.ep_size + j] = -1;
      p.quota_end[e * p.ep_size + j] = w.global_counts[e];
    }
  }
  __syncthreads();
  
  for (int32_t i = threadIdx.x; i < w.best.count; i += blockDim.x) {
    const auto entry = w.best.exports[i];
    int32_t slot = 0;
    for (int32_t j = 0; j < w.best.count; ++j) {
      slot += w.best.exports[j].target == entry.target && w.best.exports[j].expert < entry.expert;
    }
    p.mapping[entry.target * p.slots + slot] = entry.expert;
  }
}

struct OnlineEplbPlanKernel {
  static void
  run(tvm::ffi::TensorView counts,
      tvm::ffi::TensorView mapping,
      tvm::ffi::TensorView physical_ids,
      tvm::ffi::TensorView quota_end,
      tvm::ffi::TensorView instance_count,
      tvm::ffi::TensorView source_prefix,
      int64_t rank,
      int64_t min_quota,
      int64_t slots) {
    using namespace host;

    SymbolicSize P{"ep_size"}, E{"num_experts"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({P, E}).with_dtype<int32_t>().with_device(device).verify(counts);
    TensorMatcher({P, slots}).with_dtype<int32_t>().with_device(device).verify(mapping);
    TensorMatcher({E, P}).with_dtype<int32_t>().with_device(device).verify(physical_ids).verify(quota_end);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(instance_count).verify(source_prefix);

    const OnlineEplbPlanParams params{
        static_cast<const int32_t*>(counts.data_ptr()),
        static_cast<int32_t*>(mapping.data_ptr()),
        static_cast<int32_t*>(physical_ids.data_ptr()),
        static_cast<int32_t*>(quota_end.data_ptr()),
        static_cast<int32_t*>(instance_count.data_ptr()),
        static_cast<int32_t*>(source_prefix.data_ptr()),
        static_cast<uint32_t>(P.unwrap()),
        static_cast<uint32_t>(E.unwrap()),
        static_cast<uint32_t>(slots),
        static_cast<uint32_t>(rank),
        static_cast<uint32_t>(min_quota)};

    LaunchKernel(1, 128, device.unwrap())(online_eplb_plan_kernel, params);
  }
};

}  // namespace sglang
