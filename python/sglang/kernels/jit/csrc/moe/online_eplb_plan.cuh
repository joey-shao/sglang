#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

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

struct PlanMeta {
  int32_t global_counts[512];
  int32_t expert_order[512];
  int32_t source_order[8];
  int64_t initial_loads[8];
};

// Mutable solver state belongs to one warp; PlanMeta is shared by the CTA.
struct WarpWorkspace {
  int32_t remaining[512];
  uint32_t occupied[512];
  int64_t loads[8];
  int64_t need[8];
  int64_t slack[8];
  int32_t used[8];
  Candidate candidate;
  bool success;
};

inline constexpr uint32_t kMaxEvaluations = 16;

struct SearchState {
  int64_t lower;
  int64_t upper;
  int64_t good_threshold;
  uint32_t evaluations;
  uint32_t active_warps;
  int32_t winner;
  bool done;
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

SGL_DEVICE bool try_plan(const OnlineEplbPlanParams& p, const PlanMeta& meta, WarpWorkspace& w, int64_t threshold) {
  auto& candidate = w.candidate;
  const uint32_t lane = device::get_lane_id();

  // A failed previous trial may return before its final barrier. Join every
  // lane before overwriting the rank/expert state that trial was still reading.
  __syncwarp();
  if (lane == 0) {
    candidate.count = 0;
    candidate.moved = 0;
  }
  for (uint32_t e = lane; e < p.num_experts; e += 32) {
    w.remaining[e] = meta.global_counts[e];
    w.occupied[e] = 0;
  }
  for (uint32_t r = lane; r < p.ep_size; r += 32) {
    w.loads[r] = meta.initial_loads[r];
    w.need[r] = w.loads[r] > threshold ? w.loads[r] - threshold : 0;
    w.slack[r] = w.loads[r] < threshold ? threshold - w.loads[r] : 0;
    w.used[r] = 0;
  }
  __syncwarp();

  const uint32_t masters = p.num_experts / p.ep_size;
  for (uint32_t sr = 0; sr < p.ep_size; ++sr) {
    const int32_t source = meta.source_order[sr];
    for (uint32_t i = 0; i < masters && w.need[source] > 0; ++i) {
      const int32_t expert = meta.expert_order[source * masters + i];
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

// Cooperatively copy only initialized exports within the warp.
SGL_DEVICE void copy_candidate(Candidate& dst, const Candidate& src) {
  const uint32_t lane = device::get_lane_id();
  for (int32_t i = lane; i < src.count; i += 32)
    dst.exports[i] = src.exports[i];
  if (lane == 0) {
    dst.count = src.count;
    dst.max_load = src.max_load;
    dst.moved = src.moved;
  }
  __syncwarp();
}

// Called by the CTA controller (thread 0), after Prologue has built metadata.
SGL_DEVICE void init_search(const OnlineEplbPlanParams& p, const PlanMeta& meta, SearchState& search) {
  int64_t total = 0, upper = 0;
  for (uint32_t r = 0; r < p.ep_size; ++r) {
    total += meta.initial_loads[r];
    if (meta.initial_loads[r] > upper) upper = meta.initial_loads[r];
  }
  search.best.count = 0;
  search.best.moved = 0;
  search.best.max_load = upper;
  search.evaluations = 0;
  search.active_warps = 0;
  search.winner = -1;
  const int64_t lower = (total + p.ep_size - 1) / p.ep_size;
  search.good_threshold = (101 * lower + 99) / 100;
  // Keep the interval below the first near-ideal probe available when a good
  // load balance still requires too many replicas for early stopping.
  search.lower = lower;
  search.upper = upper - 1;
  search.done = total == 0 || lower == upper;
}

// Thread 0 generates a batch of distinct thresholds within the shared bounds.
template <uint32_t BlockSize>
SGL_DEVICE void generate_threshold_batch(SearchState& search, int64_t* thresholds, bool first_round) {
  constexpr uint32_t kWarps = BlockSize / 32;
  const int64_t first = search.good_threshold < search.upper ? search.good_threshold : search.upper;
  const int64_t base = first_round ? first : search.lower;
  const int64_t width = search.upper - base + 1;
  uint32_t active = kMaxEvaluations - search.evaluations;
  if (active > kWarps) active = kWarps;
  if (active > width) active = static_cast<uint32_t>(width);
  search.active_warps = active;
  search.evaluations += active;

  for (uint32_t w = 0; w < active; ++w) {
    if (first_round) {
      // Warp 0 probes the original near-ideal threshold. With four warps,
      // the others probe 1/8, 1/4 and 1/2 of the range, biased toward low loads.
      int64_t threshold = base;
      if (w > 0) {
        threshold += (width - 1) >> (active - w);
        if (threshold <= thresholds[w - 1]) threshold = thresholds[w - 1] + 1;
      }
      thresholds[w] = threshold;
    } else {
      // Evenly spaced interior probes; width <= kWarps enumerates the range.
      // A single active warp uses the original lower midpoint.
      thresholds[w] = search.lower + (w + 1) * (width + 1) / (active + 1) - 1;
    }
  }
}

// Thread 0 reduces the entire batch before deciding the next shared interval.
SGL_DEVICE void
score_round(const OnlineEplbPlanParams& p, const WarpWorkspace* warps, const int64_t* thresholds, SearchState& search) {
  search.winner = -1;
  int64_t upper = search.upper;
  for (uint32_t w = 0; w < search.active_warps; ++w) {
    if (!warps[w].success) continue;
    const Candidate& candidate = warps[w].candidate;
    const Candidate& best = search.winner < 0 ? search.best : warps[search.winner].candidate;
    if (better(candidate, best)) search.winner = static_cast<int32_t>(w);
    if (thresholds[w] - 1 < upper) upper = thresholds[w] - 1;
    if (candidate.max_load - 1 < upper) upper = candidate.max_load - 1;
  }
  int64_t lower = search.lower;
  for (uint32_t w = 0; w < search.active_warps; ++w) {
    // The greedy oracle is not monotonic. A failure above an observed success
    // must not discard the remaining lower interval.
    if (!warps[w].success && thresholds[w] <= upper && thresholds[w] >= lower) lower = thresholds[w] + 1;
  }
  search.lower = lower;
  search.upper = upper;
  const Candidate& best = search.winner < 0 ? search.best : warps[search.winner].candidate;
  // Candidate::count is the total number of new replicas across all EP ranks.
  const bool good_plan = best.max_load <= search.good_threshold && best.count <= static_cast<int32_t>(p.ep_size);
  search.done = good_plan || lower > upper || search.evaluations == kMaxEvaluations;
}

// Prologue: compute immutable metadata.
SGL_DEVICE void prologue(const OnlineEplbPlanParams& p, PlanMeta& meta) {
  const uint32_t masters = p.num_experts / p.ep_size;

  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    int64_t global = 0, prefix = 0;
    for (uint32_t r = 0; r < p.ep_size; ++r) {
      const int32_t count = p.counts[r * p.num_experts + e];
      global += count;
      if (r < p.rank) prefix += count;
    }
    meta.global_counts[e] = static_cast<int32_t>(global);
    p.source_prefix[e] = static_cast<int32_t>(prefix);
  }
  __syncthreads();

  // Each expert independently computes its stable position in its owner's shard.
  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    const uint32_t base = e / masters * masters;
    uint32_t position = 0;
    for (uint32_t other = base; other < base + masters; ++other) {
      position += meta.global_counts[other] > meta.global_counts[e] ||
                  (meta.global_counts[other] == meta.global_counts[e] && other < e);
    }
    meta.expert_order[base + position] = static_cast<int32_t>(e);
  }

  if (threadIdx.x < p.ep_size) {
    const uint32_t r = threadIdx.x;
    int64_t load = 0;
    for (uint32_t e = r * masters; e < (r + 1) * masters; ++e)
      load += meta.global_counts[e];
    meta.initial_loads[r] = load;
  }
  __syncthreads();

  if (threadIdx.x < p.ep_size) {
    const uint32_t r = threadIdx.x;
    uint32_t position = 0;
    for (uint32_t other = 0; other < p.ep_size; ++other) {
      position += meta.initial_loads[other] > meta.initial_loads[r] ||
                  (meta.initial_loads[other] == meta.initial_loads[r] && other < r);
    }
    meta.source_order[position] = static_cast<int32_t>(r);
  }
  __syncthreads();
}

// PlanVerifyAndScore: each active warp evaluates exactly one candidate per round.
SGL_DEVICE void plan_verify_and_score(
    const OnlineEplbPlanParams& p,
    const PlanMeta& meta,
    WarpWorkspace* warps,
    const int64_t* thresholds,
    SearchState& search) {
  const uint32_t warp = threadIdx.x / 32;
  bool success = false;
  if (warp < search.active_warps) success = try_plan(p, meta, warps[warp], thresholds[warp]);
  if (device::get_lane_id() == 0) warps[warp].success = success;
  __syncthreads();

  if (threadIdx.x == 0) score_round(p, warps, thresholds, search);
  __syncthreads();
  // At most one candidate copy per round; the next batch cannot overwrite its
  // source until every lane has finished copying it into the CTA-level best.
  if (threadIdx.x < 32 && search.winner >= 0) copy_candidate(search.best, warps[search.winner].candidate);
  __syncthreads();
}

// Epilogue: materialize the single CTA-level best candidate.
SGL_DEVICE void epilogue(const OnlineEplbPlanParams& p, const PlanMeta& meta, const Candidate& best) {
  const uint32_t masters = p.num_experts / p.ep_size;
  const uint32_t physical_per_rank = masters + p.slots;

  for (uint32_t slot = threadIdx.x; slot < p.ep_size * p.slots; slot += blockDim.x)
    p.mapping[slot] = -1;
  for (uint32_t e = threadIdx.x; e < p.num_experts; e += blockDim.x) {
    int32_t master_quota = meta.global_counts[e];
    for (int32_t i = 0; i < best.count; ++i) {
      if (best.exports[i].expert == e) master_quota -= best.exports[i].quota;
    }
    p.physical_ids[e * p.ep_size] = static_cast<int32_t>((e / masters) * physical_per_rank + e % masters);
    p.quota_end[e * p.ep_size] = master_quota;
    int32_t instance = 1, end = master_quota;
    for (int32_t i = 0; i < best.count; ++i) {
      const auto entry = best.exports[i];
      if (entry.expert != e) continue;
      int32_t slot = 0;
      for (int32_t j = 0; j < best.count; ++j) {
        slot += best.exports[j].target == entry.target && best.exports[j].expert < e;
      }
      // Mapping is written below, after the CTA-wide empty-slot initialization.
      end += entry.quota;
      p.physical_ids[e * p.ep_size + instance] = entry.target * physical_per_rank + masters + slot;
      p.quota_end[e * p.ep_size + instance++] = end;
    }
    p.instance_count[e] = instance;
    for (uint32_t j = instance; j < p.ep_size; ++j) {
      p.physical_ids[e * p.ep_size + j] = -1;
      p.quota_end[e * p.ep_size + j] = meta.global_counts[e];
    }
  }
  __syncthreads();

  for (int32_t i = threadIdx.x; i < best.count; i += blockDim.x) {
    const auto entry = best.exports[i];
    int32_t slot = 0;
    for (int32_t j = 0; j < best.count; ++j) {
      slot += best.exports[j].target == entry.target && best.exports[j].expert < entry.expert;
    }
    p.mapping[entry.target * p.slots + slot] = entry.expert;
  }
}

}  // namespace online_eplb_plan

template <uint32_t BlockSize>
__global__ void online_eplb_plan_kernel(const OnlineEplbPlanParams p) {
  using namespace online_eplb_plan;
  constexpr uint32_t kWarps = BlockSize / 32;
  constexpr uint32_t kMaxRounds = (kMaxEvaluations + kWarps - 1) / kWarps;
  __shared__ PlanMeta meta;
  __shared__ WarpWorkspace warps[kWarps];
  __shared__ SearchState search;
  __shared__ int64_t thresholds[kWarps];

  // 1. Prologue only prepares metadata; the kernel owns search orchestration.
  prologue(p, meta);
  if (threadIdx.x == 0) init_search(p, meta, search);
  __syncthreads();

  // 2. CTA-wide rounds: generate -> one probe per warp -> score -> next batch.
  // Round 0 includes the near-ideal probe; every round uses the same scoring.
  for (uint32_t round = 0; round < kMaxRounds && !search.done; ++round) {
    if (threadIdx.x == 0) generate_threshold_batch<BlockSize>(search, thresholds, round == 0);
    __syncthreads();
    plan_verify_and_score(p, meta, warps, thresholds, search);
  }

  // 3. Epilogue consumes the best candidate across completed rounds.
  epilogue(p, meta, search.best);
}

template <uint32_t BlockSize>
struct OnlineEplbPlanKernel {
  static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128);

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

    // All three phases run in one CTA and use only shared-memory scratch.
    LaunchKernel(1, BlockSize, device.unwrap())(online_eplb_plan_kernel<BlockSize>, params);
  }
};

}  // namespace sglang
