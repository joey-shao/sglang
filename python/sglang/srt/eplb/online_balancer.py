"""Eager reference planner for per-layer online expert replication.

The initial policy is deliberately simple: move integer quotas from a hot
master to an idle peer slot. Planning synchronizes counts to the CPU; histogram
and exact ordinal-based rerouting stay on the device. This is not graph-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class BalancePlan:
    redundancy_mapping: torch.Tensor
    source_prefix: torch.Tensor
    # (logical expert, target physical id, start ordinal, end ordinal).
    transfers: list[tuple[int, int, int, int]]


class OnlineExpertBalancer:
    def __init__(
        self,
        world_size: int,
        rank: int,
        num_experts: int,
        slots_per_rank: int,
        min_tokens: int,
    ):
        self.world_size = world_size
        self.rank = rank
        if num_experts % self.world_size or num_experts <= 0:
            raise ValueError("Online EP requires equal, nonempty master expert shards")
        if slots_per_rank <= 0 or min_tokens <= 0:
            raise ValueError(
                "Online EP replica capacity and minimum quota must be positive"
            )
        self.num_experts = num_experts
        self.num_masters = num_experts // self.world_size
        self.slots = slots_per_rank
        self.local_physical = self.num_masters + slots_per_rank
        self.min_tokens = min_tokens

    def master_ids(self, logical_ids: torch.Tensor) -> torch.Tensor:
        valid = (logical_ids >= 0) & (logical_ids < self.num_experts)
        physical = logical_ids.div(
            self.num_masters, rounding_mode="floor"
        ) * self.local_physical + logical_ids.remainder(self.num_masters)
        return torch.where(valid, physical, -1)

    def plan(self, counts: torch.Tensor) -> BalancePlan:
        """Consume EP-wide counts; communication is owned by the manager."""
        if counts.shape != (self.world_size, self.num_experts):
            raise ValueError("Online EPLB counts must have shape [EP size, experts]")
        prefix = counts[: self.rank].sum(dim=0)
        global_counts = counts.sum(dim=0).cpu().tolist()
        return self._greedy_plan(global_counts, prefix)

    def _greedy_plan(self, counts: list[int], prefix: torch.Tensor) -> BalancePlan:
        m, p, r = self.num_masters, self.world_size, self.slots
        remaining = list(counts)
        loads = [sum(counts[rank * m : (rank + 1) * m]) for rank in range(p)]
        before = tuple(loads)
        mapping = [[-1] * r for _ in range(p)]
        used = [0] * p
        transfers = []
        # All ranks see identical counts; stable iteration/tie breaking gives
        # identical mappings without a separate plan broadcast.
        for _ in range(p * r):
            best = None
            best_gain = 0
            for source in range(p):
                for target in range(p):
                    if source == target or used[target] == r:
                        continue
                    gap = loads[source] - loads[target]
                    if gap < 2 * self.min_tokens:
                        continue
                    for expert in range(source * m, (source + 1) * m):
                        if expert in mapping[target]:
                            continue
                        quota = min(gap // 2, remaining[expert])
                        if quota < self.min_tokens:
                            continue
                        # Reduction in squared rank load. Unlike a strict max
                        # test this can reduce several equally hot ranks.
                        gain = 2 * quota * (gap - quota)
                        if gain > best_gain:
                            best_gain = gain
                            best = (source, target, expert, quota)
            if best is None:
                break
            source, target, expert, quota = best
            slot = used[target]
            used[target] += 1
            mapping[target][
                slot
            ] = expert  # Single domain: logical == prefetch source ID.
            end = remaining[expert]
            remaining[expert] -= quota
            transfers.append(
                (
                    expert,
                    target * self.local_physical + m + slot,
                    remaining[expert],
                    end,
                )
            )
            loads[source] -= quota
            loads[target] += quota
        if self.rank == 0:
            logger.info("Online EPLB rank loads: %s -> %s", before, tuple(loads))
        return BalancePlan(
            redundancy_mapping=torch.tensor(
                mapping, dtype=torch.int32, device=prefix.device
            ),
            source_prefix=prefix,
            transfers=transfers,
        )
