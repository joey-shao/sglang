"""Device-resident online EPLB planning and remapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.online_eplb import ONLINE_EPLB_BLOCK_SIZE
from sglang.kernels.ops.moe.online_eplb import plan as device_plan
from sglang.kernels.ops.moe.online_eplb import prepare_counts as device_prepare_counts
from sglang.kernels.ops.moe.online_eplb import remap as device_remap


@dataclass(frozen=True, slots=True, kw_only=True)
class BalancePlan:
    """Borrowed int32 CUDA tensors, valid until the next balance.

    P = EP ranks, E = logical experts, R = replica slots per rank.
    Each expert has one master and at most one replica per other rank,
    so its instance table has capacity P. All ranks share the same plan
    values except source_prefix, which depends on the local source rank.
    """

    # [P, R]: [target_rank, replica_slot] -> logical expert ID; -1 if unused.
    # Occupied slots on each target rank are ordered by logical expert ID.
    redundancy_mapping: torch.Tensor

    # [E, P]: [logical_expert, instance] -> global physical expert ID.
    # Instance 0 is the master; replicas follow in ascending target-rank order.
    # The instance axis is NOT a rank index. Unused entries are -1.
    instance_physical_ids: torch.Tensor

    # [E, P]: exclusive cumulative quota ends in the same instance order.
    # E.g. [6, 10] assigns ordinals [0, 6) to the master and [6, 10) to a replica.
    # The last valid end and all unused entries equal this expert's global count.
    instance_quota_end: torch.Tensor

    # [E]: number of valid instances per expert, including the master (1..P).
    # Even an expert with zero routes has one valid instance with quota zero.
    instance_count: torch.Tensor

    # [E]: sum of each expert's routing counts on source ranks before this rank.
    # Add to the local occurrence ordinal to obtain the global routing ordinal.
    source_prefix: torch.Tensor


class OnlineExpertBalancer:
    def __init__(
        self,
        world_size: int,
        rank: int,
        num_experts: int,
        slots_per_rank: int,
        min_tokens: int,
        max_routing_entries: int,
    ):
        if (
            world_size not in (2, 4, 8)
            or not 0 <= rank < world_size
            or not 0 < num_experts <= 512
            or num_experts % world_size
            or not 0 < slots_per_rank <= 8
            or not 0 < min_tokens <= torch.iinfo(torch.int32).max
        ):
            raise ValueError(
                "GPU online EPLB requires P in {2, 4, 8}, rank in [0, P), "
                "0 < E <= 512, E % P == 0, 0 < R <= 8 and int32 Qmin > 0"
            )

        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.num_masters = num_experts // world_size
        self.slots = slots_per_rank
        self.local_physical = self.num_masters + slots_per_rank
        self.min_tokens = min_tokens
        self.max_routing_entries = max_routing_entries

        blocks = (
            max_routing_entries + ONLINE_EPLB_BLOCK_SIZE - 1
        ) // ONLINE_EPLB_BLOCK_SIZE
        # The runner selects the current CUDA device; bare torch.empty still
        # defaults to CPU unless a separate default-device context is active.
        self.block_counts = torch.empty(
            (blocks, num_experts), dtype=torch.int32, device="cuda"
        )
        self.block_prefix = torch.empty_like(self.block_counts)
        self.local_counts = torch.empty(num_experts, dtype=torch.int32, device="cuda")
        self.counts = torch.empty(
            (world_size, num_experts), dtype=torch.int32, device="cuda"
        )
        self._plan = BalancePlan(
            redundancy_mapping=torch.empty(
                (world_size, slots_per_rank), dtype=torch.int32, device="cuda"
            ),
            instance_physical_ids=torch.empty(
                (num_experts, world_size), dtype=torch.int32, device="cuda"
            ),
            instance_quota_end=torch.empty(
                (num_experts, world_size), dtype=torch.int32, device="cuda"
            ),
            instance_count=torch.empty(num_experts, dtype=torch.int32, device="cuda"),
            source_prefix=torch.empty(num_experts, dtype=torch.int32, device="cuda"),
        )
        self._physical_ids = torch.empty(
            max_routing_entries, dtype=torch.int64, device="cuda"
        )
        self._active_prefix = None

    def master_ids(self, logical_ids: torch.Tensor) -> torch.Tensor:
        valid = (logical_ids >= 0) & (logical_ids < self.num_experts)
        physical = logical_ids.div(
            self.num_masters, rounding_mode="floor"
        ) * self.local_physical + logical_ids.remainder(self.num_masters)
        return torch.where(valid, physical, -1)

    def prepare_local_counts(self, logical_ids: torch.Tensor) -> None:
        blocks = max(
            1,
            (logical_ids.numel() + ONLINE_EPLB_BLOCK_SIZE - 1)
            // ONLINE_EPLB_BLOCK_SIZE,
        )
        self._active_prefix = self.block_prefix[:blocks]
        device_prepare_counts(
            logical_ids,
            self.block_counts[:blocks],
            self._active_prefix,
            self.local_counts,
        )

    def plan(self) -> BalancePlan:
        device_plan(self.counts, self.rank, self.min_tokens, self._plan)
        return self._plan

    def remap(self, logical_ids: torch.Tensor, plan: BalancePlan) -> torch.Tensor:
        """Remap the unchanged IDs used by the most recent prepare_local_counts."""
        if plan is not self._plan:
            raise RuntimeError("Online EPLB remap requires the current plan")
        out = self._physical_ids[: logical_ids.numel()].view_as(logical_ids)
        return device_remap(
            logical_ids,
            self._active_prefix,
            plan.source_prefix,
            plan.instance_physical_ids,
            plan.instance_quota_end,
            plan.instance_count,
            out,
        )
