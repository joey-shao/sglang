# SPDX-License-Identifier: Apache-2.0
"""Fixed Ulysses SP x TP topology; independent of model execution and CUDA.

Physical ranks use [SP, base TP] order. Attention heads use [base TP, SP]
order, so a full-TP weight shard is not necessarily its collective rank.
All process groups retain ascending physical order. Callers must explicitly
restore logical shard order after a full-TP all-gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


@dataclass(frozen=True)
class UlyssesRankLayout:
    tp_size: int
    sp_size: int

    def __post_init__(self):
        if self.tp_size < 1 or self.sp_size < 1:
            raise ValueError("Ulysses TP and SP sizes must be positive")
        if self.tp_size % self.sp_size:
            raise ValueError("Ulysses SP size must divide the full TP size")

    @property
    def base_tp_size(self) -> int:
        return self.tp_size // self.sp_size

    @property
    def base_tp_ranks(self) -> list[list[int]]:
        t = self.base_tp_size
        return [list(range(s * t, (s + 1) * t)) for s in range(self.sp_size)]

    @property
    def sp_ranks(self) -> list[list[int]]:
        t = self.base_tp_size
        return [list(range(i, self.tp_size, t)) for i in range(t)]

    def coordinates(self, physical_rank: int) -> tuple[int, int]:
        if not 0 <= physical_rank < self.tp_size:
            raise ValueError(f"Rank {physical_rank} is outside the full TP group")
        return divmod(physical_rank, self.base_tp_size)

    def full_tp_shard_rank(self, physical_rank: int) -> int:
        sp_rank, base_tp_rank = self.coordinates(physical_rank)
        return base_tp_rank * self.sp_size + sp_rank

    @property
    def physical_to_shard(self) -> tuple[int, ...]:
        return tuple(self.full_tp_shard_rank(r) for r in range(self.tp_size))

    @property
    def shard_to_physical(self) -> tuple[int, ...]:
        """Indices to reorder gathered physical-rank chunks into shard order."""
        mapping = self.physical_to_shard
        return tuple(sorted(range(self.tp_size), key=mapping.__getitem__))

    def validate_heads(self, num_heads: int, num_kv_heads: int) -> None:
        """Initial implementation requires GQA without KV-head replication."""
        if (
            num_heads < 1
            or num_kv_heads < 1
            or num_heads % num_kv_heads
            or num_heads % self.tp_size
            or num_kv_heads % self.tp_size
        ):
            raise ValueError(
                "Ulysses requires positive Q/KV heads divisible by full TP size "
                "and Q heads divisible by KV heads; KV replication is not supported"
            )


def create_ulysses_groups(
    layout: UlyssesRankLayout,
    full_tp_group: GroupCoordinator,
    group_factory: Callable[..., GroupCoordinator],
    *,
    local_rank: int,
    backend: str,
) -> tuple[GroupCoordinator, GroupCoordinator]:
    """Create subgroups in identical order on every rank; borrow full TP."""
    if full_tp_group.ranks != list(range(layout.tp_size)):
        raise ValueError("Ulysses requires full TP ranks [0, ..., P-1]")
    kwargs = dict(
        local_rank=local_rank,
        backend=backend,
        use_custom_allreduce=False,
        use_mscclpp_allreduce=False,
        use_torch_symm_mem_allreduce=False,
    )
    base_tp = group_factory(
        layout.base_tp_ranks, group_name="ulysses_base_tp", **kwargs
    )
    try:
        sp = group_factory(layout.sp_ranks, group_name="ulysses_sp", **kwargs)
    except Exception:
        base_tp.destroy()
        raise
    return base_tp, sp
