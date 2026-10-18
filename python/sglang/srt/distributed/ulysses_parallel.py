# SPDX-License-Identifier: Apache-2.0
"""Fixed Ulysses SP x TP topology; independent of model execution and CUDA.

Physical ranks and attention heads both use [base TP, SP] order. SP groups
contain consecutive ranks; base-TP groups connect the same SP rank across
those groups. After SP all-to-all, each worker owns the same heads as ordinary
full TP. All weights and collective groups use their normal rank order.
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

    @property
    def base_tp_size(self) -> int:
        return self.tp_size // self.sp_size

    @property
    def base_tp_ranks(self) -> list[list[int]]:
        return [list(range(s, self.tp_size, self.sp_size)) for s in range(self.sp_size)]

    @property
    def sp_ranks(self) -> list[list[int]]:
        s = self.sp_size
        return [list(range(t * s, (t + 1) * s)) for t in range(self.base_tp_size)]


def create_ulysses_groups(
    layout: UlyssesRankLayout,
    group_factory: Callable[..., GroupCoordinator],
    *,
    local_rank: int,
    backend: str,
) -> tuple[GroupCoordinator, GroupCoordinator]:
    """Create attention-TP and SP subgroups in identical order on every rank."""
    kwargs = dict(
        local_rank=local_rank,
        backend=backend,
        use_custom_allreduce=False,
        use_mscclpp=False,
        use_torch_symm_mem_allreduce=False,
    )
    base_tp = group_factory(
        layout.base_tp_ranks, group_name="ulysses_attn_tp", **kwargs
    )
    sp = group_factory(layout.sp_ranks, group_name="ulysses_sp", **kwargs)

    return base_tp, sp
