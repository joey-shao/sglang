# SPDX-License-Identifier: Apache-2.0
"""Fixed Ulysses SP x TP topology; independent of model execution and CUDA.

Physical ranks use [SP, base TP] order. Attention heads use [base TP, SP]
order, so a full-TP weight shard is not necessarily its collective rank.
All process groups retain ascending physical order. Callers must explicitly
restore logical shard order after a full-TP all-gather.
"""

from __future__ import annotations

from contextlib import contextmanager
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


@dataclass(frozen=True)
class UlyssesTopology:
    layout: UlyssesRankLayout
    full_tp_group: GroupCoordinator
    base_tp_group: GroupCoordinator
    sp_group: GroupCoordinator

    @property
    def full_tp_shard_rank(self) -> int:
        return self.layout.full_tp_shard_rank(self.full_tp_group.rank_in_group)

    @property
    def base_tp_shard_rank(self) -> int:
        return self.layout.coordinates(self.full_tp_group.rank_in_group)[1]

    @property
    def sp_rank(self) -> int:
        return self.layout.coordinates(self.full_tp_group.rank_in_group)[0]

    @contextmanager
    def base_tp_scope(self):
        """Scope base-TP model construction and collectives together.

        Only use at a serialized model execution boundary. This is not an
        SP forward implementation: attention retains full-TP head ownership,
        and token redistribution must be supplied by the model adapter.
        Scheduler communication uses ``full_tp_group`` outside this scope.
        """
        from sglang.srt.distributed.parallel_state import patch_tensor_parallel_group

        with patch_tensor_parallel_group(self.base_tp_group):
            yield

    @contextmanager
    def model_tp_scope(self):
        """Construct or execute the model under base TP; restore state on failure."""
        from sglang.srt.distributed import parallel_state as ps
        from sglang.srt.runtime_context import get_parallel

        group = self.base_tp_group
        old_attention_group = ps._ATTN_TP
        with self.base_tp_scope():
            try:
                ps._ATTN_TP = group
                with get_parallel().override(
                    attn_tp_group=group,
                    attn_tp_size=group.world_size,
                    attn_tp_rank=self.base_tp_shard_rank,
                ):
                    yield
            finally:
                ps._ATTN_TP = old_attention_group


def create_ulysses_topology(
    layout: UlyssesRankLayout,
    full_tp_group: GroupCoordinator,
    group_factory: Callable[..., GroupCoordinator],
    *,
    local_rank: int,
    backend: str,
) -> UlyssesTopology:
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
    return UlyssesTopology(layout, full_tp_group, base_tp, sp)


def validate_ulysses_config(cfg) -> None:
    """Validate the fixed SP x TP topology before creating process groups."""
    UlyssesRankLayout(cfg.tp_size, cfg.ulysses_sp_size)
    if cfg.ulysses_sp_size == 1:
        return
    if cfg.attn_cp_size != 1:
        raise ValueError(
            "Ulysses parallel (--ulysses-sp-size > 1) cannot be combined with "
            "attention context parallel (--attn-cp-size != 1)"
        )
    if cfg.dcp_size != 1:
        raise ValueError(
            "Ulysses parallel (--ulysses-sp-size > 1) cannot be combined with "
            "decode context parallel (--dcp-size != 1)"
        )
    if cfg.enable_dp_attention:
        raise ValueError(
            "Ulysses parallel (--ulysses-sp-size > 1) cannot be combined with "
            "attention DP (--enable-dp-attention)"
        )
    for name in (
        "nnodes",
        "pp_size",
        "dp_size",
        "ep_size",
        "moe_dp_size",
    ):
        if getattr(cfg, name) != 1:
            raise ValueError(f"Ulysses topology currently requires {name}=1")
    if cfg.enable_pdmux or cfg.ep_join_mode:
        raise ValueError(
            "Ulysses topology does not support PD multiplexing or EP joining"
        )
