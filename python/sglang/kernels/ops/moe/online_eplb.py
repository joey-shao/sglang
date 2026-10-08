"""Device-only online EPLB operators; workspace ownership belongs to the caller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

ONLINE_EPLB_ROUTE_THREADS_PER_BLOCK = 256
ONLINE_EPLB_PLAN_THREADS_PER_BLOCK = 128

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.srt.eplb.online_balancer import BalancePlan


@cache_once
def _jit_online_eplb_module(
    dtype: torch.dtype, plan_block_size: int = ONLINE_EPLB_PLAN_THREADS_PER_BLOCK
) -> Module:
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Online EPLB IDs must be int32 or int64, got {dtype}")
    if plan_block_size not in (32, 64, 128):
        raise ValueError("Online EPLB plan block size must be 32, 64 or 128")
    args = make_cpp_args(dtype)
    planner = f"OnlineEplbPlanKernel<{plan_block_size}>"
    # No PDL: the planner must finish before downstream consumers read its output.
    return load_jit(
        "moe_online_eplb",
        *args,
        str(ONLINE_EPLB_ROUTE_THREADS_PER_BLOCK),
        str(plan_block_size),
        cuda_files=["moe/online_eplb_route.cuh", "moe/online_eplb_plan.cuh"],
        extra_cuda_cflags=[
            "-DONLINE_EPLB_ROUTE_THREADS_PER_BLOCK="
            f"{ONLINE_EPLB_ROUTE_THREADS_PER_BLOCK}"
        ],
        cuda_wrappers=[
            ("histogram", f"OnlineEplbRouteKernel<{args}>::histogram"),
            ("prefix", f"OnlineEplbRouteKernel<{args}>::prefix"),
            ("remap", f"OnlineEplbRouteKernel<{args}>::remap"),
            ("plan", f"{planner}::run"),
        ],
    )


def prepare_counts(
    ids: torch.Tensor,
    block_counts: torch.Tensor,
    block_prefix: torch.Tensor,
    local_counts: torch.Tensor,
) -> None:
    """Write counts and exclusive tile prefixes for contiguous flattened IDs.

    Tile buffers have shape
    [max(1, ceil(ids.numel() / ONLINE_EPLB_ROUTE_THREADS_PER_BLOCK)), experts].
    Invalid IDs contribute nothing. Retain block_prefix until remap completes.
    """
    module = _jit_online_eplb_module(ids.dtype)
    module.histogram(ids.view(-1), block_counts, local_counts.numel())
    module.prefix(block_counts, block_prefix, local_counts, local_counts.numel())


def plan(
    counts: torch.Tensor,
    rank: int,
    min_quota: int,
    plan: BalancePlan,
    block_size: int = ONLINE_EPLB_PLAN_THREADS_PER_BLOCK,
) -> None:
    """Fill the plan's preallocated tensors without reading device results back."""
    _jit_online_eplb_module(torch.int32, block_size).plan(
        counts,
        plan.redundancy_mapping,
        plan.instance_physical_ids,
        plan.instance_quota_end,
        plan.instance_count,
        plan.source_prefix,
        rank,
        min_quota,
        plan.redundancy_mapping.shape[1],
    )


def remap(
    ids: torch.Tensor,
    block_prefix: torch.Tensor,
    source_prefix: torch.Tensor,
    instance_physical_ids: torch.Tensor,
    instance_quota_end: torch.Tensor,
    instance_count: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Remap int32/int64 IDs into preallocated int64 output of the same shape.

    Reuse the prefixes computed for these IDs to preserve stable global ordinals.
    The output dtype matches DeepEP's dispatch contract.
    """
    _jit_online_eplb_module(ids.dtype).remap(
        ids.view(-1),
        out.view(-1),
        block_prefix,
        source_prefix,
        instance_quota_end,
        instance_physical_ids,
        instance_count,
        instance_count.numel(),
        instance_physical_ids.shape[1],
    )
    return out
