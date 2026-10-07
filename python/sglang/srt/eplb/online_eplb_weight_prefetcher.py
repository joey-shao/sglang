"""Backend-independent weight storage and prefetch contract for online EPLB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ExpertWeightBundle:
    """128x128 block-FP8 expert weights and their two scale tensors."""

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor

    @classmethod
    def from_block_fp8(cls, w13, w2, w13_scale, w2_scale):
        """Describe final post-load 128x128 FP8 weights without converting them."""
        scales = (w13_scale, w2_scale)
        formats = tuple(
            "ue8m0" if getattr(scale, "format_ue8m0", False) else "fp32"
            for scale in scales
        )
        if formats[0] != formats[1]:
            raise ValueError("Online EPLB expert scale formats must match")
        for weight, scale in zip((w13, w2), scales):
            if weight.ndim != 3:
                raise ValueError("Online EPLB requires expert-major FP8 weights")
            e, n, k = weight.shape
            if weight.dtype != torch.float8_e4m3fn or n % 128 or k % 128:
                raise ValueError("Online EPLB requires aligned E4M3 block-FP8 experts")
            if formats[0] == "fp32":
                valid = (
                    scale.dtype == torch.float32
                    and tuple(scale.shape) == (e, n // 128, k // 128)
                    and scale.is_contiguous()
                )
            else:
                valid = (
                    scale.dtype == torch.int32
                    and tuple(scale.shape) == (e, n, (k // 128 + 3) // 4)
                    and scale.stride(1) == 1
                    and scale.stride(2) >= n
                    and scale.stride(2) * scale.element_size() % 16 == 0
                )
            if not valid:
                raise ValueError(
                    "Online EPLB does not support this post-load FP8 scale layout"
                )
        return cls(w13, w2, w13_scale, w2_scale)

    def tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.w13, self.w2, self.w13_scale, self.w2_scale)


@dataclass(frozen=True)
class PrefetchTicket:
    event: Any
    # Keep all asynchronously read inputs alive through prefetch completion.
    source_weights: ExpertWeightBundle
    redundancy_mapping: torch.Tensor
    replica_weights: ExpertWeightBundle
    # Staged aligned payloads must survive asynchronous communication.
    source_payloads: tuple[torch.Tensor, ...] = ()


class OnlineEplbWeightPrefetcher(ABC):
    @abstractmethod
    def setup(
        self, layer_weights: ExpertWeightBundle, *, slots_per_rank: int
    ) -> None: ...

    @abstractmethod
    def prefetch_weight_async(
        self,
        *,
        layer_weights: ExpertWeightBundle,
        redundancy_mapping: torch.Tensor,
    ) -> PrefetchTicket: ...

    @abstractmethod
    def wait_prefetch(self, ticket: PrefetchTicket, *, consumer_stream) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...


def get_online_eplb_weight_prefetcher(
    *, group, hidden_size: int, router_topk: int
) -> OnlineEplbWeightPrefetcher:
    """Select weight transport independently of the token dispatcher.

    Backend buffer getters share runtime resources and reject inconsistent
    communication settings when either transport path acquires the buffer.
    """
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec

    backend = get_exec().moe.moe_a2a_backend
    if backend != "deepep_v2.5":
        raise NotImplementedError(
            f"Online EPLB weight prefetch is not supported by {backend}"
        )
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2_5 import DeepEPv25Buffer

    return DeepEPv25Buffer.get_prefetcher(
        group,
        hidden_size,
        router_topk,
        envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        use_fp8_dispatch=True,
    )
