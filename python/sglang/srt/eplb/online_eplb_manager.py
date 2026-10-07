"""Global online EPLB lifecycle, from post-gating balance to MoE completion."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.eplb.online_balancer import BalancePlan, OnlineExpertBalancer
from sglang.srt.eplb.online_eplb_weight_prefetcher import (
    OnlineEplbWeightPrefetcher,
    PrefetchTicket,
    get_online_eplb_weight_prefetcher,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class OnlineEplbManager:
    def __init__(self, model):
        self.prefetcher: Optional[OnlineEplbWeightPrefetcher] = None
        self._prefetch_ticket: Optional[PrefetchTicket] = None
        self.active = False
        self.ep_group = None

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.runtime_context import get_exec, get_parallel

        cfg = get_exec().moe
        self.min_forward_tokens = cfg.online_ep_min_forward_tokens
        layers = sorted(
            (
                m
                for m in model.modules()
                if isinstance(m, FusedMoE) and not m.is_shared_fused_moe
            ),
            key=lambda m: m.layer_id,
        )
        if not layers:
            raise ValueError("Online EPLB requires routed FusedMoE layers")
        # DeepEP v2.5 uses the runtime TP communication group (MoE TP=1).
        self.ep_group = get_parallel().tp_group.device_group
        self.balancer = OnlineExpertBalancer(
            dist.get_world_size(self.ep_group),
            dist.get_rank(self.ep_group),
            layers[0].num_global_routed_experts,
            cfg.online_ep_redundant_slots_per_rank,
            cfg.online_ep_min_tokens_per_replica,
        )
        self.top_k = layers[0].top_k
        try:
            self.prefetcher = get_online_eplb_weight_prefetcher(
                group=self.ep_group,
                hidden_size=layers[0].hidden_size,
                router_topk=self.top_k,
            )

            config = layers[0].moe_runner_config
            if (
                not isinstance(layers[0].quant_method, Fp8MoEMethod)
                or not layers[0].runner.runner_backend.is_deep_gemm()
                or layers[0].runner.fused_func is not None
                or layers[0].params_dtype != torch.bfloat16
                or layers[0].moe_tp_size != 1
                or layers[0].num_fused_shared_experts
                or layers[0].with_bias
                or config.activation != "silu"
                or not config.is_gated
                or config.apply_router_weight_on_input
                or config.gemm1_alpha is not None
                or config.gemm1_beta is not None
                or config.gemm1_clamp_limit is not None
                or config.swiglu_limit is not None
                or config.silu_mul_keep_fp32
                or layers[0].supports_deferred_finalize
            ):
                raise ValueError(
                    "Online EPLB requires DeepGEMM Fp8MoEMethod block-FP8 gated SiLU experts without bias/clamp, MoE TP=1 and separate shared experts"
                )

            weights = layers[0].get_online_expert_weights()
            # All layers have been checked before any collective allocation.
            self.prefetcher.setup(weights, slots_per_rank=self.balancer.slots)
        except BaseException:
            self.cleanup()
            raise

    @contextmanager
    def forward_scope(self, forward_batch):
        # DP-attention publishes the same phase and unpadded counts on every
        # EP rank, including idle ranks. Without DP all ranks see one batch.
        counts = forward_batch.original_global_num_tokens_cpu
        if counts is None:
            counts = forward_batch.global_num_tokens_cpu
        is_prefill = (
            forward_batch.is_extend_in_batch
            if counts is not None
            else forward_batch.forward_mode.is_extend()
        )
        num_tokens = (
            sum(counts) if counts is not None else forward_batch.input_ids.numel()
        )
        self.active = (
            is_prefill
            and num_tokens >= self.min_forward_tokens
            and not torch.cuda.is_current_stream_capturing()
        )
        try:
            yield
        finally:
            self.active = False

    def balance(self, layer: FusedMoE, logical_topk_ids: torch.Tensor) -> BalancePlan:
        if not self.active:
            raise RuntimeError("Online EPLB balance requires an active forward")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Online EPLB CPU reference planning cannot be captured")
        if logical_topk_ids.ndim != 2 or logical_topk_ids.shape[1] != self.top_k:
            raise ValueError("Online EPLB requires explicit [tokens, routed top-k] IDs")
        ids = logical_topk_ids.reshape(-1).to(torch.int64)
        e = self.balancer.num_experts
        valid = (ids >= 0) & (ids < e)
        local_counts = torch.zeros(e, dtype=torch.int64, device=ids.device)
        local_counts.scatter_add_(0, ids.clamp(0, e - 1), valid.to(torch.int64))
        counts = torch.empty(
            (self.balancer.world_size, e), dtype=torch.int64, device=ids.device
        )
        dist.all_gather_into_tensor(counts.view(-1), local_counts, group=self.ep_group)
        plan = self.balancer.plan(counts)
        self._prefetch_ticket = self.prefetcher.prefetch_weight_async(
            layer_weights=layer.get_online_expert_weights(),
            redundancy_mapping=plan.redundancy_mapping,
        )
        return plan

    def wait_prefetch(self, layer: FusedMoE) -> None:
        self.prefetcher.wait_prefetch(
            self._prefetch_ticket, consumer_stream=torch.cuda.current_stream()
        )
        # Copy on the consumer stream after the communication event. Dispatch
        # and the ordinary GEMMs then observe the complete M+R weight slab.
        m = self.balancer.num_masters
        for target, source in zip(
            layer._online_expert_weights.tensors(),
            self._prefetch_ticket.replica_weights.tensors(),
        ):
            target[m:].copy_(source, non_blocking=True)

    def cleanup(self) -> None:
        if self.prefetcher is not None:
            self.prefetcher.cleanup()
            self.prefetcher = None
        self._prefetch_ticket = None
        self.active = False
        self.ep_group = None
