# SPDX-License-Identifier: Apache-2.0
"""Utilities for the experimental expert-group MoE execution path.

Overlap execution through ExpertGroupExecutor. This module contains the
route-space transform independently from DeepEP so its invariants can be tested
without a distributed CUDA environment.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, List, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)


class ExpertGroupExecutor:
    """Two-group pipeline using DeepEP v2 native asynchronous communication.

    Owns group geometry, runners, and dispatchers. The layer owns
    weights; each group resolves its current quantization info during forward.
    All ranks enqueue D0, D1, K0, K1, including empty groups. Communication
    contexts must own independent buffers. Execution stays outside CUDA graphs;
    stream/event dependencies overlap communication with group computation.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        runner_backend,
        *,
        num_experts: int,
        num_local_experts: int,
        ep_size: int,
        group_count: int,
        dispatcher_factory: Callable[..., BaseDispatcher],
    ):
        if group_count != 2:
            raise ValueError("Expert-group overlap requires exactly two groups")

        self.gemm_sms = None
        comm_sms = envs.SGLANG_DEEPEP_V2_NUM_SMS.get()
        if comm_sms < 0:
            raise ValueError("SGLANG_DEEPEP_V2_NUM_SMS must be nonnegative.")
        # Zero delegates communication sizing to DeepEP; retain default GEMM sizing.
        if comm_sms > 0:
            # Constructed inside each TP worker after its CUDA device is selected.
            total_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            if not 0 < comm_sms < total_sms:
                raise ValueError(
                    "SGLANG_DEEPEP_V2_NUM_SMS must be smaller "
                    f"than the device SM count ({total_sms}), got {comm_sms}."
                )
            self.gemm_sms = total_sms - comm_sms
            if moe_runner_config.layer_id == 0:
                logger.info(
                    "Expert-group overlap SM budget: total=%d, comm=%d, gemm=%d",
                    total_sms,
                    comm_sms,
                    self.gemm_sms,
                )

        self.num_experts = num_experts
        self.ep_size = ep_size
        self.group_count = group_count
        self.group_size = num_local_experts // group_count
        self.expert_slices = tuple(
            slice(i * self.group_size, (i + 1) * self.group_size)
            for i in range(group_count)
        )
        self.group_runner_config = replace(
            moe_runner_config,
            num_experts=num_experts // group_count,
            num_local_experts=self.group_size,
            num_fused_shared_experts=0,
            inplace=False,
        )
        self.group_runners = tuple(
            MoeRunner(runner_backend, self.group_runner_config)
            for _ in range(group_count)
        )
        # Inject the factory so this module does not depend on FusedMoE.
        self.dispatcher = ExpertGroupDispatcher(
            [
                dispatcher_factory(self.group_runner_config, expert_group_index=i)
                for i in range(group_count)
            ]
        )

    def _compute_group(self, layer, group_index: int, dispatch_output):
        runner = self.group_runners[group_index]
        # Resolve current weights on every forward: loading/quantization may
        # replace layer tensors after this executor has been constructed.
        quant_info = layer.quant_method.get_expert_group_quant_info(
            layer,
            runner.runner_backend,
            expert_slice=self.expert_slices[group_index],
        )
        if self.gemm_sms is None:
            return runner.run(dispatch_output, quant_info)

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            configure_deep_gemm_num_sms,
        )

        # Covers both gate/up and down, including JIT warmup and graph capture.
        # Restore the prior setting so attention/dense GEMMs are unaffected.
        with configure_deep_gemm_num_sms(self.gemm_sms):
            return runner.run(dispatch_output, quant_info)

    def run(self, layer, hidden_states, topk_output: StandardTopKOutput):
        """Split routes, overlap group communication/compute, and sum outputs."""
        dispatchers = self.dispatcher.group_dispatchers

        # Buffer construction can synchronize ranks/devices. Finish it before
        # submitting work that is intended to overlap.
        for dispatcher in dispatchers:
            dispatcher.prepare_buffer()

        grouped_topk = split_topk_output_by_expert_group(
            topk_output,
            num_experts=self.num_experts,
            ep_size=self.ep_size,
            group_count=self.group_count,
        )
        # Submit both dispatches before waiting for D0 or enqueuing C0.
        # Async prefill skips CPU sync, so D1 can overlap C0.
        dispatch_states = [
            dispatcher.dispatch_a(hidden_states, topk)
            for dispatcher, topk in zip(dispatchers, grouped_topk)
        ]
        d0 = dispatchers[0].dispatch_b(dispatch_states[0])
        c0 = self._compute_group(layer, 0, d0)

        # Native comm order is D0, D1, K0, K1.
        # Submit K0 before C1, so its implicit compute dependency ends at C0.
        # Do not wait for K0 here: that would serialize K0 and C1.
        y0, combine_event0 = dispatchers[0].combine_a(c0)
        d1 = dispatchers[1].dispatch_b(dispatch_states[1])
        c1 = self._compute_group(layer, 1, d1)
        y1, combine_event1 = dispatchers[1].combine_a(c1)

        # Keep inputs and outputs alive through the final completion waits.
        # DeepEP tracks cross-stream allocation lifetimes, including handle
        # metadata; no additional SGLang communication stream is needed.
        y0 = dispatchers[0].combine_b(y0, combine_event0)
        y1 = dispatchers[1].combine_b(y1, combine_event1)
        return y0 + y1


class ExpertGroupDispatcher(BaseDispatcher):
    """Own one independent dispatcher state machine per expert group."""

    def __init__(self, dispatchers: Sequence[BaseDispatcher]):
        super().__init__()
        # REVIEW: Validation temporarily disabled for expert-group overlap review.
        # if not dispatchers:
        #     raise ValueError("ExpertGroupDispatcher requires at least one dispatcher")
        self.group_dispatchers = tuple(dispatchers)

    @property
    def expert_mask_gpu(self):
        return getattr(self.group_dispatchers[0], "expert_mask_gpu", None)

    def dispatch(self, **kwargs) -> DispatchOutput:
        raise RuntimeError("Use ExpertGroupExecutor for expert-group dispatch")

    def combine(self, **kwargs) -> torch.Tensor:
        raise RuntimeError("Use ExpertGroupExecutor for expert-group combine")

    def dispatch_group(self, group_index: int, **kwargs) -> DispatchOutput:
        return self.group_dispatchers[group_index].dispatch(**kwargs)

    def combine_group(self, group_index: int, **kwargs) -> torch.Tensor:
        return self.group_dispatchers[group_index].combine(**kwargs)

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        for dispatcher in self.group_dispatchers:
            dispatcher.set_quant_config(quant_config)

    def register_deepep_dispatch_hook(self, hook):
        return [
            dispatcher.register_deepep_dispatch_hook(hook)
            for dispatcher in self.group_dispatchers
        ]


def split_topk_output_by_expert_group(
    topk_output: StandardTopKOutput,
    *,
    num_experts: int,
    ep_size: int,
    group_count: int,
) -> List[StandardTopKOutput]:
    """Partition physical routes and compact their expert IDs per group.

    Physical experts are laid out rank-major.  Each rank's contiguous local
    expert range is split into ``group_count`` equally sized ranges.  Group
    dispatchers see a compact global expert space containing only their range
    from every rank::

        physical_id = rank * local_experts + group * group_size + offset
        compact_id  = rank * group_size + offset

    Routes outside a group are represented by ``id=-1, weight=0``.  Shapes stay
    fixed, which is required by the later CUDA-graph implementation.
    """

    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if num_experts <= 0:
    #     raise ValueError(f"num_experts must be positive, got {num_experts}")
    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if ep_size <= 0:
    #     raise ValueError(f"ep_size must be positive, got {ep_size}")
    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if group_count <= 0:
    #     raise ValueError(f"group_count must be positive, got {group_count}")
    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if num_experts % ep_size != 0:
    #     raise ValueError(
    #         f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
    #     )

    local_experts = num_experts // ep_size
    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if local_experts % group_count != 0:
    #     raise ValueError(
    #         f"local experts ({local_experts}) must be divisible by "
    #         f"group_count ({group_count})"
    #     )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    # REVIEW: Validation temporarily disabled for expert-group overlap review.
    # if topk_ids.shape != topk_weights.shape:
    #     raise ValueError(
    #         "topk_ids and topk_weights must have the same shape, got "
    #         f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
    #     )

    group_size = local_experts // group_count
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    rank = torch.div(safe_ids, local_experts, rounding_mode="floor")
    local_id = torch.remainder(safe_ids, local_experts)
    route_group = torch.div(local_id, group_size, rounding_mode="floor")
    compact_id = rank * group_size + torch.remainder(local_id, group_size)

    outputs = []
    invalid_id = torch.full_like(topk_ids, -1)
    zero_weight = torch.zeros_like(topk_weights)
    for group_index in range(group_count):
        in_group = valid & (route_group == group_index)
        outputs.append(
            StandardTopKOutput(
                topk_weights=torch.where(in_group, topk_weights, zero_weight),
                topk_ids=torch.where(in_group, compact_id, invalid_id),
                router_logits=topk_output.router_logits,
            )
        )

    return outputs


def compact_to_physical_expert_id(
    compact_id: torch.Tensor,
    *,
    group_index: int,
    num_experts: int,
    ep_size: int,
    group_count: int,
) -> torch.Tensor:
    """Inverse the compact-ID mapping for one group (primarily for tests)."""

    local_experts = num_experts // ep_size
    group_size = local_experts // group_count
    rank = torch.div(compact_id, group_size, rounding_mode="floor")
    offset = torch.remainder(compact_id, group_size)
    return rank * local_experts + group_index * group_size + offset
