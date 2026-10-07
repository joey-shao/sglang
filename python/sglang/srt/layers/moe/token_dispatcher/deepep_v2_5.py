"""DeepEP 2.5 EPBuffer adapter with the shared v2 dispatch/GEMM layout."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.srt.eplb.online_eplb_weight_layout import ExpertTensorLayout
from sglang.srt.eplb.online_eplb_weight_prefetcher import (
    ExpertWeightBundle,
    OnlineEplbWeightPrefetcher,
    PrefetchTicket,
)
from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
    DeepEPv2Buffer,
    DeepEPv2Dispatcher,
    _DeepEPv2Impl,
    _get_allow_hybrid_mode,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_global_online_eplb_manager,
)

_deepep_v2_5_import_error = None

try:
    from deep_ep import EPBuffer

    use_deepep_v2_5 = True
except (ImportError, OSError) as exc:
    use_deepep_v2_5 = False
    _deepep_v2_5_import_error = exc


class DeepEPv25Buffer(DeepEPv2Buffer, OnlineEplbWeightPrefetcher):
    """Own an EPBuffer separately from the legacy v2 ElasticBuffer."""

    _STATE_KEY = "deepep_v2_5_ep_state"

    @staticmethod
    def _ensure_available() -> None:
        if not use_deepep_v2_5:
            detail = (
                f" Original import error: {_deepep_v2_5_import_error}"
                if _deepep_v2_5_import_error is not None
                else ""
            )
            raise ImportError(
                "--moe-a2a-backend deepep_v2.5 requires DeepEP 2.5 (EPBuffer). "
                "Install it from https://github.com/deepseek-ai/DeepEP." + detail
            )

    @staticmethod
    def _buffer_type():
        return EPBuffer

    @classmethod
    def get_prefetcher(
        cls,
        group,
        hidden_size,
        router_topk,
        num_max_dispatch_tokens_per_rank,
        *,
        use_fp8_dispatch=False,
    ) -> OnlineEplbWeightPrefetcher:
        """Get the single runtime-owned prefetcher; setup allocates its LB region."""
        cls._ensure_available()
        state = cls._state()
        key = (
            group,
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            False,
            dist.get_world_size(group),
        )
        prefetcher = getattr(state, "prefetcher", None)
        if prefetcher is not None:
            if state.key != key:
                raise ValueError(
                    "Online EPLB prefetcher communication settings changed"
                )
            return prefetcher
        if state.buffer is not None:
            raise RuntimeError(
                "Online EPLB must reserve the LB region before initializing the EPBuffer"
            )
        prefetcher = cls(state)
        state.key = key
        state.prefetcher = prefetcher
        return prefetcher

    @classmethod
    def get_buffer(
        cls,
        group,
        hidden_size,
        router_topk,
        num_max_dispatch_tokens_per_rank,
        use_fp8_dispatch,
        allow_hybrid_mode=None,
    ):
        cls._ensure_available()
        state = cls._state()
        if allow_hybrid_mode is None:
            allow_hybrid_mode = _get_allow_hybrid_mode()
        if getattr(state, "prefetcher", None) is not None:
            key = (
                group,
                hidden_size,
                router_topk,
                num_max_dispatch_tokens_per_rank,
                use_fp8_dispatch,
                allow_hybrid_mode,
                dist.get_world_size(group),
            )
            if state.key != key:
                raise ValueError(
                    "Cannot rebuild an online EPLB EPBuffer with different settings; "
                    "clean up the manager before reinitializing"
                )
            if state.buffer is None:
                raise RuntimeError("Online EPLB prefetcher setup has not completed")
            return state.buffer
        try:
            online_enabled = get_exec().moe.enable_online_eplb
        except ValueError:
            online_enabled = False  # Standalone buffer users need no server config.
        if online_enabled:
            raise RuntimeError("Online EPLB requires prefetcher setup before dispatch")
        return super().get_buffer(
            group,
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
        )

    @classmethod
    def destroy(cls):
        if getattr(cls._state(), "prefetcher", None) is not None:
            raise RuntimeError(
                "Clean up the online EPLB manager before destroying its buffer"
            )
        super().destroy()

    def __init__(self, state):
        # Keep the owning runtime state, so cleanup cannot clear a different
        # runtime's buffer after a context switch.
        self._owner_state = state
        self.replica_weights = None
        self._replica_payloads = ()
        self._layouts = ()

    @property
    def buffer(self):
        state = self._owner_state
        return state.buffer if getattr(state, "prefetcher", None) is self else None

    def setup(self, layer_weights, *, slots_per_rank):
        self._ensure_available()
        from deep_ep import BufferAllocator, get_num_tma_alignment
        from deep_ep.utils.envs import get_physical_domain_size

        state = self._owner_state
        if getattr(state, "prefetcher", None) is not self:
            raise RuntimeError("Online EPLB prefetcher has been released")
        group, hidden_size, topk, max_tokens, use_fp8_dispatch, _, world_size = (
            state.key
        )
        if self.buffer is not None:
            return
        # The topology query already acquires and caches the NCCL communicator.
        # Configure reuse before it, just as in the ordinary buffer path.
        os.environ.setdefault("EP_REUSE_NCCL_COMM", "0")
        rdma_ranks, nvlink_ranks = get_physical_domain_size(group)
        if rdma_ranks != 1 or nvlink_ranks != world_size:
            raise ValueError(
                "Online EPLB requires a single NVLink domain covering the EP group"
            )
        tensors = layer_weights.tensors()
        layouts = tuple(
            ExpertTensorLayout.from_tensor(w, get_num_tma_alignment()) for w in tensors
        )
        allocation = BufferAllocator()
        replica_payloads = tuple(
            allocation.allocate((slots_per_rank, layout.pitch), layout.dtype)
            for layout in layouts
        )
        buffer = EPBuffer(
            group,
            num_max_tokens_per_rank=max_tokens,
            hidden=hidden_size,
            num_topk=topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=False,
            sl_idx=0,
            prefer_overlap_with_compute=False,
            lb_allocation_plan_or_num_bytes=allocation,
        )
        self._layouts = layouts
        self._replica_payloads = replica_payloads
        self.replica_weights = ExpertWeightBundle(
            *(
                layout.view(payload)
                for layout, payload in zip(layouts, replica_payloads)
            )
        )
        state.buffer = buffer

    def prefetch_weight_async(self, *, layer_weights, redundancy_mapping):
        # Layers execute serially. DeepEP waits for the current compute stream
        # before overwriting the shared replicas; combine already joins its
        # completion to that stream. No separate bank-release event is needed.
        source_payloads = tuple(
            layout.pack(w) for layout, w in zip(self._layouts, layer_weights.tensors())
        )
        event = self.buffer.lb_prefetch_weights(
            self._replica_payloads,
            source_payloads,
            redundancy_mapping,
            previous_event=None,
        )
        return PrefetchTicket(
            event,
            layer_weights,
            redundancy_mapping,
            self.replica_weights,
            source_payloads,
        )

    def wait_prefetch(self, ticket, *, consumer_stream):
        if self.buffer is None or ticket.replica_weights is not self.replica_weights:
            raise RuntimeError(
                "Online EPLB ticket belongs to a different replica buffer"
            )
        with torch.cuda.stream(consumer_stream):
            ticket.event.current_stream_wait()

    def cleanup(self):
        state = self._owner_state
        if getattr(state, "prefetcher", None) is not self:
            return
        if state.buffer is not None:
            # Drain communication as well as compute before destroying LB views.
            torch.cuda.synchronize()
        self.replica_weights = None
        self._replica_payloads = ()
        self._layouts = ()
        state.buffer = None
        state.key = None
        state.prefetcher = None


class _DeepEPv25Impl(_DeepEPv2Impl):
    buffer_class = DeepEPv25Buffer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = get_exec().moe
        self._online_eplb = cfg.enable_online_eplb
        self.num_master_experts = self.num_local_experts
        if self._online_eplb:
            slots = cfg.online_ep_redundant_slots_per_rank
            world_size = dist.get_world_size(self.group)
            if (
                slots <= 0
                or self.num_master_experts <= 0
                or self.num_experts != world_size * self.num_master_experts
            ):
                raise ValueError(
                    "Online EPLB requires positive replica capacity and equal master shards"
                )
            self.num_local_experts += slots
            self.num_experts = world_size * self.num_local_experts

    def _online_active(self) -> bool:
        if not self._online_eplb:
            return False
        manager = get_global_online_eplb_manager()
        return manager.active

    def _use_expand_layout(self) -> bool:
        return not self._online_active() and super()._use_expand_layout()

    def dispatch(self, hidden_states, topk_output):
        output = super().dispatch(hidden_states, topk_output)
        if self._online_eplb and not self._online_active():
            # Routing still uses M+R physical slots, but inactive calls only
            # address masters. Give the existing GEMM exactly M expert counts;
            # keep the native handle intact for combine (including graph replay).
            output = output._replace(
                psum_num_recv_tokens_per_expert=(
                    output.psum_num_recv_tokens_per_expert[: self.num_master_experts]
                )
            )
        return output

    def _dummy_topk_ids(self, topk_ids):
        ids = super()._dummy_topk_ids(topk_ids)
        if self._online_eplb:
            m = self.num_master_experts
            ids = ids.div(
                m, rounding_mode="floor"
            ) * self.num_local_experts + ids.remainder(m)
        return ids


class DeepEPv25Dispatcher(DeepEPv2Dispatcher):
    # The inherited _get_buffer() uses the same runtime-owned EPBuffer as the
    # prefetcher. Expert layout is fixed at construction; no per-layer binding.
    impl_class = _DeepEPv25Impl
