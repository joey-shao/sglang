# SPDX-License-Identifier: Apache-2.0
"""Ulysses token/head redistribution at common model and attention boundaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F


if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc


@dataclass(frozen=True)
class SPKVWrite:
    """Cache destination for redistributed KV; disabled writes still run attention."""

    pool: KVCache
    location: KVWriteLoc
    enabled: bool = True


@dataclass(frozen=True)
class UlyssesTokenView:
    num_tokens: int
    sp_size: int
    sp_rank: int

    @property
    def local_tokens(self):
        return (self.num_tokens + self.sp_size - 1) // self.sp_size

    @property
    def padded_tokens(self):
        return self.local_tokens * self.sp_size

    def slice(self, tensor):
        if tensor.shape[0] != self.num_tokens:
            raise ValueError("Ulysses token input does not match global token count")
        pad = tensor.new_zeros(
            (self.padded_tokens - self.num_tokens, *tensor.shape[1:])
        )
        padded = torch.cat((tensor, pad), dim=0)
        start = self.sp_rank * self.local_tokens
        return padded[start : start + self.local_tokens]


def _collective(op, output, input_tensor, group):
    getattr(dist, op)(output, input_tensor, group=getattr(group, "device_group", group))


def exchange_qkv(q, k, v, *, view, group):
    """Local tokens / base-TP heads -> global real tokens / full-TP heads."""
    parts = [x.reshape(view.local_tokens, view.sp_size, -1) for x in (q, k, v)]
    widths = [x.shape[-1] for x in parts]
    packed = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
    received = torch.empty_like(packed)
    _collective("all_to_all_single", received, packed, group)
    # Padding never reaches the backend or out_cache_loc: metadata stays global
    # and describes only real tokens, including cached-prefix/chunk boundaries.
    real = received.reshape(view.padded_tokens, -1)[: view.num_tokens]
    return tuple(x.contiguous() for x in real.split(widths, dim=-1))


def exchange_attention_output(output, *, view, group):
    output = output.reshape(view.num_tokens, -1)
    padded = F.pad(output, (0, 0, 0, view.padded_tokens - view.num_tokens))
    received = torch.empty_like(padded)
    _collective("all_to_all_single", received, padded.contiguous(), group)
    return (
        received.reshape(view.sp_size, view.local_tokens, -1)
        .transpose(0, 1)
        .reshape(view.local_tokens, -1)
    )


@dataclass(frozen=True)
class SPBatchMetadata:
    global_batch: object
    view: UlyssesTokenView


_STRATEGY = None


def get_global_sp_batch(batch):
    metadata = getattr(batch, "sp_metadata", None)
    if metadata is None:
        return batch
    # Eager normalizes MIXED to EXTEND on the execution view. Preserve the
    # global tensor layout while exposing the same mode to attention/logits.
    if hasattr(batch, "forward_mode"):
        metadata.global_batch.forward_mode = batch.forward_mode
    return metadata.global_batch


def get_sp_strategy():
    """Resolve fixed SP x TP independently of prefill CP."""
    from sglang.srt.runtime_context import get_parallel

    global _STRATEGY
    size = get_parallel().ulysses_sp_size
    if size <= 1:
        _STRATEGY = None
        return None
    if _STRATEGY is None or _STRATEGY.sp_size != size:
        _STRATEGY = UlyssesParallelStrategy(sp_size=size)
    return _STRATEGY


class UlyssesParallelStrategy:
    """Own local model inputs, global attention metadata and SP exchanges."""

    def __init__(self, topology=None, *, sp_size=None):
        self._topology = topology
        self.sp_size = topology.layout.sp_size if topology is not None else sp_size

    @property
    def topology(self):
        if self._topology is not None:
            return self._topology
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().ulysses_topology

    def prepare_batch(self, batch):
        """Return an idempotent local view without changing scheduler tensors."""
        metadata = getattr(batch, "sp_metadata", None)
        if metadata is not None:
            if (
                metadata.view.sp_size != self.sp_size
                or metadata.view.sp_rank != self.topology.sp_rank
            ):
                raise ValueError(
                    "ForwardBatch is already sharded for another SP layout"
                )
            return batch
        if batch.positions.ndim != 1:
            raise ValueError("SP requires one-dimensional positions")
        for name in ("input_embeds", "replace_embeds", "replace_positions"):
            if getattr(batch, name, None) is not None:
                raise ValueError("SP does not support embedding overrides")
        full = copy.copy(batch)
        local = copy.copy(batch)
        view = UlyssesTokenView(
            batch.input_ids.numel(), self.sp_size, self.topology.sp_rank
        )
        local.sp_metadata = SPBatchMetadata(full, view)
        for name in ("input_ids", "positions", "out_cache_loc", "token_type_ids"):
            value = getattr(batch, name, None)
            if value is not None:
                setattr(local, name, view.slice(value))
        local_count = max(
            0,
            min(view.local_tokens, view.num_tokens - view.sp_rank * view.local_tokens),
        )
        if hasattr(local, "global_num_token_non_padded_cpu"):
            local.global_num_token_non_padded_cpu = local_count
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            count = getattr(batch, name, None)
            if count is not None:
                setattr(local, name, count.new_full(count.shape, local_count))
        # Request-level metadata remains global. Attention and logits consume
        # full, while execution buffers copy local per-token tensors.
        return local

    def gather_logits_inputs(self, hidden, batch):
        metadata = batch.sp_metadata
        view, full = metadata.view, get_global_sp_batch(batch)
        if hidden.shape[0] != view.local_tokens:
            raise ValueError("SP logits input does not match the local token count")
        if view.num_tokens == 0:
            return full.input_ids, hidden, full
        output = hidden.new_empty((view.padded_tokens, *hidden.shape[1:]))
        _collective(
            "all_gather_into_tensor",
            output,
            hidden.contiguous(),
            self.topology.sp_group,
        )
        return full.input_ids, output[: view.num_tokens], full

    def forward_attention(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        *,
        attention_kernel: Callable[..., torch.Tensor],
        kv_write: SPKVWrite,
    ):
        """Exchange QKV, write global KV, call the kernel and restore local tokens.

        attention_kernel binds backend-specific options and accepts only q.
        kv_write describes where to store the exchanged K/V before attention.
        """
        if k is None or v is None:
            raise ValueError("Ulysses requires dense Q, K and V")
        metadata = forward_batch.sp_metadata
        view = metadata.view
        attention = copy.copy(layer)
        for name in ("tp_q_head_num", "tp_k_head_num", "tp_v_head_num"):
            heads = getattr(layer, name)
            if heads % self.sp_size:
                raise ValueError("Ulysses heads must divide evenly across SP")
            setattr(attention, name, heads // self.sp_size)
        q, k, v = exchange_qkv(q, k, v, view=view, group=self.topology.sp_group)
        k = k.view(-1, attention.tp_k_head_num, attention.qk_head_dim)
        v = v.view(-1, attention.tp_v_head_num, attention.v_head_dim)
        from sglang.srt.runtime_context import get_parallel

        full = self.topology.full_tp_group
        with get_parallel().override(
            attn_tp_group=full,
            attn_tp_size=full.world_size,
            attn_tp_rank=self.topology.full_tp_shard_rank,
        ):
            if kv_write.enabled:
                kv_write.pool.set_kv_buffer(
                    attention,
                    kv_write.location,
                    k,
                    v,
                    attention.k_scale,
                    attention.v_scale,
                )
            output = attention_kernel(
                q=q.view(-1, attention.tp_q_head_num, attention.qk_head_dim),
            )
        return exchange_attention_output(
            output, view=view, group=self.topology.sp_group
        )
