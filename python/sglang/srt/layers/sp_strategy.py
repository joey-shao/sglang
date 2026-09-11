# SPDX-License-Identifier: Apache-2.0
"""Ulysses token/head redistribution at common model and attention boundaries."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc


@dataclass(frozen=True)
class SPKVWrite:
    """Cache destination for redistributed KV; disabled writes still run attention."""

    pool: KVCache
    location: KVWriteLoc
    enabled: bool = True


@dataclass(frozen=True)
class SPBatchMetadata:
    """Token counts and the SP layout used to shard model inputs."""

    num_tokens: int
    sp_size: int
    sp_rank: int

    @property
    def local_tokens(self):
        return (self.num_tokens + self.sp_size - 1) // self.sp_size

    @property
    def padded_tokens(self):
        return self.local_tokens * self.sp_size

    def slice_tokens(self, tensor):
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


def exchange_qkv(q, k, v, *, metadata: SPBatchMetadata, group):
    """Local tokens / base-TP heads -> global real tokens / full-TP heads."""
    parts = [x.reshape(metadata.local_tokens, metadata.sp_size, -1) for x in (q, k, v)]
    widths = [x.shape[-1] for x in parts]
    packed = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
    received = torch.empty_like(packed)
    _collective("all_to_all_single", received, packed, group)
    # Padding never reaches the backend or out_cache_loc: metadata stays global
    # and describes only real tokens, including cached-prefix/chunk boundaries.
    real = received.reshape(metadata.padded_tokens, -1)[: metadata.num_tokens]
    return tuple(x.contiguous() for x in real.split(widths, dim=-1))


def exchange_attention_output(output, *, metadata: SPBatchMetadata, group):
    output = output.reshape(metadata.num_tokens, -1)
    padded = F.pad(output, (0, 0, 0, metadata.padded_tokens - metadata.num_tokens))
    received = torch.empty_like(padded)
    _collective("all_to_all_single", received, padded.contiguous(), group)
    return (
        received.reshape(metadata.sp_size, metadata.local_tokens, -1)
        .transpose(0, 1)
        .reshape(metadata.local_tokens, -1)
    )


_STRATEGY = None


def get_sp_strategy():
    """Resolve fixed SP x TP independently of prefill CP."""
    global _STRATEGY
    size = get_parallel().ulysses_sp_size
    if size <= 1:
        _STRATEGY = None
        return None
    if _STRATEGY is None or _STRATEGY.sp_size != size:
        _STRATEGY = UlyssesParallelStrategy(sp_size=size)
    return _STRATEGY


@contextmanager
def sp_shard_model_inputs(input_ids, positions, forward_batch):
    """Shard model inputs while keeping ``forward_batch`` in global layout."""
    strategy = get_sp_strategy()
    if strategy is None:
        yield input_ids, positions
        return

    had_sp_metadata = hasattr(forward_batch, "sp_metadata")
    sp_metadata_backup = getattr(forward_batch, "sp_metadata", None)
    if sp_metadata_backup is not None:
        raise ValueError("ForwardBatch is already sharded for SP")
    if positions.ndim != 1:
        raise ValueError("SP requires one-dimensional positions")
    for name in ("input_embeds", "replace_embeds", "replace_positions"):
        if getattr(forward_batch, name, None) is not None:
            raise ValueError("SP does not support embedding overrides")

    metadata = strategy.build_metadata(num_tokens=input_ids.numel())
    if positions.shape[0] != metadata.num_tokens:
        raise ValueError("SP positions do not match the global token count")
    sharded_input_ids = metadata.slice_tokens(input_ids)
    sharded_positions = metadata.slice_tokens(positions)

    forward_batch.sp_metadata = metadata
    try:
        yield sharded_input_ids, sharded_positions
    finally:
        if had_sp_metadata:
            forward_batch.sp_metadata = sp_metadata_backup
        else:
            delattr(forward_batch, "sp_metadata")


class UlyssesParallelStrategy:
    """Own model-input sharding, global attention metadata and SP exchanges."""

    def __init__(self, *, sp_size):
        self.sp_size = sp_size

    def build_metadata(self, *, num_tokens: int) -> SPBatchMetadata:
        return SPBatchMetadata(
            num_tokens=num_tokens,
            sp_size=self.sp_size,
            sp_rank=get_parallel().ulysses_sp_group.rank_in_group,
        )

    def gather_hidden_states(self, hidden_states, batch):
        metadata = batch.sp_metadata
        if hidden_states.shape[0] != metadata.local_tokens:
            raise ValueError("SP hidden states do not match the local token count")
        if metadata.num_tokens == 0:
            return hidden_states
        output = hidden_states.new_empty(
            (metadata.padded_tokens, *hidden_states.shape[1:])
        )
        _collective(
            "all_gather_into_tensor",
            output,
            hidden_states.contiguous(),
            get_parallel().ulysses_sp_group,
        )
        return output[: metadata.num_tokens]

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
        attention = copy.copy(layer)
        for name in ("tp_q_head_num", "tp_k_head_num", "tp_v_head_num"):
            heads = getattr(layer, name)
            if heads % self.sp_size:
                raise ValueError("Ulysses heads must divide evenly across SP")
            setattr(attention, name, heads // self.sp_size)
        q, k, v = exchange_qkv(
            q, k, v, metadata=metadata, group=get_parallel().ulysses_sp_group
        )
        k = k.view(-1, attention.tp_k_head_num, attention.qk_head_dim)
        v = v.view(-1, attention.tp_v_head_num, attention.v_head_dim)
        full = get_parallel().ulysses_full_tp_group
        with get_parallel().override(
            attn_tp_group=full,
            attn_tp_size=full.world_size,
            attn_tp_rank=get_parallel().ulysses_attention_shard_rank,
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
            output, metadata=metadata, group=get_parallel().ulysses_sp_group
        )
