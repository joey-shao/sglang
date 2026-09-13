# SPDX-License-Identifier: Apache-2.0
"""Ulysses token/head redistribution at common model and attention boundaries."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


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
    # BCG keeps the bucket layout fixed while attention runs eagerly on the
    # live prefix. Decode graphs leave this unset (dummy requests are valid).
    real_num_tokens: int | None = None

    @property
    def attention_tokens(self):
        return self.num_tokens if self.real_num_tokens is None else self.real_num_tokens

    def __post_init__(self):
        if not 0 <= self.attention_tokens <= self.num_tokens:
            raise ValueError("SP real token count must fit the bucket")

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
    collective = getattr(group, op, None)
    collective(output.view(-1), input_tensor.view(-1))


def exchange_qkv(q, k, v, *, metadata: SPBatchMetadata, group):
    """Local tokens / base-TP heads -> global real tokens / full-TP heads."""
    parts = [x.reshape(metadata.local_tokens, metadata.sp_size, -1) for x in (q, k, v)]
    widths = [x.shape[-1] for x in parts]
    qkv = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
    received = torch.empty_like(qkv)
    _collective("all_to_all_single", received, qkv, group)
    # Padding never reaches the backend or out_cache_loc: metadata stays global
    # and describes only real tokens, including cached-prefix/chunk boundaries.
    qkv_ = received.reshape(metadata.padded_tokens, -1)[: metadata.attention_tokens]
    return tuple(x.contiguous() for x in qkv_.split(widths, dim=-1))


def exchange_attention_output(output, *, metadata: SPBatchMetadata, group):
    output = output.flatten(start_dim=1)
    if output.shape[0] != metadata.attention_tokens:
        raise ValueError("SP attention output does not match real token count")
    padded = F.pad(
        output, (0, 0, 0, metadata.padded_tokens - metadata.attention_tokens)
    )
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


def prepare_sp_forward(
    forward_batch: ForwardBatch, *, real_num_tokens: int | None = None
) -> None:
    """Build SP metadata for a global batch without slicing its tensors.

    Input length defines the token layout (the fixed bucket for graphs).
    BCG replay supplies the live attention extent separately; eager forwards
    and decode capture use every input row. Rebuild on each call so reused
    batches cannot retain a previous replay's real token count.
    """
    strategy = get_sp_strategy()
    assert strategy is not None

    if forward_batch.positions.ndim != 1:
        raise ValueError("SP requires one-dimensional positions")
    for name in ("input_embeds", "replace_embeds", "replace_positions"):
        if getattr(forward_batch, name, None) is not None:
            raise ValueError("SP does not support embedding overrides")

    metadata = strategy.build_metadata(
        num_tokens=forward_batch.input_ids.numel(), real_num_tokens=real_num_tokens
    )
    if forward_batch.positions.shape[0] != metadata.num_tokens:
        raise ValueError("SP positions do not match the global token count")
    forward_batch.sp_metadata = metadata


@contextmanager
def sp_shard_model_inputs(input_ids, positions, forward_batch):
    """Shard model inputs while keeping ``forward_batch`` in global layout."""
    if get_sp_strategy() is None:
        yield input_ids, positions
        return

    prepare_sp_forward(forward_batch)
    try:
        metadata = forward_batch.sp_metadata
        sharded_input_ids = metadata.slice_tokens(input_ids)
        sharded_positions = metadata.slice_tokens(positions)
        yield sharded_input_ids, sharded_positions
    finally:
        delattr(forward_batch, "sp_metadata")


class UlyssesParallelStrategy:
    """Own model-input sharding, global attention metadata and SP exchanges."""

    def __init__(self, *, sp_size):
        self.sp_size = sp_size

    def build_metadata(
        self, *, num_tokens: int, real_num_tokens: int | None = None
    ) -> SPBatchMetadata:
        return SPBatchMetadata(
            num_tokens=num_tokens,
            sp_size=self.sp_size,
            sp_rank=get_parallel().ulysses_sp_group.rank_in_group,
            real_num_tokens=real_num_tokens,
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


def sp_model_forward(model, forward_batch, **kwargs):
    """Run local SP tokens and gather global hidden states before logits.

    The caller owns the model TP scope. During decode graph capture the input
    views describe the fixed global bucket, including graph dummy requests;
    only the additional SP alignment padding is removed inside attention.
    """
    strategy = get_sp_strategy()
    assert strategy is not None
    if kwargs.get("get_embedding", False):
        raise ValueError("Ulysses SP does not support embedding models")

    model_kwargs = {}
    if (pp_proxy_tensors := kwargs.get("pp_proxy_tensors")) is not None:
        model_kwargs["pp_proxy_tensors"] = pp_proxy_tensors

    with sp_shard_model_inputs(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    ) as (input_ids, positions):
        hidden_states = model.model(
            input_ids,
            positions,
            forward_batch,
            **model_kwargs,
        )
        capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
        aux_hidden_states = None
        if capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if not model.pp_group.is_last_rank:
            return (
                (hidden_states, aux_hidden_states)
                if capture_aux_hidden_states
                else hidden_states
            )
        if aux_hidden_states is not None:
            raise ValueError("SP logits do not support auxiliary hidden states")
        hidden_states = strategy.gather_hidden_states(hidden_states, forward_batch)

    return model.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        model.lm_head,
        forward_batch,
        aux_hidden_states,
    )


@dataclass
class PrefillSPBCGInput:
    """Fixed-address SP-local inputs and per-bucket replay state.

    The captured body consumes local input_embeds instead of input_ids (the
    model's forward must accept input_embeds). Request and KV metadata stay
    global; the original batch is used for logits.
    """

    input_embeds: torch.Tensor
    positions: torch.Tensor
    bucket_local_tokens: dict[int, int] = field(default_factory=dict)
    live_local_tokens: int = 0

    @classmethod
    def create(cls, runner: PrefillCudaGraphRunner) -> PrefillSPBCGInput:
        strategy = get_sp_strategy()
        assert strategy is not None
        capacity = (runner.max_num_tokens + strategy.sp_size - 1) // strategy.sp_size
        with torch.device(runner.device):
            return cls(
                input_embeds=torch.zeros(
                    (capacity, runner.model_runner.model_config.hidden_size),
                    dtype=runner.model_runner.dtype,
                ),
                positions=torch.zeros(capacity, dtype=torch.int64),
            )

    def prepare(
        self,
        runner: PrefillCudaGraphRunner,
        forward_batch: ForwardBatch,
        *,
        static_num_tokens: int,
        capture: bool,
    ) -> None:
        """Refresh metadata and copy the local shard before capture/replay."""
        if forward_batch.input_ids.numel() != static_num_tokens:
            raise ValueError("SP prefill inputs must match the global token bucket")
        prepare_sp_forward(
            forward_batch,
            real_num_tokens=None if capture else int(forward_batch.extend_num_tokens),
        )
        metadata = forward_batch.sp_metadata
        local_tokens = metadata.local_tokens
        if not capture:
            captured = self.bucket_local_tokens.get(static_num_tokens)
            if captured is None:
                raise RuntimeError(
                    f"Missing SP-local capture capacity for bucket {static_num_tokens}"
                )
            if captured != local_tokens:
                raise RuntimeError("SP prefill replay layout differs from capture")
        if local_tokens > min(self.input_embeds.shape[0], self.positions.numel()):
            raise RuntimeError("SP prefill bucket exceeds local input buffer capacity")
        if capture:
            self.bucket_local_tokens[static_num_tokens] = local_tokens

        start = metadata.sp_rank * local_tokens
        live = min(local_tokens, max(0, metadata.attention_tokens - start))
        global_input_embeds = runner.model_runner.model.get_input_embeddings()(
            forward_batch.input_ids[: metadata.attention_tokens]
        )
        self.input_embeds[:local_tokens].zero_()
        self.positions[:local_tokens].zero_()
        self.input_embeds[:live].copy_(global_input_embeds[start : start + live])
        self.positions[:live].copy_(forward_batch.positions[start : start + live])
        forward_batch.input_embeds = self.input_embeds[:local_tokens]
        forward_batch.positions = self.positions[:local_tokens]
        self.live_local_tokens = live


def execute_prefill_sp_bcg(
    runner: PrefillCudaGraphRunner,
    forward_batch: ForwardBatch,
    static_forward_batch: ForwardBatch,
    static_num_tokens: int,
    raw_num_tokens: int,
    shape_key: ShapeKey,
):
    """Replay the local SP body, then gather and compute logits eagerly.

    The caller owns the model TP scope and the backend replay session.
    """
    with runner._prefill_forward_context(
        static_forward_batch,
        num_tokens=static_num_tokens,
        raw_num_tokens=raw_num_tokens,
    ):
        hidden_states = runner.backend.replay(shape_key, static_forward_batch)
    strategy = get_sp_strategy()
    assert strategy is not None
    # Gather using the bucket's local layout before trimming to live tokens.
    # Logits then use the original global request metadata.
    hidden_states = strategy.gather_hidden_states(hidden_states, static_forward_batch)
    forward_batch.next_token_logits_buffer = (
        static_forward_batch.next_token_logits_buffer
    )
    model = runner.model_runner.model
    return model.logits_processor(
        forward_batch.input_ids,
        hidden_states[:raw_num_tokens],
        model.lm_head,
        forward_batch,
    )
