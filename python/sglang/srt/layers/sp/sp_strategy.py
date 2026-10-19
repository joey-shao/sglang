# SPDX-License-Identifier: Apache-2.0
"""Ulysses token/head redistribution at common model and attention boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


@dataclass(frozen=True)
class SPBatchMetadata:
    """Global input/attention extents and the derived SP communication layout.

    attn_num_tokens <= layout_num_tokens <= exchange_num_tokens.
    """

    # Global input rows, including graph padding but excluding SP alignment.
    layout_num_tokens: int
    # Global rows consumed by attention after all-to-all. Decode graph dummy
    # requests count here; prefill BCG replay excludes unused bucket rows.
    attn_num_tokens: int
    sp_size: int
    sp_rank: int

    @property
    def local_num_tokens(self) -> int:
        return (self.layout_num_tokens + self.sp_size - 1) // self.sp_size

    @property
    def exchange_num_tokens(self) -> int:
        return self.local_num_tokens * self.sp_size

    def slice_tokens(self, tensor):
        pad = tensor.new_zeros(
            (self.exchange_num_tokens - self.layout_num_tokens, *tensor.shape[1:])
        )
        padded = torch.cat((tensor, pad), dim=0)
        start = self.sp_rank * self.local_num_tokens
        return padded[start : start + self.local_num_tokens]


def exchange_qkv(q, k, v, *, metadata: SPBatchMetadata, group):
    """Local layout / base-TP heads -> global attention rows / full-TP heads."""
    parts = [
        x.reshape(metadata.local_num_tokens, metadata.sp_size, -1) for x in (q, k, v)
    ]
    widths = [x.shape[-1] for x in parts]
    qkv = torch.cat(parts, dim=-1).transpose(0, 1).contiguous()
    received = torch.empty_like(qkv)
    group.all_to_all_single(received.view(-1), qkv.view(-1))
    # Remove SP alignment and unused graph rows only after exchanging the fixed
    # local layout. Decode graph dummy requests remain part of attention.
    qkv_ = received.reshape(metadata.exchange_num_tokens, -1)[
        : metadata.attn_num_tokens
    ]
    return tuple(x.contiguous() for x in qkv_.split(widths, dim=-1))


def exchange_attention_output(output, *, metadata: SPBatchMetadata, group):
    output = output.flatten(start_dim=1)
    padded = F.pad(
        output, (0, 0, 0, metadata.exchange_num_tokens - metadata.attn_num_tokens)
    )
    received = torch.empty_like(padded)
    group.all_to_all_single(received.view(-1), padded.contiguous().view(-1))
    return (
        received.reshape(metadata.sp_size, metadata.local_num_tokens, -1)
        .transpose(0, 1)
        .reshape(metadata.local_num_tokens, -1)
    )


def get_sp_strategy(num_tokens: int | None = None):
    """Resolve SP for a forward, or report availability when no length is given."""
    parallel = get_parallel()
    if parallel.ulysses_sp_size == 1:
        return None
    if (
        num_tokens is not None
        and parallel.enable_shift_parallel
        and num_tokens < parallel.shift_parallel_threshold
    ):
        return None
    return _STRATEGY


def prepare_sp_forward(forward_batch: ForwardBatch, *, attn_num_tokens: int) -> None:
    """Build SP metadata for a global batch without slicing its tensors.

    Input length defines the token layout (the fixed bucket for graphs).
    Callers supply the attention extent explicitly: BCG replay uses its live
    prefix; eager forwards and graph capture use every input row. Rebuild on
    each call so reused batches cannot retain a previous replay's extent.
    """
    forward_batch.sp_metadata = get_sp_strategy().build_metadata(
        layout_num_tokens=forward_batch.input_ids.numel(),
        attn_num_tokens=attn_num_tokens,
    )


@contextmanager
def sp_shard_model_inputs(
    complete_hidden_states, complete_positions, forward_batch, complete_input_ids
):
    """Enter the ordinary transformer body with local inputs and global batch metadata."""
    prepare_sp_forward(forward_batch, attn_num_tokens=forward_batch.input_ids.numel())
    metadata = forward_batch.sp_metadata
    try:
        yield (
            metadata.slice_tokens(complete_hidden_states),
            metadata.slice_tokens(complete_positions),
            metadata.slice_tokens(complete_input_ids),
        )
    finally:
        forward_batch.sp_metadata = None


class UlyssesParallelStrategy:
    """Own attention token layouts, global metadata and SP exchanges."""

    def build_metadata(
        self, *, layout_num_tokens: int, attn_num_tokens: int
    ) -> SPBatchMetadata:
        return SPBatchMetadata(
            layout_num_tokens=layout_num_tokens,
            attn_num_tokens=attn_num_tokens,
            sp_size=get_parallel().ulysses_sp_size,
            sp_rank=get_parallel().ulysses_sp_group.rank_in_group,
        )

    def gather_hidden_states(self, hidden_states, batch):
        metadata = batch.sp_metadata
        if metadata.layout_num_tokens == 0:
            return hidden_states
        output = hidden_states.new_empty(
            (metadata.exchange_num_tokens, *hidden_states.shape[1:])
        )
        get_parallel().ulysses_sp_group.all_gather_into_tensor(
            output.view(-1), hidden_states.contiguous().view(-1)
        )
        return output[: metadata.layout_num_tokens]


_STRATEGY = UlyssesParallelStrategy()


def sp_gather_after_forward(output, forward_batch):
    """Restore global tokens after the local body, including auxiliary captures."""
    if isinstance(output, tuple):
        return tuple(sp_gather_after_forward(item, forward_batch) for item in output)
    if isinstance(output, list):
        return [sp_gather_after_forward(item, forward_batch) for item in output]
    hidden_states = get_sp_strategy().gather_hidden_states(output, forward_batch)
    return hidden_states[: forward_batch.sp_metadata.attn_num_tokens]


@torch.no_grad()
def sp_model_forward(
    model,
    forward_batch,
    *,
    input_embeds=None,
    pp_proxy_tensors=None,
    get_embedding=False,
):
    """Shard at the model entrance, run the original body, then gather for logits."""
    if input_embeds is None:
        input_embeds = model.get_input_embeddings()(forward_batch.input_ids)
    with sp_shard_model_inputs(
        input_embeds, forward_batch.positions, forward_batch, forward_batch.input_ids
    ) as (local_embeds, local_positions, local_ids):
        output = model.model(
            local_ids,
            local_positions,
            forward_batch,
            input_embeds=local_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        output = sp_gather_after_forward(output, forward_batch)
    hidden_states, aux_hidden_states = (
        output if isinstance(output, tuple) else (output, None)
    )
    if get_embedding:
        return model.pooler(hidden_states, forward_batch)
    return model.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        model.lm_head,
        forward_batch,
        aux_hidden_states,
    )


@dataclass
class PrefillSPBCGInput:
    """Fixed-address local inputs for a body that retains its SP token shard."""

    input_embeds: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def create(cls, runner: PrefillCudaGraphRunner) -> PrefillSPBCGInput:
        sp_size = get_parallel().ulysses_sp_size
        local_capacity = (runner.max_num_tokens + sp_size - 1) // sp_size
        with torch.device(runner.device):
            return cls(
                input_embeds=torch.zeros(
                    (
                        local_capacity,
                        runner.model_runner.model_config.hidden_size,
                    ),
                    dtype=runner.model_runner.dtype,
                ),
                positions=torch.zeros(local_capacity, dtype=torch.int64),
            )

    def prepare(
        self,
        runner: PrefillCudaGraphRunner,
        forward_batch: ForwardBatch,
        *,
        static_num_tokens: int,
        capture: bool,
    ) -> None:
        prepare_sp_forward(
            forward_batch,
            attn_num_tokens=(
                forward_batch.input_ids.numel()
                if capture
                else int(forward_batch.extend_num_tokens)
            ),
        )
        metadata = forward_batch.sp_metadata
        live = metadata.attn_num_tokens
        global_embeds = runner.model_runner.model.get_input_embeddings()(
            forward_batch.input_ids[:live]
        )
        global_embeds = F.pad(global_embeds, (0, 0, 0, static_num_tokens - live))
        global_positions = F.pad(
            forward_batch.positions[:live], (0, static_num_tokens - live)
        )
        embeds = self.input_embeds[: metadata.local_num_tokens]
        positions = self.positions[: metadata.local_num_tokens]
        embeds.copy_(metadata.slice_tokens(global_embeds))
        positions.copy_(metadata.slice_tokens(global_positions))
        forward_batch.input_embeds = embeds
        forward_batch.positions = positions


def execute_prefill_sp_bcg(
    runner: PrefillCudaGraphRunner,
    forward_batch: ForwardBatch,
    static_forward_batch: ForwardBatch,
    static_num_tokens: int,
    raw_num_tokens: int,
    shape_key: ShapeKey,
):
    """Gather the local captured body output before the eager logits tail."""
    with runner._prefill_forward_context(
        static_forward_batch,
        num_tokens=static_num_tokens,
        raw_num_tokens=raw_num_tokens,
    ):
        output = runner.backend.replay(shape_key, static_forward_batch)
        output = sp_gather_after_forward(output, static_forward_batch)
    hidden_states, aux_hidden_states = (
        output if isinstance(output, tuple) else (output, None)
    )
    forward_batch.next_token_logits_buffer = (
        static_forward_batch.next_token_logits_buffer
    )
    model = runner.model_runner.model
    return model.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        model.lm_head,
        forward_batch,
        aux_hidden_states,
    )
