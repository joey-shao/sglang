# SPDX-License-Identifier: Apache-2.0
"""Full-TP attention projections for short forwards on a Ulysses topology."""

from torch import nn

from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class ShiftParallelProjections(nn.Module):
    """Load TP shards through the same loaders and quantizers as SP projections."""

    def __init__(self, attention, *, bias, quant_config, prefix):
        super().__init__()
        parallel = get_parallel()
        self.q_size = (
            attention.total_num_heads // parallel.attn_tp_size * attention.head_dim
        )
        self.kv_size = (
            attention.total_num_kv_heads // parallel.attn_tp_size * attention.head_dim
        )
        self.qkv_proj = QKVParallelLinear(
            attention.hidden_size,
            attention.head_dim,
            attention.total_num_heads,
            attention.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            attention.total_num_heads * attention.head_dim,
            attention.hidden_size,
            bias=bias,
            quant_config=quant_config,
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

    def prepare(self, attention, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=attention.q_norm,
            k_norm=attention.k_norm,
            head_dim=attention.head_dim,
            alt_stream=attention.alt_stream,
        )
        q, k = attention.rotary_emb(positions, q, k)
        return q, k, v


def with_shift_parallel_weights(weights):
    """Feed each attention projection to both ordinary checkpoint loaders."""
    enabled = get_parallel().enable_shift_parallel
    for name, weight in weights:
        yield name, weight
        if enabled and any(
            f".self_attn.{projection}." in name
            for projection in ("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj")
        ):
            yield name.replace(".self_attn.", ".self_attn.shift_tp."), weight
