# Fixed Ulysses SP × TP for Qwen3

Each worker loads one model sharded over a TP subgroup and replicated across
SP. The entire model processes a local token shard. There is no Shift mode,
second model, or per-layer model wrapper.

## Launch

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --dtype bfloat16 \
  --tp-size 4 --ulysses-sp-size 2 \
  --attention-backend fa3 \
  --disable-overlap-schedule \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph
```

P=`tp_size` is the total worker count and scheduler communication width.
S=`ulysses_sp_size` divides P, and T=P/S is the model TP width. Embedding,
QKV/O projections, MLP and LM head all use T-way TP. With P=4 and S=2, TP
groups [0,1] and [2,3] each process a different token shard; SP communication
uses [0,2] and [1,3]. S=1 preserves normal TP.

Communication groups live in `parallel_state`: `get_ulysses_model_tp_group()`
and `get_ulysses_sp_group()` expose the owned subgroups, while
`get_ulysses_full_tp_group()` retains a borrowed reference to scheduling TP.
`ulysses_model_tp_scope()` switches model TP and attention TP together and
restores both on exit. The full-group getter remains stable inside the scope.
`UlyssesRankLayout` only computes rank layouts; no runtime topology object is stored.

## Batch and model execution

`ForwardBatch` keeps the ordinary global request, position, and KV metadata.
At the eager model boundary, `sp_shard_model_inputs` temporarily attaches the
SP token layout and slices/pads only the input IDs and positions passed to the
model. The context restores `sp_metadata` on exit, including exceptions, so
sampling and result handling continue to see the unchanged global batch.

The worker executes ModelRunner inside the model TP subgroup scope, restoring
the full scheduling group on exit, including exceptions. Hidden states stay local through every layer;
embedding, output projection and MLP collectives never combine different SP
token shards. All model weights are constructed/loaded under the same subgroup.

## Attention and output

FlashAttention metadata initialization uses the global batch directly.
Only `FlashAttentionBackend.forward_extend` and `forward_decode` contain SP
dispatch. `sp_strategy.forward_attention` exchanges local-token/base-TP-head
QKV into global-real-token/full-worker-head layout before cache writes. The
backend passes `flash_attn_with_kvcache` directly, with the FA3/FA4 version
and phase-specific global metadata. The strategy writes redistributed KV to
global cache locations before invoking the kernel; it never calls backend
`forward_*` methods. Padding is excluded. Inverse all-to-all restores local tokens and base-TP heads before
native O projection. Cache head ownership is `tp_rank * S + sp_rank`.

After the model body, the eager runner gathers hidden states across SP and
trims padding before calling LogitsProcessor; input IDs and batch metadata
remain global.
The same T-sharded LM head gathers vocabulary logits within its TP subgroup.
The worker restores the global batch for sampling and result handling, using
the original full scheduling group. Eager forward-mode normalization is
reflected in the global view without changing scheduler tensors.

## Scope and tests

Supported: native dense Qwen3, BF16, FlashAttention fa3/fa4, single-node eager
text generation, decode, uneven batches, chunked prefill and prefix reuse.
Q/KV head counts must be divisible by P; MLP intermediate size by T. CUDA
graphs, overlap, PP, DP attention, other CP, EP, speculative decoding, LoRA,
quantization, offloading, custom loaders, weight caches, embedding overrides
and online weight updates are outside the supported scope.

CPU/Gloo tests compare full SP×TP embedding/MLP/LM-head execution and per-rank
KV contents with a dense reference at SP=2, TP=1/2. They cover padding, cached
prefix continuation, scope recovery, local/global views and one model load.
The native 2/4-GPU smoke test is `test/manual/test_ulysses_qwen3_native.py`.
CPU tests do not validate CUDA kernels or performance.
