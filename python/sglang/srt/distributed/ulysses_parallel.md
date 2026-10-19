# Ulysses sequence parallelism for Qwen3

Ulysses runs on native Qwen3 dense and Qwen3 MoE attention. Attention QKV and O
projections are constructed with their Ulysses TP layout and loaded through the
ordinary parameter weight loaders. Fixed Ulysses uses one attention projection
weight set per layer. FFN/expert weights, embeddings, norms and the LM head keep their
ordinary parallel layout.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --dtype bfloat16 \
  --tp-size 4 --ulysses-sp-size 2 \
  --attention-backend fa3 \
  --cuda-graph-backend-prefill breakable --cuda-graph-backend-decode full
```

Use a native Qwen3 MoE checkpoint, such as `Qwen/Qwen3-30B-A3B`, for sparse
FFNs. FlashAttention (`fa3` or `fa4`) is required for both prefill and decode. Setting
`--ulysses-sp-size` greater than one enables Ulysses for both prefill and decode.

### Switching by forward length

Add `--enable-shift-parallel --shift-parallel-threshold 1024` to select ordinary
TP below 1024 tokens and Ulysses SP at or above 1024 tokens. The threshold uses
the total input token count of this forward, not the cached context length or
the longest request. It applies to both prefill and decode. When
`--ulysses-sp-size` is left at 1, shift parallel uses `--tp-size` as the SP size;
an explicit larger SP size retains its SP x TP topology.

Both modes use the same attention instance, per-rank KV head ownership and
cache locations. Prefix caching and subsequent forwards can therefore switch
in either direction without migrating KV. Shift parallel additionally loads
one full-TP shard of QKV/O projection weights per rank through the ordinary
checkpoint loaders, including FP8 scales. This costs one extra ordinary-TP
attention projection shard; norms, FFNs and cache storage are shared.

CUDA graph buckets capture the mode selected by their token count. If padding
would cross the threshold, that forward runs eagerly in its selected mode.
Other buckets retain prefill breakable and decode full graph execution.
Without `--enable-shift-parallel`, fixed Ulysses behavior is unchanged.

## Weights and groups

P=`tp_size` remains the model/scheduler TP width. S=`ulysses_sp_size` divides
P, and T=P/S is the SP attention projection TP width. For P=4, S=2, projection
TP groups are [0,2] and [1,3]; SP groups are [0,1] and [2,3].
`ParallelContext.attn_proj_tp_rank`, `attn_proj_tp_size`, and `attn_proj_tp_group`
describe the projection layout for model construction and O-output reduction.
They use ordinary attention TP when SP is disabled and the T-way Ulysses subgroup
when it is enabled. Backend/cache `attn_tp_*` values retain their ordinary layout;
no full-TP alias or runtime TP context switch is needed.

The ordinary loader reads an unsharded checkpoint. QKVParallelLinear selects
Q/K/V output rows using its T-way projection rank; RowParallelLinear selects
O projection input columns using the same rank. Ranks within an SP group load
the same projection shard. No weight all-gather, repacking, or module replacement
is needed after loading, so there is no Ulysses-specific serial startup constraint.
Each layer has one RadixAttention, constructed with T-way input head counts and
the SP size. It keeps P-way backend/cache head counts separately and owns the
QKV and output exchanges around the ordinary attention backend. The backend
writes KV and runs attention using the same layer instance and scales.
There is no second model instance or model-wide TP scope. Loading `presharded`
or `sharded_state` checkpoints is not supported: ordinary full-TP shards do not
match the projection layout.

Q/KV head counts must be divisible by P. KV head replication is not supported.

Native `fp8` quantization is accepted for Qwen3 dense and Qwen3 MoE, including
per-tensor and block-scaled FP8 checkpoints. QKV/O weights and block scales use
the same T-way projection partition; per-tensor weight and activation scales
are replicated as required by the ordinary FP8 loader. FFN/expert weights retain
their ordinary parallel layout. Partitioned dimensions must align with the
checkpoint's quantization blocks, using the actual linear-layer partition rather
than the global TP width. Other quantization formats remain unsupported.

For an unsharded native FP8 checkpoint, use `--quantization fp8 --dtype bfloat16`
with the Ulysses arguments above. BF16 checkpoints can also use `--quantization
fp8` for the existing online per-tensor quantization path. FP8 GEMMs return
BF16/FP16 tensors before attention, SP exchanges and boundary reductions; no
FP8 communication or new FP8 KV-cache path is introduced.

`check_ulysses_sp_args` validates the resolved model, topology, attention backend,
checkpoint format and execution modes during server argument checking, before
workers create communication groups or load weights. These configuration checks
are not repeated by the model loader, rank layout or attention layers.

## Attention and FFN boundaries

The runner computes embeddings using ordinary TP, then
`sp_shard_model_inputs` supplies local embeddings, positions and input IDs to
the original transformer body. The shared batch retains global request and
attention metadata. Attention inputs and residuals stay on the same contiguous,
padded SP token shard throughout the stack.

QKV all-to-all changes local-token/T-way-head layout into
global-token/P-way-head layout. Cache writes and attention use global request
metadata and the same head ownership as ordinary TP. Inverse all-to-all
restores local tokens before O projection, whose partial output is reduced
only over the T-way attention group at the FFN input boundary.

Like CP, FFN input/output boundaries own token movement. Dense FFNs and ordinary
TP experts gather the normalized local input, compute with their original
P-way TP weights, then return the completed output to the local token shard.
Native all-to-all MoE dispatch consumes local rows with ep_size == tp_size.
The residual stays local in both cases, including the final layer and norm.

`sp_gather_after_forward` gathers the body output and auxiliary captures at the
runner boundary, before the shared logits processor or pooler. The original
transformer loop, residual stream and expert forwards are reused; attention
boundaries do not split or gather tokens.

Attention writes the global KV locations with ordinary per-rank head ownership.
Extra SP padding never enters attention or cache writes.

## CUDA graphs

Decode captures the complete forward, including embedding, attention
redistribution, FFN communication and logits. Breakable prefill captures the
transformer body with local fixed-address embeddings/positions; attention
and its exchanges run at the existing eager attention breaks. The body
returns local hidden states, which the runner gathers for the eager logits tail.

The global bucket B fixes each attention shard at ceil(B/S) rows. Replay
updates the live attention extent N independently. All-to-all removes padded
rows before cache writes, and the inverse exchange restores the bucket's
local width, including ranks with no live tokens.

## Validation

Validation should compare fixed Ulysses execution against ordinary TP for Qwen3
dense and MoE models with 2/4 workers. Cover cached-prefix continuation,
uneven token counts, padding-only ranks and graph replay.
Compare both outputs and per-rank KV contents against ordinary TP, and verify
that only the attention projections use the smaller TP width. Check direct
checkpoint slices against the equivalent concatenated full-TP weight shards.
For FP8, also compare checkpoint scale slices and run the same checkpoint with
SP disabled/enabled using BF16 activations and KV cache. Cover both per-tensor
and block scaling. Dynamic activation scales can differ with token partitioning,
so use numerical tolerances rather than requiring bitwise-identical outputs.

CPU collective simulations can check token/head layouts and boundary contracts.
CUDA/NCCL, native MoE dispatch, FlashAttention kernels and capture/replay require
GPU validation; CPU checks do not establish their correctness or speed.
The execution path targets single-domain text generation without PP, DP
attention, another CP mode, speculative decoding, LoRA or torch.compile.
Online weight updates remain unsupported.
