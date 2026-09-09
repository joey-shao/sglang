# SPDX-License-Identifier: Apache-2.0
"""Real Gloo execution of the SP adapter against a dense PyTorch reference.

The CPU attention fixture implements paged-cache semantics by request/position;
native FlashAttention and Qwen3 kernels are covered by the GPU integration test.
"""

from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed.ulysses_parallel import UlyssesRankLayout
from sglang.srt.layers.sp_strategy import (
    UlyssesParallelStrategy,
    UlyssesTokenView,
    get_global_sp_batch,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def load_backend(backend_name, kernel=None):
    """Load the real dispatcher with only hardware/logging imports stubbed.

    Attention dispatch and the SP branch run unchanged; the native kernel
    body is replaced with CPU reference attention below.
    """
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType
    from unittest.mock import patch

    logging = ModuleType("sglang.kernels.kernel_api_logging")
    logging.debug_kernel_api = lambda fn: fn
    common = ModuleType("sglang.srt.utils.common")
    common.is_npu = lambda: False
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/base_attn_backend.py"
    )
    spec = importlib.util.spec_from_file_location("ulysses_test_backend", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "sglang.kernels.kernel_api_logging": logging,
            "sglang.srt.utils.common": common,
        },
    ):
        spec.loader.exec_module(module)
    # Compile the concrete backend's actual extend/decode methods in a minimal class.
    # Its GPU imports/constructor/kernels are excluded; the SP branch and
    # direct kernel call execute unchanged.
    import ast

    path = path.with_name(backend_name + ".py")
    tree = ast.parse(path.read_text())
    names = {
        "flashattention_backend": "FlashAttentionBackend",
    }
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == names[backend_name]
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("forward_extend", "forward_decode")
    ]
    assert len(cls.body) == 2
    for method in cls.body:
        # Preserve the production SP guard/callback, replacing only the ordinary
        # kernel path starting at the FA4 score_mod validation.
        kernel_start = next(
            i
            for i, statement in enumerate(method.body)
            if isinstance(statement, ast.If)
            and "score_mod" in ast.unparse(statement.test)
        )
        arguments = method.args.args[1:]
        reference_call = ast.Return(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_" + method.name,
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in arguments[:5]],
                keywords=[
                    ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load()))
                    for arg in arguments[5:]
                ],
            )
        )
        method.body = method.body[:kernel_start] + [reference_call]
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {
        "AttentionBackend": module.AttentionBackend,
        "partial": partial,
        "AttentionType": SimpleNamespace(DECODER="decoder"),
        "KVWriteLoc": lambda loc, swa_loc: SimpleNamespace(loc=loc, swa_loc=swa_loc),
        "flash_attn_with_kvcache": kernel,
    }
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace[cls.name]


class Norm:
    def __call__(self, x, residual=None):
        summed = x if residual is None else x + residual
        result = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        return result if residual is None else (result, summed)


class Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, skip_all_reduce=False):
        return F.linear(x, self.weight), None


class CacheAttention:
    def __init__(self, q_heads, kv_heads, cache):
        self.tp_q_head_num = q_heads
        self.tp_k_head_num = self.tp_v_head_num = kv_heads
        self.cache = cache
        self.qk_head_dim = self.head_dim = self.v_head_dim = 2
        self.k_scale = self.v_scale = None
        self.is_cross_attention = False
        self.attn_type = "decoder"
        self.sliding_window_size = -1
        self.scaling = 2**-0.5
        self.logit_cap = 0

    def __call__(self, q, k, v, batch):
        assert len(q) == len(batch.tokens)  # padding never reaches cache writes
        q = q.reshape(-1, self.tp_q_head_num, 2)
        k = k.reshape(-1, self.tp_k_head_num, 2)
        v = v.reshape_as(k)
        outputs = []
        for i, (request, position) in enumerate(batch.tokens):
            self.cache[request, position] = (k[i].clone(), v[i].clone())
            keys = torch.stack([self.cache[request, p][0] for p in range(position + 1)])
            values = torch.stack(
                [self.cache[request, p][1] for p in range(position + 1)]
            )
            repeats = self.tp_q_head_num // self.tp_k_head_num
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
            scores = torch.einsum("hd,thd->ht", q[i], keys) / 2**0.5
            outputs.append(
                torch.einsum("ht,thd->hd", scores.softmax(-1), values).flatten()
            )
        return torch.stack(outputs)


def prepare(qkv_weights, positions, hidden):
    q, k, v = [F.linear(hidden, w).reshape(len(hidden), -1, 2) for w in qkv_weights]
    q, k = Norm()(q), Norm()(k)
    # A two-dimensional RoPE with explicit token positions.
    c, s = positions[:, None].cos(), positions[:, None].sin()

    def rope(x):
        return torch.stack(
            (x[..., 0] * c - x[..., 1] * s, x[..., 0] * s + x[..., 1] * c), dim=-1
        )

    return rope(q).flatten(1), rope(k).flatten(1), v.flatten(1)


def _worker(rank, p, backend_name, fa_version, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=p,
        timeout=timedelta(seconds=30),
    )
    layout = UlyssesRankLayout(p, 2)
    groups = []

    def make_group(rank_lists):
        selected = None
        for ranks in rank_lists:
            group = dist.new_group(ranks, timeout=timedelta(seconds=30))
            if rank in ranks:
                groups.append(group)
                selected = group
        return selected

    base_group, sp_group = make_group(layout.base_tp_ranks), make_group(layout.sp_ranks)
    try:
        t = layout.base_tp_size
        sp_rank, tp_rank = layout.coordinates(rank)

        def reduce(x):
            dist.all_reduce(x, group=base_group)
            return x

        in_base_tp_scope = False

        @contextmanager
        def base_tp_scope():
            nonlocal in_base_tp_scope
            assert not in_base_tp_scope
            in_base_tp_scope = True
            try:
                with get_parallel().override(attn_tp_size=t, attn_tp_rank=tp_rank):
                    yield
            finally:
                in_base_tp_scope = False

        topology = SimpleNamespace(
            layout=layout,
            sp_rank=sp_rank,
            base_tp_group=SimpleNamespace(world_size=t, all_reduce=reduce),
            sp_group=SimpleNamespace(device_group=sp_group),
            model_tp_scope=base_tp_scope,
            full_tp_group=SimpleNamespace(world_size=p),
            full_tp_shard_rank=layout.full_tp_shard_rank(rank),
        )
        torch.manual_seed(123)
        embedding, lm_head = torch.randn(24, 16), torch.randn(24, 16)
        layers, full_weights, reference_caches = [], [], []
        local_caches = []
        for _ in range(2):
            q, k, v = [torch.randn(n, 16) * 0.1 for n in (16, 8, 8)]
            o, gate, up, down = [
                torch.randn(*shape) * 0.1
                for shape in ((16, 16), (32, 16), (32, 16), (16, 32))
            ]
            cache = {}
            local_caches.append(cache)
            reference_caches.append({})
            local_qkv = [w.chunk(t, dim=0)[tp_rank] for w in (q, k, v)]
            attn = SimpleNamespace(
                attn=CacheAttention(8 // t, 4 // t, cache),
                forward_prepare_native=lambda positions, hidden, w=local_qkv: prepare(
                    w, positions, hidden
                ),
                o_proj=Linear(o.chunk(t, dim=1)[tp_rank]),
            )
            mlp = SimpleNamespace(
                gate_up_proj=Linear(
                    torch.cat((gate.chunk(t)[tp_rank], up.chunk(t)[tp_rank]))
                ),
                act_fn=lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1],
                down_proj=Linear(down.chunk(t, dim=1)[tp_rank]),
            )
            layers.append(
                SimpleNamespace(
                    self_attn=attn,
                    mlp=mlp,
                    input_layernorm=Norm(),
                    post_attention_layernorm=Norm(),
                )
            )
            full_weights.append((q, k, v, o, gate, up, down))

        def embed(ids):
            start, end = tp_rank * (24 // t), (tp_rank + 1) * (24 // t)
            output = F.embedding(
                ids.clamp(start, end - 1) - start, embedding[start:end]
            ).clone()
            output[(ids < start) | (ids >= end)] = 0
            return reduce(output)

        strategy = UlyssesParallelStrategy(topology)
        from sglang.srt.layers import sp_strategy

        sp_strategy.get_sp_strategy = lambda: strategy
        pending_kv = {}

        def write_kv(descriptor, loc, k, v, k_scale, v_scale):
            assert get_parallel().attn_tp_size == p
            torch.testing.assert_close(
                loc.loc, torch.arange(len(backend.batch.input_ids))
            )
            assert len(k) == len(backend.batch.input_ids)
            pending_kv.update(descriptor=descriptor, k=k, v=v)

        def reference_kernel(*, q, **kwargs):
            assert get_parallel().attn_tp_size == p
            assert q.shape[0] == len(backend.batch.input_ids)
            assert kwargs["ver"] == backend.fa_impl_ver
            assert kwargs["causal"] is True
            assert kwargs["page_table"] is backend.forward_metadata.page_table
            if backend.batch.forward_mode.is_decode():
                assert kwargs["num_splits"] == backend.decode_num_splits
                assert (
                    kwargs["scheduler_metadata"]
                    is backend.forward_metadata.scheduler_metadata
                )
                assert "cu_seqlens_k_new" not in kwargs
            else:
                assert kwargs["num_splits"] == backend.num_splits
                assert (
                    kwargs["cu_seqlens_k_new"] is backend.forward_metadata.cu_seqlens_k
                )
            if backend.fail:
                raise RuntimeError("backend failed")
            return pending_kv["descriptor"](
                q, pending_kv["k"], pending_kv["v"], backend.batch
            )

        Backend = load_backend(backend_name, kernel=reference_kernel)

        class ReferenceBackend(Backend):
            fail = False
            use_mla = fa_skip_kv_cache = has_local_attention = False
            _decode_uses_static_max_seqlen_k = True
            fa_impl_ver = fa_version
            num_splits = 1
            decode_num_splits = 2
            token_to_kv_pool = SimpleNamespace(set_kv_buffer=write_kv)
            forward_metadata = SimpleNamespace(
                page_table=object(),
                cache_seqlens_int32=object(),
                cu_seqlens_q=object(),
                cu_seqlens_k=object(),
                max_seq_len_q=1,
                max_seq_len_k=16,
                swa_out_cache_loc=None,
                scheduler_metadata=object(),
            )

            def get_paged_mha_kv_cache(self, layer, *, head_group_num):
                assert head_group_num == strategy.sp_size
                return None, None

        backend = ReferenceBackend()

        def native_forward(input_ids, positions, batch):
            # Simulate ForwardBatch.init_new followed by the worker TP scope.
            backend.batch = batch
            batch.positions = positions
            batch.out_cache_loc = torch.arange(len(input_ids))
            local = strategy.prepare_batch(batch)
            assert strategy.prepare_batch(local) is local
            with topology.model_tp_scope():
                hidden = embed(local.input_ids)
                residual = None
                for layer in layers:
                    assert hidden.shape[0] == local.sp_metadata.view.local_tokens
                    if residual is None:
                        residual = hidden
                        hidden = layer.input_layernorm(hidden)
                    else:
                        hidden, residual = layer.input_layernorm(hidden, residual)
                    q, k, v = layer.self_attn.forward_prepare_native(
                        local.positions, hidden
                    )
                    forward = (
                        backend.forward_decode
                        if batch.forward_mode.is_decode()
                        else backend.forward_extend
                    )
                    hidden = forward(q, k, v, layer.self_attn.attn, local)
                    hidden, _ = layer.self_attn.o_proj(hidden)
                    hidden = reduce(hidden)
                    assert in_base_tp_scope
                    assert get_parallel().attn_tp_size == t
                    hidden, residual = layer.post_attention_layernorm(hidden, residual)
                    gate_up, _ = layer.mlp.gate_up_proj(hidden)
                    hidden, _ = layer.mlp.down_proj(layer.mlp.act_fn(gate_up))
                    hidden = reduce(hidden)
                hidden, _ = Norm()(hidden, residual)
                ids, hidden, global_batch = strategy.gather_logits_inputs(hidden, local)
                torch.testing.assert_close(ids, input_ids)
                assert global_batch.input_ids is batch.input_ids
                local_logits = F.linear(hidden, lm_head.chunk(t)[tp_rank])
                logits = [torch.empty_like(local_logits) for _ in range(t)]
                dist.all_gather(logits, local_logits, group=base_group)
                return torch.cat(logits, dim=-1)

        steps = [
            ([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], [1, 2, 3, 4, 5]),
            ([(0, 3), (1, 2)], [6, 7]),
            ([(0, 4)], [8]),
            ([(1, 3), (1, 4), (1, 5)], [9, 10, 11]),
        ]
        for tokens, ids in steps:
            batch = SimpleNamespace(
                tokens=tokens,
                forward_mode=SimpleNamespace(
                    is_idle=lambda: False,
                    is_decode=lambda: (
                        all(pos > 0 for _, pos in tokens) and len(tokens) <= 2
                    ),
                    is_mixed=lambda: False,
                ),
            )
            input_ids = torch.tensor(ids)
            positions = torch.tensor([pos for _, pos in tokens])
            batch.input_ids = input_ids
            actual = native_forward(input_ids, positions, batch)
            assert not in_base_tp_scope
            hidden = F.embedding(input_ids, embedding)
            residual = None
            for weights, cache in zip(full_weights, reference_caches):
                if residual is None:
                    residual = hidden
                    hidden = Norm()(hidden)
                else:
                    hidden, residual = Norm()(hidden, residual)
                q, k, v, o, gate, up, down = weights
                qv, kv, vv = prepare((q, k, v), positions, hidden)
                hidden = F.linear(CacheAttention(8, 4, cache)(qv, kv, vv, batch), o)
                hidden, residual = Norm()(hidden, residual)
                hidden = F.linear(
                    F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down
                )
            hidden, _ = Norm()(hidden, residual)
            torch.testing.assert_close(
                actual, F.linear(hidden, lm_head), atol=2e-5, rtol=2e-5
            )
            shard = layout.full_tp_shard_rank(rank)
            for local, reference in zip(local_caches, reference_caches):
                assert local.keys() == reference.keys()
                for token in local:
                    for value, full in zip(local[token], reference[token]):
                        torch.testing.assert_close(
                            value, full.chunk(p)[shard], atol=2e-5, rtol=2e-5
                        )
        backend.fail = True
        with pytest.raises(RuntimeError, match="backend failed"):
            native_forward(input_ids, positions, batch)
        assert not in_base_tp_scope
        backend.fail = False
        torch.testing.assert_close(native_forward(input_ids, positions, batch), actual)
    finally:
        for group in reversed(groups):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("backend_name", ["flashattention_backend"])
@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize("fa_version", [3, 4])
def test_sp_decode_prefill_and_kv(p, backend_name, fa_version, tmp_path):
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(
        _worker,
        args=(p, backend_name, fa_version, (tmp_path / "rendezvous").as_uri()),
        nprocs=p,
        join=True,
    )


def test_local_padding_does_not_mutate_inputs():
    tensor = torch.arange(5)
    view = UlyssesTokenView(5, 2, 1)
    assert view.slice(tensor).tolist() == [3, 4, 0]
    assert tensor.tolist() == list(range(5))


def test_batch_view_is_idempotent_and_preserves_global_fields():
    strategy = UlyssesParallelStrategy(
        SimpleNamespace(layout=UlyssesRankLayout(4, 2), sp_rank=1)
    )
    batch = SimpleNamespace(
        input_ids=torch.arange(5),
        positions=torch.arange(5) + 10,
        out_cache_loc=torch.arange(5) + 20,
        seq_lens=torch.tensor([3, 2]),
    )
    local = strategy.prepare_batch(batch)
    assert local.input_ids.tolist() == [3, 4, 0]
    assert local.positions.tolist() == [13, 14, 0]
    assert local.out_cache_loc.tolist() == [23, 24, 0]
    assert local.seq_lens is batch.seq_lens
    assert get_global_sp_batch(local).out_cache_loc is batch.out_cache_loc
    assert strategy.prepare_batch(local) is local
    assert not hasattr(batch, "sp_metadata")
    assert batch.input_ids.tolist() == list(range(5))
    # Eager uses dataclasses.replace, so metadata must survive copied views.
    import copy

    assert get_global_sp_batch(copy.copy(local)).positions is batch.positions


def test_empty_batch_skips_gather():
    strategy = UlyssesParallelStrategy(
        SimpleNamespace(layout=UlyssesRankLayout(2, 2), sp_rank=1)
    )
    batch = SimpleNamespace(
        input_ids=torch.empty(0, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
    )
    local = strategy.prepare_batch(batch)
    ids, hidden, full = strategy.gather_logits_inputs(torch.empty(0, 16), local)
    assert ids.numel() == 0 and hidden.shape == (0, 16)
    assert full.input_ids is batch.input_ids


def test_sp_strategy_resolution_is_independent_and_lazy(monkeypatch):
    from sglang.srt import runtime_context
    from sglang.srt.layers import sp_strategy

    config = SimpleNamespace(ulysses_sp_size=2)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: config)
    monkeypatch.setattr(sp_strategy, "_STRATEGY", None)
    strategy = sp_strategy.get_sp_strategy()
    assert strategy.sp_size == 2
    assert strategy._topology is None
    assert sp_strategy.get_sp_strategy() is strategy
    config.ulysses_sp_size = 1
    assert sp_strategy.get_sp_strategy() is None
    config.ulysses_sp_size = 4
    assert sp_strategy.get_sp_strategy().sp_size == 4


@pytest.mark.parametrize("backend_name", ["flashattention_backend"])
def test_concrete_backend_without_sp_preserves_dispatch(backend_name):
    Backend = load_backend(backend_name)
    marker = object()

    class ReferenceBackend(Backend):
        def _forward_extend(self, *args, **kwargs):
            assert kwargs["save_kv_cache"] is False
            assert kwargs["sinks"] is marker
            assert all(
                value is None
                for key, value in kwargs.items()
                if key not in ("save_kv_cache", "sinks")
            )
            return marker

        _forward_decode = _forward_extend

    backend = ReferenceBackend()
    for decode in (False, True):
        mode = SimpleNamespace(
            is_idle=lambda: False, is_decode=lambda: decode, is_mixed=lambda: False
        )
        assert (
            backend.forward(
                None,
                None,
                None,
                None,
                SimpleNamespace(forward_mode=mode),
                save_kv_cache=False,
                sinks=marker,
            )
            is marker
        )
