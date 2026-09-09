# SPDX-License-Identifier: Apache-2.0
"""Exercise actual worker/batch boundary methods without CUDA imports."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed.ulysses_parallel import UlyssesRankLayout
from sglang.srt.layers import sp_strategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_method(path, class_name, method_name, namespace):
    """Keep production method bodies; replace only imports/GPU construction."""
    path = Path(__file__).resolve().parents[4] / path
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("source", ["schedule", "global", "local"])
@pytest.mark.parametrize("fail", [False, True])
def test_worker_scopes_model_and_restores_global_sampling(monkeypatch, source, fail):
    active = False

    @contextmanager
    def scope():
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    strategy = sp_strategy.UlyssesParallelStrategy(
        SimpleNamespace(
            layout=UlyssesRankLayout(4, 2),
            sp_rank=1,
            model_tp_scope=scope,
        )
    )
    monkeypatch.setattr(sp_strategy, "get_sp_strategy", lambda: strategy)
    prepare = load_method(
        "python/sglang/srt/model_executor/forward_batch_info.py",
        "ForwardBatch",
        "prepare_sp_batch",
        {},
    )

    class Batch(SimpleNamespace):
        prepare_sp_batch = prepare

        def apply_deprecated_skip_attn_backend_init(self, value):
            pass

    original = Batch(
        input_ids=torch.arange(5),
        positions=torch.arange(5) + 10,
        out_cache_loc=torch.arange(5) + 30,
        seq_lens=torch.tensor([3, 2]),
        forward_mode="extend",
        is_prefill_only=False,
        hicache_consumer_index=0,
    )
    sampled = []

    def run(local, **kwargs):
        assert active
        assert local.input_ids.tolist() == [3, 4, 0]
        assert local.positions.tolist() == [13, 14, 0]
        assert local.out_cache_loc.tolist() == [33, 34, 0]
        assert local.prepare_sp_batch() is local
        assert sp_strategy.get_global_sp_batch(local).input_ids is original.input_ids
        if fail:
            raise RuntimeError("model failed")
        return SimpleNamespace(
            logits_output=object(),
            can_run_graph=False,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    def sample(logits, global_batch):
        assert not active
        assert global_batch.input_ids is original.input_ids
        assert global_batch.out_cache_loc is original.out_cache_loc
        sampled.append(global_batch)
        return torch.tensor([7, 8])

    worker = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True),
        model_runner=SimpleNamespace(forward=run, sample=sample),
        is_dllm=lambda: False,
        enable_overlap=False,
        set_hicache_consumer=lambda value: None,
    )
    # The ScheduleBatch branch invokes init_new; it returns a prepared local
    # view, exercising the worker's repeated preparation on the real method.
    namespace = {
        "nullcontext": nullcontext,
        "GenerationBatchResult": SimpleNamespace,
        "ForwardBatch": SimpleNamespace(
            init_new=lambda batch, *args, **kwargs: batch.prepare_sp_batch()
        ),
        "capture_pre_sample_logits": lambda *args: None,
    }
    generate = load_method(
        "python/sglang/srt/managers/tp_worker.py",
        "TpModelWorker",
        "forward_batch_generation",
        namespace,
    )
    batch = original if source == "schedule" else None
    forward_batch = original.prepare_sp_batch() if source == "local" else original
    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            generate(worker, batch=batch, forward_batch=forward_batch)
        assert not sampled
    else:
        result = generate(worker, batch=batch, forward_batch=forward_batch)
        assert result.next_token_ids.tolist() == [7, 8]
        assert len(sampled) == 1
    assert not active
    assert original.input_ids.tolist() == list(range(5))
    assert not hasattr(original, "sp_metadata")


def test_global_view_tracks_execution_mode_without_mutating_source():
    strategy = sp_strategy.UlyssesParallelStrategy(
        SimpleNamespace(
            layout=UlyssesRankLayout(4, 2),
            sp_rank=1,
        )
    )
    source = SimpleNamespace(
        input_ids=torch.arange(3),
        positions=torch.arange(3),
        out_cache_loc=torch.arange(3),
        forward_mode="mixed",
    )
    local = strategy.prepare_batch(source)
    local.forward_mode = "extend"
    assert sp_strategy.get_global_sp_batch(local).forward_mode == "extend"
    assert source.forward_mode == "mixed"
