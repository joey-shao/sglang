# SPDX-License-Identifier: Apache-2.0
"""Single-model loader contract tests; checkpoint I/O uses real CPU tensors."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed.ulysses_parallel import UlyssesRankLayout
from sglang.srt.model_executor.model_runner_components.ulysses_weight_load import (
    load_ulysses_model,
    validate_ulysses_eager_options,
    validate_ulysses_loading_options,
    validate_ulysses_weight_model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3ForCausalLM"], intermediate_size=24
        ),
        dtype=torch.bfloat16,
        quantization=None,
        get_total_num_attention_heads=lambda: 12,
        get_total_num_kv_heads=lambda: 6,
    )


class RecordingTopology:
    def __init__(self, rank):
        self.layout = UlyssesRankLayout(6, 2)
        self.full_tp_group = SimpleNamespace(rank_in_group=rank)
        self.base_tp_shard_rank = rank % 3
        self.active = None

    @contextmanager
    def model_tp_scope(self):
        assert self.active is None
        self.active = True
        try:
            yield
        finally:
            self.active = None


@pytest.mark.parametrize("rank", range(6))
def test_loads_one_model_under_base_tp(rank, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    torch.save({"q": torch.arange(72).reshape(12, 6).to(torch.bfloat16)}, checkpoint)
    model_config = config()
    load_config = SimpleNamespace(load_format="pt", tp_rank=rank)
    topology = RecordingTopology(rank)
    calls = []

    def loader(*, model_config, load_config):
        assert topology.active
        weight = (
            torch.load(checkpoint, weights_only=True)["q"].chunk(3)[rank % 3].clone()
        )
        calls.append(load_config.tp_rank)
        # A loader mutating its input must not corrupt the caller.
        model_config.hf_config.intermediate_size = -1
        load_config.tp_rank = -1
        return SimpleNamespace(model=SimpleNamespace(q=weight), loader=object())

    loaded = load_ulysses_model(
        model_config=model_config,
        load_config=load_config,
        topology=topology,
        load_one=loader,
    )
    assert calls == [rank % 3]
    expected = torch.arange(72).reshape(12, 6).to(torch.bfloat16).chunk(3)[rank % 3]
    torch.testing.assert_close(loaded.model.q, expected)
    assert model_config.hf_config.intermediate_size == 24
    assert load_config.tp_rank == rank
    assert topology.active is None


def test_load_failure_restores_scope():
    topology = RecordingTopology(2)

    def loader(**kwargs):
        raise RuntimeError("checkpoint read failed")

    with pytest.raises(RuntimeError, match="checkpoint read failed"):
        load_ulysses_model(
            model_config=config(),
            load_config=SimpleNamespace(load_format="pt", tp_rank=2),
            topology=topology,
            load_one=loader,
        )
    assert topology.active is None


@pytest.mark.parametrize(
    "field,value", [("dtype", torch.float16), ("quantization", "fp8")]
)
def test_invalid_model(field, value):
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        validate_ulysses_weight_model(
            cfg, SimpleNamespace(load_format="pt"), RecordingTopology(0)
        )


def test_wrong_architecture_and_loader():
    cfg = config()
    cfg.hf_config.architectures = ["Qwen3MoeForCausalLM"]
    with pytest.raises(ValueError, match="dense only"):
        validate_ulysses_weight_model(
            cfg, SimpleNamespace(load_format="pt"), RecordingTopology(0)
        )
    with pytest.raises(ValueError, match="checkpoint loading"):
        validate_ulysses_weight_model(
            config(), SimpleNamespace(load_format="presharded"), RecordingTopology(0)
        )


def options():
    return SimpleNamespace(
        startup_weight_load_mode="serial",
        weight_cache_mode="off",
        cpu_offload_gb=0,
        offload_group_size=-1,
        enable_memory_saver=False,
        enable_weights_cpu_backup=False,
        enable_lora=False,
        enable_dp_lm_head=False,
        enable_tp_lm_head_all_to_all=False,
        speculative_algorithm=None,
        quantization=None,
        modelopt_quant=None,
        custom_weight_loader=None,
        lora_paths=None,
        load_format="safetensors",
    )


def test_loading_options():
    validate_ulysses_loading_options(options())


@pytest.mark.parametrize(
    "field,value",
    [
        ("startup_weight_load_mode", "overlap"),
        ("weight_cache_mode", "ipc"),
        ("enable_dp_lm_head", True),
        ("enable_tp_lm_head_all_to_all", True),
        ("enable_lora", True),
        ("cpu_offload_gb", 1),
        ("speculative_algorithm", "EAGLE"),
        ("load_format", "sharded_state"),
    ],
)
def test_reject_unsupported_loading_paths(field, value):
    cfg = options()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        validate_ulysses_loading_options(cfg)


def eager_options():
    return SimpleNamespace(
        device="cuda",
        attention_backend="fa3",
        disable_overlap_schedule=True,
        enable_two_batch_overlap=False,
        enable_layernorm_sp=False,
        enable_prefill_cp=False,
        enable_attn_tp_input_scattered=False,
        enable_torch_compile=False,
        enable_return_hidden_states=False,
        rl_on_policy_target=None,
        disaggregation_mode="null",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        kv_cache_dtype="auto",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="disabled"),
            decode=SimpleNamespace(backend="disabled"),
        ),
    )


def test_eager_options():
    validate_ulysses_eager_options(eager_options())


@pytest.mark.parametrize("backend", ["fa3", "fa4"])
def test_flashattention_phase_options(backend):
    cfg = eager_options()
    cfg.attention_backend = backend
    cfg.prefill_attention_backend = backend
    cfg.decode_attention_backend = backend
    validate_ulysses_eager_options(cfg)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_reject_flashinfer_phase_override(phase):
    cfg = eager_options()
    setattr(cfg, f"{phase}_attention_backend", "flashinfer")
    with pytest.raises(ValueError):
        validate_ulysses_eager_options(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cpu"),
        ("attention_backend", "flashinfer"),
        ("disable_overlap_schedule", False),
        ("enable_layernorm_sp", True),
        ("enable_prefill_cp", True),
        ("enable_attn_tp_input_scattered", True),
        ("enable_torch_compile", True),
        ("kv_cache_dtype", "fp8_e4m3"),
    ],
)
def test_reject_unsupported_eager_options(field, value):
    cfg = eager_options()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        validate_ulysses_eager_options(cfg)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_reject_graph_execution(phase):
    cfg = eager_options()
    getattr(cfg.cuda_graph_config, phase).backend = "full"
    with pytest.raises(ValueError, match="requires disabled backend"):
        validate_ulysses_eager_options(cfg)
