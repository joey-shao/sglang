# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3/FlashAttention smoke test: pytest test/manual/test_ulysses_qwen3_native.py.

Creates a tiny local checkpoint, then compares TP and fixed SP generation with
chunked prefill and a repeated prefix. Requires 2/4 CUDA GPUs and SGLang deps.
"""

import pytest
import torch


@pytest.mark.parametrize("tp_size", [2, 4])
def test_native_qwen3_fixed_sp(tp_size, tmp_path):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} CUDA GPUs")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from sglang import Engine

    torch.manual_seed(123)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(tmp_path, safe_serialization=True)
    del model
    inputs = [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12]]
    sampling = {"temperature": 0, "max_new_tokens": 6, "ignore_eos": True}

    def run(ulysses):
        engine = Engine(
            model_path=str(tmp_path),
            dtype="bfloat16",
            load_format="safetensors",
            skip_tokenizer_init=True,
            tp_size=tp_size,
            attention_backend="fa3",
            disable_overlap_schedule=True,
            disable_prefill_cuda_graph=True,
            disable_decode_cuda_graph=True,
            max_total_tokens=1024,
            max_running_requests=8,
            page_size=1,
            chunked_prefill_size=4,
            ulysses_sp_size=2 if ulysses else 1,
        )
        try:
            first = engine.generate(input_ids=inputs, sampling_params=sampling)
            # Prefix reuse plus a new suffix after the initial requests finish.
            second = engine.generate(
                input_ids=[inputs[0] + [13, 14]], sampling_params=sampling
            )
            return [
                [item["output_ids"] for item in results] for results in (first, second)
            ]
        finally:
            engine.shutdown()

    assert run(False) == run(True)
