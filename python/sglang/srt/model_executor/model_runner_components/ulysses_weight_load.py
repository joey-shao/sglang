# SPDX-License-Identifier: Apache-2.0
"""Load one Qwen3 model with fixed Ulysses SP x TP."""

from __future__ import annotations

import copy
from typing import Callable


def load_ulysses_model(*, model_config, load_config, load_one: Callable):
    """Load one base-TP model replicated across the SP dimension."""
    from sglang.srt.distributed.parallel_state import (
        get_ulysses_model_tp_group,
        ulysses_model_tp_scope,
    )

    local_config = copy.deepcopy(model_config)
    local_load = copy.deepcopy(load_config)
    local_load.tp_rank = get_ulysses_model_tp_group().rank_in_group
    with ulysses_model_tp_scope():
        return load_one(model_config=local_config, load_config=local_load)


def load_shift_parallel_models(*, model_config, load_config, load_one: Callable):
    """Load full-TP and Ulysses model replicas for runtime mode switching."""
    from sglang.srt.distributed.parallel_state import get_ulysses_full_tp_group

    tp_config = copy.deepcopy(model_config)
    tp_load = copy.deepcopy(load_config)
    tp_load.tp_rank = get_ulysses_full_tp_group().rank_in_group
    tp_loaded = load_one(model_config=tp_config, load_config=tp_load)
    sp_loaded = load_ulysses_model(
        model_config=model_config,
        load_config=load_config,
        load_one=load_one,
    )
    return tp_loaded, sp_loaded
