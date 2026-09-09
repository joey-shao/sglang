# SPDX-License-Identifier: Apache-2.0
"""Validate and load one Qwen3 model with attention-only Ulysses SP."""

from __future__ import annotations

import copy
from typing import Callable


def load_ulysses_model(*, model_config, load_config, topology, load_one: Callable):
    """Load one base-TP model replicated across the SP dimension."""
    local_config = copy.deepcopy(model_config)
    local_load = copy.deepcopy(load_config)
    local_load.tp_rank = topology.base_tp_shard_rank
    with topology.model_tp_scope():
        return load_one(model_config=local_config, load_config=local_load)
