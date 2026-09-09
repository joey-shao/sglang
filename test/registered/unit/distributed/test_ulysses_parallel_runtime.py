# SPDX-License-Identifier: Apache-2.0
"""Exercise actual parallel-state lifecycle with mocked communicator creation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ps = pytest.importorskip("sglang.srt.distributed.parallel_state")
from sglang.srt.runtime_context import get_parallel


@pytest.fixture
def groups(monkeypatch):
    for name in (
        "_TP",
        "_PP",
        "_SELF_PP",
        "_ATTN_TP",
        "_ATTN_CP",
        "_DCP",
        "_MOE_TP",
        "_MOE_DP",
        "_MOE_EP",
        "_ULYSSES_TOPOLOGY",
        "_PDMUX_PREFILL_TP_GROUP",
    ):
        monkeypatch.setattr(ps, name, None)
    monkeypatch.setattr(ps, "_TP_STATE_PATCHED", False)
    monkeypatch.setattr(ps, "_ENABLE_PDMUX_P_TP", False)
    monkeypatch.setattr(ps, "get_global_dwdp_manager", lambda: None)
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(ps.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(ps, "get_world_group", lambda: SimpleNamespace(local_rank=2))
    created = {}

    def factory(rank_lists, local_rank, backend, **kwargs):
        ranks = next(ranks for ranks in rank_lists if 2 in ranks)
        group = SimpleNamespace(
            ranks=ranks,
            world_size=len(ranks),
            rank_in_group=ranks.index(2),
            destroy=Mock(),
        )
        created[kwargs["group_name"]] = group
        return group

    monkeypatch.setattr(ps, "init_model_parallel_group", factory)
    yield created
    ps.destroy_model_parallel()


def test_lifecycle_and_scope_restore(groups):
    ps.initialize_model_parallel(
        tensor_model_parallel_size=4,
        ulysses_sequence_parallel_size=2,
        backend="gloo",
    )
    topology = get_parallel().ulysses_topology
    full = ps.get_tp_group()
    assert full.ranks == [0, 1, 2, 3]
    assert topology.full_tp_group is full
    assert topology.base_tp_group.ranks == [2, 3]
    assert topology.sp_group.ranks == [0, 2]
    assert topology.full_tp_shard_rank == 1
    with get_parallel().override(tp_size=4):
        with pytest.raises(RuntimeError, match="execution failed"):
            with topology.base_tp_scope():
                assert ps.get_tp_group() is topology.base_tp_group
                assert get_parallel().tp_group is topology.base_tp_group
                assert get_parallel().tp_size == 2
                assert get_parallel().tp_rank == 0
                assert topology.full_tp_group is full
                raise RuntimeError("execution failed")
        assert ps.get_tp_group() is full
        assert get_parallel().tp_size == 4
        assert get_parallel().tp_rank == 2
        assert not ps._TP_STATE_PATCHED
    ps.ensure_model_parallel_initialized(
        4, 1, 1, backend="gloo", ulysses_sequence_parallel_size=2
    )
    with pytest.raises(ValueError, match="different SP size"):
        ps.ensure_model_parallel_initialized(4, 1, 1, backend="gloo")
    ps.destroy_model_parallel()
    groups["ulysses_sp"].destroy.assert_called_once_with()
    groups["ulysses_base_tp"].destroy.assert_called_once_with()
    with pytest.raises(RuntimeError, match="not initialized"):
        ps.get_ulysses_topology()
    # Teardown followed by reinitialization must not retain Ulysses state.
    ps.initialize_model_parallel(tensor_model_parallel_size=4, backend="gloo")
    assert ps._ULYSSES_TOPOLOGY is None


def test_disabled_creates_no_extra_groups(groups):
    ps.initialize_model_parallel(tensor_model_parallel_size=4, backend="gloo")
    assert "ulysses_sp" not in groups
    assert "ulysses_base_tp" not in groups
    assert ps.get_tp_group().ranks == [0, 1, 2, 3]


def test_model_tp_scope(groups):
    ps.initialize_model_parallel(
        tensor_model_parallel_size=4, ulysses_sequence_parallel_size=2, backend="gloo"
    )
    topology = ps.get_ulysses_topology()
    full = ps.get_tp_group()
    old_attn = ps.get_attn_tp_group()
    with get_parallel().override(tp_size=4):
        with pytest.raises(RuntimeError, match="load failed"):
            with topology.model_tp_scope():
                assert get_parallel().tp_size == 2
                assert get_parallel().tp_rank == 0
                assert get_parallel().attn_tp_size == 2
                assert get_parallel().attn_tp_rank == 0
                assert ps.get_attn_tp_group() is topology.base_tp_group
                raise RuntimeError("load failed")
        assert get_parallel().tp_size == 4
        assert get_parallel().tp_rank == 2
        assert ps.get_attn_tp_group() is old_attn
        assert ps.get_tp_group() is full


@pytest.mark.parametrize(
    "updates",
    [
        dict(ulysses_sequence_parallel_size=3),
        dict(pipeline_model_parallel_size=2),
        dict(attention_data_parallel_size=2),
        dict(recovered_rank=True),
    ],
)
def test_invalid_before_collective_creation(groups, updates):
    kwargs = dict(
        tensor_model_parallel_size=4, ulysses_sequence_parallel_size=2, backend="gloo"
    )
    with pytest.raises(ValueError):
        ps.initialize_model_parallel(**(kwargs | updates))
    assert not groups
