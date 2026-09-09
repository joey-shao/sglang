# SPDX-License-Identifier: Apache-2.0
"""CPU tests for Ulysses layout, subgroup ownership, and configuration guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.distributed.ulysses_parallel import (
    UlyssesRankLayout,
    create_ulysses_topology,
    validate_ulysses_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.mark.parametrize("p,s", [(2, 2), (4, 2), (6, 2), (8, 4), (8, 2)])
def test_head_ownership_and_gather_order(p, s):
    layout = UlyssesRankLayout(p, s)
    # Each projection shard has S contiguous head chunks. The SP exchange
    # chooses chunk sp_rank: it must match the full TP model's head shard.
    heads_per_rank = 3
    for physical in range(p):
        sp_rank, tp_rank = layout.coordinates(physical)
        projection_heads = list(
            range(tp_rank * s * heads_per_rank, (tp_rank + 1) * s * heads_per_rank)
        )
        received_heads = projection_heads[
            sp_rank * heads_per_rank : (sp_rank + 1) * heads_per_rank
        ]
        shard = layout.full_tp_shard_rank(physical)
        assert received_heads == list(
            range(shard * heads_per_rank, (shard + 1) * heads_per_rank)
        )
    gathered = layout.physical_to_shard
    assert [gathered[r] for r in layout.shard_to_physical] == list(range(p))


def test_tp4_sp2_groups():
    layout = UlyssesRankLayout(4, 2)
    assert layout.base_tp_ranks == [[0, 1], [2, 3]]
    assert layout.sp_ranks == [[0, 2], [1, 3]]
    assert layout.physical_to_shard == (0, 2, 1, 3)


@pytest.mark.parametrize("rank", range(4))
def test_create_groups_preserves_full_tp(rank):
    full = SimpleNamespace(ranks=list(range(4)), rank_in_group=rank, world_size=4)
    created = {}

    def factory(groups, **kwargs):
        ranks = next(g for g in groups if rank in g)
        group = SimpleNamespace(
            ranks=ranks, rank_in_group=ranks.index(rank), world_size=len(ranks)
        )
        created[kwargs["group_name"]] = group
        assert kwargs["use_custom_allreduce"] is False
        return group

    topology = create_ulysses_topology(
        UlyssesRankLayout(4, 2), full, factory, local_rank=rank, backend="gloo"
    )
    assert topology.full_tp_group is full
    assert full.world_size == 4
    assert full.rank_in_group == rank
    assert topology.base_tp_shard_rank == topology.base_tp_group.rank_in_group
    assert topology.sp_rank == topology.sp_group.rank_in_group
    assert topology.full_tp_shard_rank == (0, 2, 1, 3)[rank]


def test_partial_group_creation_cleans_up():
    full = SimpleNamespace(ranks=list(range(4)))
    base = Mock()
    factory = Mock(side_effect=[base, RuntimeError("group creation failed")])
    with pytest.raises(RuntimeError, match="group creation failed"):
        create_ulysses_topology(
            UlyssesRankLayout(4, 2), full, factory, local_rank=0, backend="gloo"
        )
    base.destroy.assert_called_once_with()


@pytest.mark.parametrize("p,s", [(0, 1), (4, 0), (4, 3), (2, 4), (-2, 2)])
def test_invalid_layout(p, s):
    with pytest.raises(ValueError):
        UlyssesRankLayout(p, s)


def test_head_validation():
    layout = UlyssesRankLayout(4, 2)
    layout.validate_heads(32, 8)
    for q, kv in [(32, 2), (30, 8), (0, 8), (32, 0), (12, 8)]:
        with pytest.raises(ValueError):
            layout.validate_heads(q, kv)
    for rank in (-1, 4):
        with pytest.raises(ValueError):
            layout.coordinates(rank)


def config(**updates):
    values = dict(
        tp_size=4,
        ulysses_sp_size=2,
        nnodes=1,
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        ep_size=1,
        moe_dp_size=1,
        enable_dp_attention=False,
        enable_pdmux=False,
        ep_join_mode=None,
    )
    return SimpleNamespace(**(values | updates))


def test_config_defaults_and_topology_only():
    validate_ulysses_config(config())
    validate_ulysses_config(config(ulysses_sp_size=1))


@pytest.mark.parametrize(
    "updates",
    [
        dict(ulysses_sp_size=3),
        dict(nnodes=2),
        dict(pp_size=2),
        dict(dp_size=2),
        dict(attn_cp_size=2),
        dict(dcp_size=2),
        dict(ep_size=2),
        dict(moe_dp_size=2),
        dict(enable_dp_attention=True),
        dict(enable_pdmux=True),
        dict(ep_join_mode="scale"),
    ],
)
def test_invalid_config(updates):
    with pytest.raises(ValueError):
        validate_ulysses_config(config(**updates))


@pytest.mark.parametrize(
    "updates,conflicting_flag",
    [
        (dict(attn_cp_size=2), "--attn-cp-size"),
        (dict(dcp_size=2), "--dcp-size"),
        (dict(enable_dp_attention=True), "--enable-dp-attention"),
        (dict(enable_dp_attention=True, dp_size=2), "--enable-dp-attention"),
    ],
)
def test_ulysses_parallel_conflicts(updates, conflicting_flag):
    with pytest.raises(ValueError, match=conflicting_flag):
        validate_ulysses_config(config(**updates))
    # These restrictions apply only when Ulysses is enabled.
    validate_ulysses_config(config(ulysses_sp_size=1, **updates))


def test_cli_metadata():
    import argparse

    from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
    from sglang.srt.arg_groups.fields.parallel import Parallel

    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, Parallel)
    defaults = parser.parse_args([])
    assert defaults.ulysses_sp_size == 1
    args = parser.parse_args(
        [
            "--tp-size",
            "4",
            "--ulysses-sp-size",
            "2",
        ]
    )
    assert args.tp_size == 4
    assert args.ulysses_sp_size == 2


def _gloo_worker(rank, p, s, rendezvous):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=p,
        timeout=timedelta(seconds=30),
    )
    groups = []

    def factory(rank_lists, **kwargs):
        selected = None
        for ranks in rank_lists:
            handle = dist.new_group(
                ranks, backend="gloo", timeout=timedelta(seconds=30)
            )
            if rank in ranks:
                groups.append(handle)
                selected = SimpleNamespace(
                    ranks=ranks,
                    rank_in_group=ranks.index(rank),
                    world_size=len(ranks),
                    device_group=handle,
                )
        return selected

    try:
        layout = UlyssesRankLayout(p, s)
        full = SimpleNamespace(ranks=list(range(p)), rank_in_group=rank, world_size=p)
        topology = create_ulysses_topology(
            layout, full, factory, local_rank=rank, backend="gloo"
        )
        value = torch.tensor([rank], dtype=torch.int64)
        dist.all_reduce(value, group=topology.base_tp_group.device_group)
        assert value.item() == sum(topology.base_tp_group.ranks)

        # Each source sends destination-tagged chunks. Reception must follow
        # ascending source order within SP, which restores global token order.
        sent = torch.tensor([rank * 100 + dest for dest in range(s)])
        received = torch.empty_like(sent)
        dist.all_to_all_single(received, sent, group=topology.sp_group.device_group)
        assert received.tolist() == [
            src * 100 + topology.sp_rank for src in topology.sp_group.ranks
        ]

        shard = torch.tensor([topology.full_tp_shard_rank])
        gathered = [torch.empty_like(shard) for _ in range(p)]
        dist.all_gather(gathered, shard)
        assert [gathered[i].item() for i in layout.shard_to_physical] == list(range(p))
    finally:
        for group in reversed(groups):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("p,s", [(4, 2), (6, 2)])
def test_gloo_collective_order(p, s, tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    mp.spawn(
        _gloo_worker,
        args=(p, s, (tmp_path / "rendezvous").as_uri()),
        nprocs=p,
        join=True,
    )
