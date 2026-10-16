# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportCallIssue=false
"""DeepSeek V4 snapshots reference the compressed pool instead of copying it.

Uses the tiny randomly initialized DeepSeek V4; no model download required.
"""

from collections.abc import Sequence

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache, Model

from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    PoolReferencingV4Cache,
    _copy_v4_caches,
    copy_snapshot_entry,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.tests.tiny_deepseek_v4 import VOCAB_SIZE, tiny_model

TOKENS = np.random.default_rng(21).integers(0, VOCAB_SIZE, (1, 400))


def _run(model: Model, cache: Sequence[object], start: int, end: int) -> mx.array:
    logits = model(mx.array(TOKENS[:, start:end]), cache=cache)
    mx.eval(logits)
    return logits


def _v4_arrays(c: DeepseekV4Cache) -> list[mx.array | None]:
    arrays: list[mx.array | None] = [c.local.keys, c.local.values]
    for branch in c._branches.values():
        arrays += [
            branch.buffer_kv,
            branch.buffer_gate,
            branch.prev_kv,
            branch.prev_gate,
            branch.pool,
        ]
    return arrays


def _assert_same(a: DeepseekV4Cache, b: DeepseekV4Cache) -> None:
    for x, y in zip(_v4_arrays(a), _v4_arrays(b), strict=True):
        assert (x is None) == (y is None)
        if x is not None and y is not None:
            assert x.shape == y.shape
            assert mx.array_equal(x, y).item()


@pytest.mark.parametrize("snapshot_at", [96, 250])
def test_restore_from_later_cache_matches_full_copy(snapshot_at: int) -> None:
    model = tiny_model()
    cache = model.make_cache()
    _run(model, cache, 0, snapshot_at)
    snapshot = snapshot_ssm_states(cache)
    full_copies = _copy_v4_caches([c for c in cache if isinstance(c, DeepseekV4Cache)])
    pool_rows = [
        sum(b.pool.shape[1] for b in c._branches.values() if b.pool is not None)
        for c in cache
        if isinstance(c, DeepseekV4Cache)
    ]
    assert sum(pool_rows) > 0
    for state in snapshot.states:
        if isinstance(state, PoolReferencingV4Cache):
            assert all(b.pool is None for b in state._branches.values())

    _run(model, cache, snapshot_at, 400)

    restored = [
        copy_snapshot_entry(state, c)
        for state, c in zip(snapshot.states, cache, strict=True)
    ]
    v4_restored = [c for c in restored if isinstance(c, DeepseekV4Cache)]
    for got, expected in zip(v4_restored, full_copies, strict=True):
        _assert_same(got, expected)

    # Continuing from the restored cache matches continuing from a full copy.
    reference = model.make_cache()
    _run(model, reference, 0, snapshot_at)
    expected_logits = _run(model, reference, snapshot_at, snapshot_at + 30)
    got_logits = _run(model, restored, snapshot_at, snapshot_at + 30)
    assert mx.array_equal(expected_logits, got_logits).item()


def test_trim_cache_restores_pool_referencing_snapshot() -> None:
    model = tiny_model()
    cache = model.make_cache()
    _run(model, cache, 0, 200)
    snapshot = snapshot_ssm_states(cache)
    _run(model, cache, 200, 400)
    trim_cache(cache, 200, snapshot)
    reference = model.make_cache()
    _run(model, reference, 0, 200)
    assert mx.array_equal(
        _run(model, reference, 200, 220), _run(model, cache, 200, 220)
    ).item()


def test_restore_rejects_cache_without_the_rows() -> None:
    model = tiny_model()
    cache = model.make_cache()
    _run(model, cache, 0, 300)
    snapshot: CacheSnapshot = snapshot_ssm_states(cache)
    fresh = model.make_cache()
    state = next(
        s
        for s in snapshot.states
        if isinstance(s, PoolReferencingV4Cache) and any(s.pool_rows.values())
    )
    with pytest.raises(ValueError):
        copy_snapshot_entry(state, fresh[snapshot.states.index(state)])
