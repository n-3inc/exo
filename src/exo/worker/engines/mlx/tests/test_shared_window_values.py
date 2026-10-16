# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportArgumentType=false
# pyright: reportPrivateUsage=false
"""Sliding-window caches updated with ``update_and_fetch(kv, kv)`` keep their
values as the keys, so no unevaluated chain of value writes builds up.

Uses the tiny randomly initialized DeepSeek V4; no model download required.
"""

import io
from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.engines.mlx.patches import shared_window_values
from exo.worker.engines.mlx.tests.tiny_deepseek_v4 import VOCAB_SIZE, tiny_model


def _graph_lines(array: mx.array) -> int:
    dot = io.StringIO()
    mx.export_to_dot(dot, array)
    return dot.getvalue().count("\n")


def _unpatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other tests may have applied every exo patch already.
    monkeypatch.setattr(
        RotatingKVCache,
        "update_and_fetch",
        shared_window_values._original_rotating_update,
    )
    monkeypatch.setattr(
        BatchRotatingKVCache,
        "update_and_fetch",
        shared_window_values._original_batch_rotating_update,
    )


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RotatingKVCache, "update_and_fetch", shared_window_values._rotating_update
    )
    monkeypatch.setattr(
        BatchRotatingKVCache,
        "update_and_fetch",
        shared_window_values._batch_rotating_update,
    )


def _decode(steps: int) -> tuple[list[mx.array], list[DeepseekV4Cache]]:
    model = tiny_model()
    cache = model.make_cache()
    prompt = np.random.default_rng(5).integers(0, VOCAB_SIZE, (1, 40))
    logits = model(mx.array(prompt), cache=cache)
    all_logits = [logits[:, -1, :]]
    for _ in range(steps):
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        logits = model(token, cache=cache)
        mx.eval(logits)
        all_logits.append(logits[:, -1, :])
    return all_logits, cast(list[DeepseekV4Cache], cache)


def test_values_do_not_chain_over_a_long_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unpatch(monkeypatch)
    _, unpatched = _decode(200)
    _patch(monkeypatch)
    _, patched = _decode(200)
    assert _graph_lines(unpatched[0].local.values) > 1000
    assert all(c.local.values is c.local.keys for c in patched)
    assert max(_graph_lines(c.local.values) for c in patched) < 20


def test_decoding_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _unpatch(monkeypatch)
    expected, _ = _decode(60)
    _patch(monkeypatch)
    got, _ = _decode(60)
    for a, b in zip(expected, got, strict=True):
        assert mx.array_equal(a, b).item()


def _batch_updates(cache: BatchRotatingKVCache) -> list[mx.array]:
    rng = np.random.default_rng(9)
    fetched: list[mx.array] = []
    for length in [5, 1, 1, 3, *([1] * 40)]:
        kv = mx.array(rng.standard_normal((2, 1, length, 8)), dtype=mx.float16)
        keys, values = cache.update_and_fetch(kv, kv)
        assert mx.array_equal(keys, values).item()
        fetched.append(keys)
    mx.eval(fetched)
    return fetched


def test_batch_rotating_cache_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _unpatch(monkeypatch)
    expected = _batch_updates(BatchRotatingKVCache(16, [2, 0]))
    _patch(monkeypatch)
    cache = BatchRotatingKVCache(16, [2, 0])
    got = _batch_updates(cache)
    for a, b in zip(expected, got, strict=True):
        assert mx.array_equal(a, b).item()
    assert cache.values is cache.keys
