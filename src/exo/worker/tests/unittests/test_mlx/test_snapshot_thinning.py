# pyright: reportPrivateUsage=false
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    prefix_entry_nbytes,
    thin_snapshots,
)


def _positions(snapshots: list[CacheSnapshot] | None) -> list[int]:
    return [s.token_count for s in snapshots or []]


def _snapshots(*positions: int) -> list[CacheSnapshot]:
    return [CacheSnapshot(states=[], token_count=p) for p in positions]


def test_thin_keeps_two_latest_and_one_per_power_of_two_range() -> None:
    # One 34K-token prefill: a snapshot per 4096-token chunk and one at the end.
    chunks = [*range(4096, 34055, 4096), 34053]
    kept = thin_snapshots(_snapshots(*chunks))
    assert _positions(kept) == [4096, 12288, 28672, 32768, 34053]


def test_thinning_during_prefill_matches_thinning_at_the_end() -> None:
    # 100K-token prefill: thinned after every chunk, as prefill() does.
    chunks = [*range(4096, 105884, 4096), 105882]
    running: list[CacheSnapshot] = []
    for snap in _snapshots(*chunks):
        running = thin_snapshots([*running, snap])
        assert sum(_positions(running)) <= 5 * running[-1].token_count
    assert _positions(running) == _positions(thin_snapshots(_snapshots(*chunks)))


def test_thin_keeps_short_lists() -> None:
    assert _positions(thin_snapshots(_snapshots(100, 200))) == [100, 200]
    assert thin_snapshots([]) == []


def test_agent_session_snapshots_stay_bounded() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    cache = [ArraysCache(1)]
    with patch(
        "exo.worker.engines.mlx.cache.get_memory_used_percentage", return_value=0.0
    ):
        kv_prefix_cache.add_kv_cache(
            mx.zeros((34055,), dtype=mx.int32),
            cache,
            _snapshots(*range(4096, 34055, 4096), 34053),
        )
    length = 34055
    for _ in range(200):
        restore_pos = length
        length += 1700
        kv_prefix_cache.update_kv_cache(
            0,
            mx.zeros((length,), dtype=mx.int32),
            cache,
            _snapshots(length - 3, length - 2),
            restore_pos=restore_pos,
        )
    positions = _positions(kv_prefix_cache._snapshots[0])
    assert positions[-2:] == [length - 3, length - 2]
    # Memory grows with position: stays within 5x the latest snapshot.
    assert sum(positions) <= 5 * positions[-1]
    assert len(positions) <= 12


def _kv_entry(megabytes: int) -> list[KVCache]:
    entry = KVCache()
    keys = mx.zeros((1, 1, megabytes * 256, 1024), dtype=mx.float16)
    entry.update_and_fetch(keys, keys)  # pyright: ignore[reportUnknownMemberType]
    return [entry]


def test_stored_entries_stay_within_byte_limit() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    with (
        patch(
            "exo.worker.engines.mlx.cache.get_memory_used_percentage",
            return_value=0.0,
        ),
        patch(
            "exo.worker.engines.mlx.cache.PREFIX_CACHE_BYTE_LIMIT",
            Memory.from_mb(10),
        ),
    ):
        for i in range(6):
            kv_prefix_cache.add_kv_cache(
                mx.full((2000,), i, dtype=mx.int32), _kv_entry(2)
            )
            stored = sum(
                prefix_entry_nbytes(c, s)
                for c, s in zip(
                    kv_prefix_cache.caches, kv_prefix_cache._snapshots, strict=True
                )
            )
            # Eviction runs before an add, so one new entry may exceed the limit.
            assert stored <= Memory.from_mb(10).in_bytes + 4 * 2**20 + 2**20
        # The newest entries are kept.
        assert [int(p[0].item()) for p in kv_prefix_cache.prompts][-1] == 5

        # An entry larger than the limit is still kept when it is the only one.
        kv_prefix_cache.clear()
        kv_prefix_cache.add_kv_cache(mx.full((2000,), 9, dtype=mx.int32), _kv_entry(20))
        kv_prefix_cache.update_kv_cache(
            0,
            mx.full((2001,), 9, dtype=mx.int32),
            _kv_entry(20),
            None,
            restore_pos=2000,
        )
        assert len(kv_prefix_cache.caches) == 1


def test_only_a_continuation_replaces_an_entry() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    with patch(
        "exo.worker.engines.mlx.cache.get_memory_used_percentage", return_value=0.0
    ):
        kv_prefix_cache.add_kv_cache(mx.zeros((4437,), dtype=mx.int32), _kv_entry(1))
    # The same session's next turn restores up to two tokens before the end.
    assert kv_prefix_cache.continues_entry(0, 4435)
    # Another session that only shares the 4096-token system prompt.
    assert not kv_prefix_cache.continues_entry(0, 4096)


def test_restores_from_the_entry_that_restores_most() -> None:
    kv_prefix_cache = KVPrefixCache(None)
    session = mx.arange(6000, dtype=mx.int32)
    with patch(
        "exo.worker.engines.mlx.cache.get_memory_used_percentage", return_value=0.0
    ):
        # An earlier run shares 5800 tokens but only has a snapshot at 4096
        # before them; this session's own entry ends at 5500.
        stale = mx.concatenate([session[:5800], mx.full((2200,), -1, mx.int32)])
        kv_prefix_cache.add_kv_cache(stale, [ArraysCache(1)], _snapshots(4096, 7999))
        kv_prefix_cache.add_kv_cache(
            session[:5500], [ArraysCache(1)], _snapshots(4096, 5498)
        )
    query = session
    with patch("exo.worker.engines.mlx.cache.make_kv_cache", return_value=[]):
        _, remaining, index, _ = kv_prefix_cache.get_kv_cache(None, query)  # pyright: ignore[reportArgumentType]
    assert index == 1
    assert len(remaining) == len(query) - 5498


def _three_entries() -> tuple[KVPrefixCache, float]:
    kv_prefix_cache = KVPrefixCache(None)
    with patch(
        "exo.worker.engines.mlx.cache.get_memory_used_percentage", return_value=0.0
    ):
        for i in range(3):
            kv_prefix_cache.add_kv_cache(
                mx.full((1000,), i, dtype=mx.int32), _kv_entry(1)
            )
    return kv_prefix_cache, prefix_entry_nbytes(kv_prefix_cache.caches[0], None) / 1000


def test_a_request_that_fits_is_admitted_without_evicting() -> None:
    kv_prefix_cache, per_token = _three_entries()
    # A follow-up of entry 2: 1000 cached + 500 new tokens.
    prompt = mx.concatenate(
        [mx.full((1000,), 2, mx.int32), mx.full((500,), 9, mx.int32)]
    )
    with patch(
        "exo.worker.engines.mlx.cache._mlx_headroom",
        return_value=(1500 + 500) * per_token + 1,
    ):
        assert kv_prefix_cache.admits(prompt, others_active=True)
    assert len(kv_prefix_cache.caches) == 3


def test_entries_are_evicted_before_a_request_waits() -> None:
    kv_prefix_cache, per_token = _three_entries()
    prompt = mx.full((5000,), 7, mx.int32)  # nothing cached
    needed = 2 * 5000 * per_token
    # Each eviction frees one entry's worth; two of them are enough.
    free = [needed - 2 * 1000 * per_token]
    original_evict = kv_prefix_cache._evict_lru

    def evict() -> int:
        free[0] += 1000 * per_token
        return original_evict()

    with (
        patch(
            "exo.worker.engines.mlx.cache._mlx_headroom",
            side_effect=lambda: free[0] + 1,
        ),
        patch.object(kv_prefix_cache, "_evict_lru", side_effect=evict),
    ):
        assert kv_prefix_cache.admits(prompt, others_active=True)
    assert len(kv_prefix_cache.caches) == 1


def test_a_request_that_cannot_fit_waits_only_while_others_run() -> None:
    kv_prefix_cache, _ = _three_entries()
    prompt = mx.full((5000,), 7, mx.int32)
    with patch("exo.worker.engines.mlx.cache._mlx_headroom", return_value=0.0):
        assert not kv_prefix_cache.admits(prompt, others_active=True)
        assert kv_prefix_cache.caches == []
        assert kv_prefix_cache.admits(prompt, others_active=False)
