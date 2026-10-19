import gc
import os
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in: when less than
# 4GB is left. A large model sharded over a cluster leaves each node above a
# fixed fraction such as 80% at all times (DeepSeek V4 Flash on 4x 64GB uses
# ~87% with nothing cached), so every entry was evicted as soon as it was
# stored and two agent sessions re-prefilled their whole context on every turn.
# PREFIX_CACHE_BYTE_LIMIT bounds what the entries themselves hold.
def _default_memory_threshold() -> float:
    total = Memory.from_bytes(psutil.virtual_memory().total)
    return min(0.94, 1 - Memory.from_gb(4).in_bytes / total.in_bytes)


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)

# Upper bound on the memory held by stored prefix-cache entries. On macOS
# psutil counts reclaimable file cache as available, so the percentage above
# can read 64% while the GPU is short of memory, and entries from every agent
# session piled up until prefill stalled for minutes. 3GB holds two 60K-token
# DeepSeek V4 sessions; with 4GB, decoding a 240K-token prompt next to them
# left 2GB free on a 64GB node.
PREFIX_CACHE_BYTE_LIMIT = Memory.from_gb(3)

# Free memory a request must leave to be let in next to others, after its
# KV cache and its prefill (which holds about one more KV cache's worth of
# its uncached tokens at its peak).
PREFILL_RESERVE = Memory.from_gb(3)


def _cache_entry_nbytes(c: object) -> int:
    try:
        return int(c.nbytes)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    except (AttributeError, NotImplementedError):
        return 0


def prefix_entry_nbytes(
    cache: KVCacheType, snapshots: "list[CacheSnapshot] | None"
) -> int:
    total = sum(_cache_entry_nbytes(c) for c in cache)
    for snap in snapshots or []:
        total += sum(_cache_entry_nbytes(s) for s in snap.states if s is not None)
    return total


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        # numpy has no bfloat16; move its bits as uint16 (no float32 copy).
        return mx.array(np.array(a.view(mx.uint16))).view(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copies(arrays: list[mx.array | None]) -> list[mx.array | None]:
    """Batched `_detached_copy`: two GPU syncs in total instead of one per array."""
    # numpy has no bfloat16; move its bits as uint16 (no float32 copy).
    staged = [
        None if a is None else a.view(mx.uint16) if a.dtype == mx.bfloat16 else a
        for a in arrays
    ]
    mx.eval([s for s in staged if s is not None])
    copies: list[mx.array | None] = []
    for original, host in zip(arrays, staged, strict=True):
        if original is None or host is None:
            copies.append(None)
            continue
        copy = mx.array(np.array(host))
        copies.append(copy.view(mx.bfloat16) if original.dtype == mx.bfloat16 else copy)
    mx.eval([c for c in copies if c is not None])
    return copies


class PoolReferencingV4Cache(DeepseekV4Cache):
    """A DeepseekV4Cache snapshot that records how many compressed-pool rows
    each branch had instead of copying the pool.

    The pool only grows by appending rows (``update_pool`` concatenates), so
    any later cache for the same prompt starts with exactly these rows; a
    restore slices them from that cache. The pool is the only part that grows
    with the context (2.4 GB per snapshot at 350K tokens on DeepSeek V4
    Flash); everything copied here is bounded by the window and ratio.
    """

    pool_rows: dict[str, int]


def _references_pool(branch: CompressorBranch) -> bool:
    # Batched (per-row length) pools are rebuilt on merge, so copy those.
    return branch.pool is not None and branch.pool_lengths is None


def _copy_v4_caches(
    caches: list[DeepseekV4Cache], reference_pools: bool = False
) -> list[DeepseekV4Cache]:
    sources: list[mx.array | None] = []
    for c in caches:
        local: RotatingKVCache = c.local
        if local.keys is None or local.values is None:
            sources.extend([None, None])
        else:
            n = min(local.max_size, local.keys.shape[2])
            sources.extend([local.keys[..., -n:, :], local.values[..., -n:, :]])
        for branch in c._branches.values():
            sources.extend(
                [
                    branch.buffer_kv,
                    branch.buffer_gate,
                    branch.prev_kv,
                    branch.prev_gate,
                    None
                    if reference_pools and _references_pool(branch)
                    else branch.pool,
                ]
            )
    copies = iter(_detached_copies(sources))

    snaps: list[DeepseekV4Cache] = []
    for c in caches:
        local = c.local
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = next(copies)
        local_snap.values = next(copies)
        local_snap.offset = local.offset
        local_snap._idx = 0 if local_snap.keys is None else local_snap.keys.shape[2]
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size

        snap = (
            PoolReferencingV4Cache.__new__(PoolReferencingV4Cache)
            if reference_pools
            else DeepseekV4Cache.__new__(DeepseekV4Cache)
        )
        if isinstance(snap, PoolReferencingV4Cache):
            snap.pool_rows = {
                key: branch.pool.shape[1]
                for key, branch in c._branches.items()
                if branch.pool is not None and _references_pool(branch)
            }
        snap.local = local_snap
        snap._branches = {}
        for key, branch in c._branches.items():
            out = CompressorBranch.__new__(CompressorBranch)
            out.buffer_kv = next(copies)
            out.buffer_gate = next(copies)
            out.prev_kv = next(copies)
            out.prev_gate = next(copies)
            out.pool = next(copies)
            out.buffer_lengths = deepcopy(branch.buffer_lengths)
            out.pool_lengths = deepcopy(branch.pool_lengths)
            out.buffer_count = deepcopy(branch.buffer_count)
            out._new_pool_lengths = deepcopy(branch._new_pool_lengths)
            snap._branches[key] = out
        snap._pending_lengths = deepcopy(c._pending_lengths)
        snaps.append(snap)
    return snaps


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    return _copy_v4_caches([c])[0]


def _restore_v4_cache(snap: PoolReferencingV4Cache, source: object) -> DeepseekV4Cache:
    """Copy ``snap`` and take its pool rows from ``source``, a later cache of
    the same prompt (the cache being rolled back)."""
    if not isinstance(source, DeepseekV4Cache):
        raise ValueError("pool-referencing snapshot restored without its cache")
    restored = _copy_v4_cache(snap)
    for key, rows in snap.pool_rows.items():
        pool = source._branches[key].pool
        if (
            pool is None
            or source._branches[key].pool_lengths is not None
            or pool.shape[1] < rows
        ):
            raise ValueError(f"cache holds fewer {key} pool rows than snapshot")
        restored._branches[key].pool = pool[:, :rows]
    return restored


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
    source: object = None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    """Copy a snapshot entry to restore it. ``source`` is the cache entry being
    replaced; pool-referencing DeepSeek V4 snapshots take their pool from it."""
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case PoolReferencingV4Cache():
            return _restore_v4_cache(entry, source)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        else:
            # DeepseekV4Cache entries are copied together below.
            states.append(None)
    v4_entries = [(i, c) for i, c in enumerate(cache) if isinstance(c, DeepseekV4Cache)]
    v4_snaps = _copy_v4_caches([c for _, c in v4_entries], reference_pools=True)
    for (i, _), snap in zip(v4_entries, v4_snaps, strict=True):
        states[i] = snap
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def thin_snapshots(snapshots: list[CacheSnapshot]) -> list[CacheSnapshot]:
    """Bound the snapshots kept for one prompt.

    Every prefill takes a snapshot per prefill chunk and one at the end, so a
    100K-token prefill holds ~25 growing snapshots, and a long agent session
    that keeps extending one prefix cache entry would accumulate them without
    limit, until the GPU runs out of memory. Keep the two latest (resuming or
    re-asking the newest turn) and, below them, the latest snapshot in each
    power-of-two range of positions. A snapshot's size grows with its
    position, so the kept set stays within five times the latest one,
    and thinning as snapshots arrive keeps one in every range seen.
    """
    by_position = sorted(snapshots, key=lambda s: s.token_count)
    latest = by_position[-2:]
    per_range: dict[int, CacheSnapshot] = {}
    for snap in by_position[:-2]:
        per_range[snap.token_count.bit_length()] = snap
    return [*per_range.values(), *latest]


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    return isinstance(c, DeepseekV4Cache)


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group
        self._bytes_per_token: float | None = None

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(thin_snapshots(ssm_snapshots or []) or None)
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        logger.info(f"KV cache added: {len(prompt_tokens)} tokens")
        # Also after adding, so the stored entries stay within
        # PREFIX_CACHE_BYTE_LIMIT instead of exceeding it by the new entry.
        self._evict_if_needed()

    def continues_entry(self, index: int, prefix_hit_length: int) -> bool:
        """Whether a prompt that restored ``prefix_hit_length`` tokens from
        entry ``index`` continues it, rather than branching off part of it.

        Agent sessions share a long system prompt, so a hit can cover most of
        a short prompt while belonging to another session. Replacing that
        session's entry made two concurrent sessions evict each other and
        re-prefill their whole history on every turn.
        """
        # Prefill trims the last two prompt tokens, so a continuation
        # restores from two tokens before the end of the stored prompt.
        return prefix_hit_length >= len(self.prompts[index]) - 2

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = thin_snapshots(merged) or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        logger.info(f"KV cache updated (index {index}): {len(prompt_tokens)} tokens")
        self._evict_if_needed()

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def _restore_point(
        self, index: int, match_length: int, is_exact: bool, max_length: int
    ) -> tuple[int, "CacheSnapshot | None"]:
        """Token position entry ``index`` can be restored to for a prompt that
        shares ``match_length`` tokens with it, and the snapshot to use."""
        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to match_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        if has_non_kv_caches(self.caches[index]):
            target = match_length
        else:
            desired = (max_length - 1) if is_exact else match_length
            target = min(cache_length(self.caches[index]), desired)
        return self._get_snapshot(index, target)

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        # Pick the entry that restores the most tokens, not the longest raw
        # match: a stale longer entry (an earlier run of the same agent task)
        # can share more tokens but only restore to a snapshot far back,
        # while this session's own entry restores to its end.
        best_index: int | None = None
        best_restore: tuple[int, CacheSnapshot | None] = (0, None)
        is_exact = False
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length <= 0:
                continue
            exact = length >= max_length - 1
            restore = self._restore_point(i, length, exact, max_length)
            if restore[0] > best_restore[0] or best_index is None:
                best_index, best_restore, is_exact = i, restore, exact

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None, False

        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        restore_pos, restore_snap = best_restore

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None, False

        prompt_cache = deepcopy(self.caches[best_index])
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            try:
                trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            except ValueError:
                logger.warning(
                    "Prefix cache snapshot unusable; prefilling from scratch"
                )
                return make_kv_cache(model), prompt_tokens, None, False
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_if_needed(self):
        """Evict least recently used entries while memory usage is high."""
        if len(self.caches) == 0:
            return

        evicted_any = False
        # Evict LRU entries until below threshold
        while len(self.caches) > 0 and (
            self.get_memory_used_percentage() > _MEMORY_THRESHOLD
            # Always keep the most recently used entry.
            or (len(self.caches) > 1 and self._stored_bytes_fraction() > 1.0)
        ):
            self._evict_lru()
            evicted_any = True

        if evicted_any:
            gc.collect()
            mx.clear_cache()

    def admits(self, prompt_tokens: mx.array, others_active: bool) -> bool:
        """Whether a request fits next to the ones already decoding.

        Evicts LRU entries until every rank has room for the request's KV
        cache and its prefill plus PREFILL_RESERVE. A request that still does
        not fit waits while others are active; alone, it is let in anyway.

        Six agent sessions, three of them with a 240K-token prompt, held 8GB
        of KV cache next to 2.8GB of stored entries, and one rank was left
        with less than nothing free.
        """
        stored_tokens = sum(len(prompt) for prompt in self.prompts)
        if stored_tokens > 0:
            self._bytes_per_token = (
                sum(
                    prefix_entry_nbytes(cache, snapshots)
                    for cache, snapshots in zip(
                        self.caches, self._snapshots, strict=True
                    )
                )
                / stored_tokens
            )
        if self._bytes_per_token is None:
            return True
        total = len(prompt_tokens)
        cached = max(
            (get_prefix_length(prompt_tokens, prompt) for prompt in self.prompts),
            default=0,
        )
        needed = (
            total + total - cached
        ) * self._bytes_per_token + PREFILL_RESERVE.in_bytes

        def fits() -> bool:
            return (
                _min_across_ranks(float(get_available_memory().in_bytes), self._group)
                >= needed
            )

        evicted_any = False
        fit = fits()
        while not fit and self.caches:
            self._evict_lru()
            evicted_any = True
            fit = fits()
        if evicted_any:
            gc.collect()
            mx.clear_cache()
        return fit or not others_active

    def _evict_lru(self) -> int:
        lru_index = self._last_used.index(min(self._last_used))
        evicted_tokens = len(self.prompts[lru_index])
        self.prompts.pop(lru_index)
        self.caches.pop(lru_index)
        self._snapshots.pop(lru_index)
        self._media_regions.pop(lru_index)
        self._last_used.pop(lru_index)
        self.prefill_tps.pop(lru_index)
        logger.info(
            f"KV cache evicted LRU entry ({evicted_tokens} tokens) due to memory usage"
        )
        return lru_index

    def _stored_bytes_fraction(self) -> float:
        """Stored entry bytes over PREFIX_CACHE_BYTE_LIMIT, the max across ranks
        so every rank evicts the same entries."""
        stored = sum(
            prefix_entry_nbytes(cache, snapshots)
            for cache, snapshots in zip(self.caches, self._snapshots, strict=True)
        )
        return _max_across_ranks(stored / PREFIX_CACHE_BYTE_LIMIT.in_bytes, self._group)

    def get_memory_used_percentage(self) -> float:
        return _max_across_ranks(get_memory_used_percentage(), self._group)


def _min_across_ranks(value: float, group: mx.distributed.Group | None) -> float:
    return -_max_across_ranks(-value, group)


def _max_across_ranks(value: float, group: mx.distributed.Group | None) -> float:
    """The largest ``value`` of any rank, so every rank takes the same branch.

    The gathered values are read on the CPU. A GPU op on them (``mx.max``)
    waits for the slowest rank in a Metal fence that gives up after ~15s and
    then computes on stale data: on 4x M5 Pro, a rank that saved its prefix
    cache 30s later than the others made two ranks evict entries the other
    two kept, and the ranks' collectives went out of step until jaccl
    segfaulted.
    """
    if group is None:
        return value
    gathered = mx.distributed.all_gather(
        mx.array([value], dtype=mx.float32), group=group
    )
    return max(cast(list[float], gathered.tolist()))


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        if is_non_trimmable_cache_entry(c):
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i], c)
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, DeepseekV4Cache):
                # trim() is a no-op for DeepseekV4Cache (its compressor pool
                # can't be rolled back), so without a snapshot start empty.
                cache[i] = DeepseekV4Cache(c.local.max_size)  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
