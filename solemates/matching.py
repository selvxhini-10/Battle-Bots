from __future__ import annotations

from functools import lru_cache

import numpy as np

from .models import PairMatch, SockObservation


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def score_pair(a: SockObservation, b: SockObservation, weights: dict) -> PairMatch:
    color = float(np.clip(_cosine(a.color_hist, b.color_hist), 0.0, 1.0))
    pattern_distance = float(np.linalg.norm(a.pattern - b.pattern))
    pattern = float(np.exp(-2.5 * pattern_distance))
    size = min(a.area_px, b.area_px) / max(a.area_px, b.area_px)
    shape = float(np.exp(-3.0 * np.linalg.norm(a.shape - b.shape)))
    parts = {"color": color, "pattern": pattern, "size": size, "shape": shape}
    if a.embedding is not None and b.embedding is not None:
        parts["clip"] = float(np.clip(_cosine(a.embedding, b.embedding), 0.0, 1.0))
    active = {name: float(weight) for name, weight in weights.items() if name in parts and weight > 0}
    weight_sum = sum(active.values())
    if weight_sum <= 0:
        raise ValueError("At least one available matching weight must be positive")
    total = sum(weight * parts[name] for name, weight in active.items()) / weight_sum
    return PairMatch(a.id, b.id, float(total), parts)


def find_pairs(
    socks: list[SockObservation], weights: dict, threshold: float
) -> tuple[list[PairMatch], list[int], list[PairMatch]]:
    """Find the maximum-score non-overlapping pairing with optional singles."""
    by_ids: dict[tuple[int, int], PairMatch] = {}
    all_scores: list[PairMatch] = []
    for i, first in enumerate(socks):
        for second in socks[i + 1 :]:
            pair = score_pair(first, second, weights)
            by_ids[(first.id, second.id)] = pair
            all_scores.append(pair)

    ids = tuple(sock.id for sock in socks)

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]) -> tuple[float, tuple[tuple[int, int], ...]]:
        if not remaining:
            return 0.0, ()
        first, *rest = remaining
        best_score, best_pairs = solve(tuple(rest))
        for index, second in enumerate(rest):
            key = (min(first, second), max(first, second))
            candidate = by_ids[key]
            if candidate.score < threshold:
                continue
            tail = tuple(value for pos, value in enumerate(rest) if pos != index)
            tail_score, tail_pairs = solve(tail)
            score = candidate.score + tail_score
            if score > best_score:
                best_score = score
                best_pairs = (key,) + tail_pairs
        return best_score, best_pairs

    _, chosen_keys = solve(ids)
    chosen = [by_ids[key] for key in chosen_keys]
    matched = {sock_id for pair in chosen_keys for sock_id in pair}
    singles = [sock.id for sock in socks if sock.id not in matched]
    return chosen, singles, sorted(all_scores, key=lambda pair: pair.score, reverse=True)
