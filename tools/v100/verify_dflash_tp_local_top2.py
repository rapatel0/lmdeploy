#!/usr/bin/env python3
"""Verify exact sharded top-2 ordering, EOS masks, padding, and rejection."""
from __future__ import annotations

import math

TP, LOCAL, VOCAB, K = 4, 8, 29, 7


def top2(values: list[float], valid: range, eos: set[int]) -> list[tuple[float, int]]:
    return sorted(((values[i], i) for i in valid if i not in eos), key=lambda x: (-x[0], x[1]))[:2]


def reject(rows: list[list[tuple[float, int]]], drafts: list[int], margin: float = 0.0) -> tuple[int, int, int]:
    ambiguous = False
    for pos, candidates in enumerate(rows):
        best = sorted(candidates, key=lambda x: (-x[0], x[1]))[:2]
        ambiguous |= len(best) == 2 and best[1][0] >= best[0][0] - margin
        if pos == K or drafts[pos] != best[0][1]:
            return pos, best[0][1], int(ambiguous)
    raise AssertionError("missing bonus row")


def main() -> None:
    winners = [3, 6, 7, 8, 9, 10, 11, 12]
    logits = [[-100.0] * (TP * LOCAL) for _ in range(K + 1)]
    for pos, winner in enumerate(winners):
        for token in range(VOCAB):
            logits[pos][token] = -float(token)
        logits[pos][winner] = 20.0
    logits[0][5] = 19.0
    logits[1][14] = 20.0
    full_rows: list[list[tuple[float, int]]] = []
    shard_rows: list[list[tuple[float, int]]] = []
    for pos, values in enumerate(logits):
        masked = {3, 10} if pos < 2 else set()
        full_rows.append(top2(values, range(VOCAB), masked))
        compact: list[tuple[float, int]] = []
        for rank in range(TP):
            begin, end = rank * LOCAL, min((rank + 1) * LOCAL, VOCAB)
            compact.extend(top2(values, range(begin, end), masked))
        shard_rows.append(sorted(compact, key=lambda x: (-x[0], x[1]))[:2])
    assert full_rows == shard_rows
    assert all(math.isfinite(score) for row in shard_rows for score, _ in row)
    assert reject(shard_rows, [-1] * K) == (0, 5, 0)
    assert reject(shard_rows, [-1] * K, margin=20.0) == (0, 5, 1)
    assert reject(shard_rows, [5, 6] + [-1] * 5) == (2, 7, 1)
    assert reject(shard_rows, [5, 6, 7, 8, 9, 10, 11]) == (7, 12, 1)
    print("DFLASH_TP_LOCAL_TOP2_MICRO_PASS ties=1 padded_tail=1 eos_mask=1 zero=1 partial=1 full=1")


if __name__ == "__main__":
    main()
