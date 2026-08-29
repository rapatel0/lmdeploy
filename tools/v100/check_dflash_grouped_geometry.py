#!/usr/bin/env python3
"""Check the static Q8H4 mapping and the audited 1K launch geometry."""

import math

CTA_Q = 8
CTA_H = 4
LOCAL_Q_HEADS = 6
LOCAL_KV_HEADS = 1
CONTEXT_LEN = 1008
CTA_S = 64
MAX_SPLITS = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    flat_rows = [(flat // CTA_H, flat % CTA_H) for flat in range(CTA_Q * CTA_H)]
    require(
        flat_rows == [(query, head) for query in range(CTA_Q) for head in range(CTA_H)],
        "flattened query/head order is incorrect",
    )

    q_group_size = LOCAL_Q_HEADS // LOCAL_KV_HEADS
    ctas_per_group = math.ceil(q_group_size / CTA_H)
    require(ctas_per_group == 2, "expected two GQA head CTAs")

    outputs = set()
    head_groups = []
    for cta_h_index in range(ctas_per_group):
        local_head = cta_h_index * CTA_H
        valid_heads = min(CTA_H, q_group_size - local_head)
        heads = tuple(range(local_head, local_head + valid_heads))
        head_groups.append(heads)
        for query in range(CTA_Q):
            for head in heads:
                outputs.add((query, head))
    require(head_groups == [(0, 1, 2, 3), (4, 5)], "expected a 4+2 head partition")
    require(
        outputs == {(query, head) for query in range(CTA_Q) for head in range(LOCAL_Q_HEADS)},
        "grouped CTAs do not cover every query/head output exactly once",
    )

    # Q bias indexing must decode the logical head from every flattened
    # [query, head] row, including the 4+2 tail CTA.
    for head_base, valid_heads in ((0, 4), (4, 2)):
        for flat_row in range(CTA_Q * CTA_H):
            logical_head = flat_row % CTA_H
            if logical_head < valid_heads:
                require(
                    0 <= head_base + logical_head < LOCAL_Q_HEADS,
                    "grouped Q-bias indexing escapes the local head buffer",
                )

    tiles = math.ceil(CONTEXT_LEN / CTA_S)
    splits = min(MAX_SPLITS, tiles)
    tiles_per_split = math.ceil(tiles / splits)
    require((tiles, splits, tiles_per_split) == (16, 8, 2), "unexpected 1K split geometry")
    require(ctas_per_group * splits == 16, "expected sixteen grouped CTAs")

    # Fixed graph geometry may launch eight splits for a one-tile live
    # context. Split zero is the last active split and must publish one partial.
    live_tiles = 1
    active_split_indices = [index for index in range(MAX_SPLITS) if index < live_tiles]
    require(active_split_indices == [0], "one-tile context should activate only split zero")
    published_partial_count = active_split_indices[-1] + 1
    require(published_partial_count == 1, "split zero must publish one partial")

    print("GROUPED_Q8H4_GEOMETRY_PASS flat_m=32 heads=4+2 base_ctas=2 splits=8 launched_ctas=16 span_tokens=128")


if __name__ == "__main__":
    main()
