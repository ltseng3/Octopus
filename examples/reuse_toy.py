#!/usr/bin/env python3
"""Exact inference-reuse toy: independent Bernoulli artifacts, not a detector.

A group of s receivers adopts one binary result; a wrong artifact makes all
s receivers wrong. Distinct artifacts have independent errors of probability q.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

PARTITIONS = ((1, 1, 1, 1, 1), (2, 2, 1), (3, 1, 1), (5,))


def risk_distribution(groups: tuple[int, ...], q: float) -> list[float]:
    """Return P(K=r) for r=0,...,n by exhaustive enumeration."""
    if not groups or any(s <= 0 or int(s) != s for s in groups):
        raise ValueError("Group sizes must be positive integers")
    if not 0 <= q <= 1:
        raise ValueError("q must lie in [0,1]")
    pmf = [0.0] * (sum(groups) + 1)
    for errors in itertools.product((0, 1), repeat=len(groups)):
        k = sum(s * e for s, e in zip(groups, errors))
        pmf[k] += math.prod(q if e else 1 - q for e in errors)
    return pmf


def rows(q: float):
    for groups in PARTITIONS:
        pmf = risk_distribution(groups, q)
        for m in range(1, sum(groups) + 1):
            yield {"partition": "+".join(map(str, groups)), "jobs": len(groups),
                   "q": q, "m": m, "risk": sum(pmf[m:])}


def self_test() -> None:
    a, b = (risk_distribution(s, 0.1) for s in ((2, 2, 1), (3, 1, 1)))
    assert math.isclose(sum(a), 1.0) and math.isclose(sum(b), 1.0)
    assert math.isclose(sum(a[3:]), 0.028)
    assert math.isclose(sum(b[3:]), 0.1)
    assert math.isclose(a[5], b[5]) and math.isclose(a[5], 0.001)
    for groups in PARTITIONS:
        p = risk_distribution(groups, 0.1)
        assert math.isclose(sum(k * x for k, x in enumerate(p)) / 5, 0.1)
    assert risk_distribution((2, 1), 0) == [1, 0, 0, 0]
    assert risk_distribution((2, 1), 1) == [0, 0, 0, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("outputs/reuse_toy.csv"))
    args = parser.parse_args()
    self_test()
    data = list(rows(args.q))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    for row in data:
        if row["m"] == 3:
            print(f'{row["partition"]}: {row["jobs"]} jobs; R3={row["risk"]:.4%}')
    print(f"Self-tests passed; wrote {args.output}")


if __name__ == "__main__":
    main()
