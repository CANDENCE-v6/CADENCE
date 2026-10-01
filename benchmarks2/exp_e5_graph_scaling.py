"""E5b — CDAG construction cost vs graph size (reviewer #5 / graph scalability).

Reviewer asks: PC+NOTEARS costs 374 ms/window at the current ~35-node graph;
how does it scale to 100 / 300 / 1000 nodes? This isolates the graph-build
cost by feeding synthetic (n_windows, n_nodes, 4) per-node signals of growing
n_nodes directly into build_cdag_from_windows (the real PC+NOTEARS path),
timing wall-clock and peak memory. Synthetic signals isolate discovery cost
from the model forward pass — exactly the quantity the reviewer wants.

Usage:
    python -m benchmarks.exp_e5_graph_scaling
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from cadence.cdag import build_cdag_from_windows
from cadence.common.logging import get_logger

log = get_logger("cadence.benchmarks.e5")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--node-counts", type=int, nargs="+", default=[30, 100, 300, 1000])
    p.add_argument("--n-windows", type=int, default=12)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--notears-max-iter", type=int, default=25)
    p.add_argument("--out", default="experiments/exp_e5_graph_scaling.json")
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    rows = []
    for n_nodes in args.node_counts:
        # Synthetic per-node windowed signals with mild inter-node correlation
        # (so PC/NOTEARS have real structure to chew on, not pure noise).
        base = rng.standard_normal((args.n_windows, 1))
        sig = rng.standard_normal((args.n_windows, n_nodes, 4)).astype(np.float32)
        sig[:, :, 3] += 0.5 * base  # correlate the drift_z stat across nodes

        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            build_cdag_from_windows(
                sig, pc_alpha=0.05, notears_lambda=0.05,
                notears_max_iter=args.notears_max_iter)
            times.append(time.perf_counter() - t0)

        tracemalloc.start()
        build_cdag_from_windows(sig, pc_alpha=0.05, notears_lambda=0.05,
                                notears_max_iter=args.notears_max_iter)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        med = float(np.median(times))
        rows.append({"n_nodes": n_nodes, "median_s": med,
                     "peak_mem_mb": peak / 1024**2})
        print(f"n_nodes={n_nodes:>5}  median={med*1e3:8.1f} ms  peak_mem={peak/1024**2:7.1f} MB")
        log.info("e5_point", n_nodes=n_nodes, median_s=med, peak_mem_mb=peak / 1024**2)

    # Rough empirical scaling exponent from the two extremes (log-log slope).
    if len(rows) >= 2:
        n0, t0 = rows[0]["n_nodes"], rows[0]["median_s"]
        n1, t1 = rows[-1]["n_nodes"], rows[-1]["median_s"]
        exponent = float(np.log(t1 / t0) / np.log(n1 / n0)) if t0 > 0 else float("nan")
    else:
        exponent = float("nan")

    summary = {"experiment": "E5_graph_scaling", "n_windows": args.n_windows,
               "notears_max_iter": args.notears_max_iter, "rows": rows,
               "empirical_scaling_exponent": exponent, "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nEmpirical scaling exponent (t ~ n^k): k = {exponent:.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
