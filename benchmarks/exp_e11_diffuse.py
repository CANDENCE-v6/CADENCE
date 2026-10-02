"""E11 — Diffuse vs concentrated responsibility (reviewer #11).

CADENCE's central idea is "identify the responsible part of the system," and
its policy state includes an attribution-concentration signal. But real drift
is not always localized. This experiment injects drift on a growing number of
features (1, 3, 5, 10) and measures:

  * recall@k : of the k injected (responsible) features, how many appear in the
               scorer's top-k ranking (attribution quality as drift diffuses).
  * concentration : max(score)/sum(|score|) of the GNN attribution — should
               DECREASE as responsibility spreads, which is exactly the signal
               the RSO uses to escalate no-op/partial -> full retrain.

If concentration falls monotonically with the number of responsible features,
the diffuse-responsibility guard is empirically justified.

Usage:
    python -m benchmarks.exp_e11_diffuse --seeds 8
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import (
    GNNConfig,
    GNNResponsibilityScorer,
    GNNTrainConfig,
    generate_sandbox_dataset,
    train_gnn,
)
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import get_device
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.e11")


def _ms(vs):
    vs = list(vs)
    return (
        (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)
        if vs
        else (0.0, 0.0)
    )


def _f1(adapter, X, y):
    p = adapter.predict_proba(X)
    return float(f1_score(y, (p >= adapter.decision_threshold).astype(int), zero_division=0))


def _recall_at_k(scores, responsible, k):
    topk = set(np.argsort(-scores)[:k].tolist())
    return len(topk & set(responsible)) / len(responsible)


def _concentration(scores):
    s = np.abs(np.asarray(scores, dtype=float))
    tot = s.sum()
    return float(s.max() / tot) if tot > 0 else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-responsible", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument(
        "--shift",
        type=float,
        default=2.0,
        help="additive shift (in scaled space) on responsible features",
    )
    p.add_argument("--window-size", type=int, default=3072)
    p.add_argument("--gnn-samples", type=int, default=200)
    p.add_argument("--gnn-epochs", type=int, default=20)
    p.add_argument("--out", default="experiments/exp_e11_diffuse.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n = ds.X_train.shape[0]
    rng0 = np.random.default_rng(42)
    idx = np.arange(n)
    rng0.shuffle(idx)
    npre = int(0.70 * n)
    nstr = int(0.25 * n)
    X_pre, y_pre = ds.X_train[idx[:npre]], ds.y_train[idx[:npre]]
    X_stream, y_stream = ds.X_train[idx[npre : npre + nstr]], ds.y_train[idx[npre : npre + nstr]]
    n_features = ds.X_train.shape[1]

    set_global_seed(42)
    adapter = FraudNet(
        FraudNetConfig(
            input_dim=n_features,
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            max_epochs=cfg.model.max_epochs,
            early_stopping_patience=cfg.model.early_stopping_patience,
            class_weighted=cfg.model.class_weighted,
        )
    )
    adapter.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    bf1 = _f1(adapter, X_pre[-5000:], y_pre[-5000:])

    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    psi_scorer = PSIResponsibilityScorer(trigger=trigger)
    gnn_scorer = GNNResponsibilityScorer.build(
        adapter=adapter,
        baseline_X=X_pre[:20000],
        feature_names=ds.feature_names,
        baseline_f1=bf1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
        gnn_cfg=GNNConfig(),
    )
    samples = generate_sandbox_dataset(
        adapter=adapter,
        tap=gnn_scorer.tap,
        node_set=gnn_scorer.node_set,
        baseline_means=gnn_scorer.baseline_means,
        baseline_stds=gnn_scorer.baseline_stds,
        baseline_stream_X=X_stream,
        baseline_stream_y=y_stream,
        feature_names=ds.feature_names,
        n_samples=args.gnn_samples,
        window_size=512,
        seed=0,
        device=dev,
    )
    train_gnn(
        gnn_scorer.model, samples, cfg=GNNTrainConfig(lr=3e-3, epochs=args.gnn_epochs), device=dev
    )
    log.info("setup_done", baseline_f1=round(bf1, 4))

    results = {
        k: {"recall_gnn": [], "recall_psi": [], "conc_gnn": [], "conc_psi": [], "f1_drop": []}
        for k in args.n_responsible
    }

    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        base = X_stream[rng.integers(0, X_stream.shape[0], size=args.window_size)]
        base_y = y_stream[rng.integers(0, y_stream.shape[0], size=args.window_size)]
        # use a coherent (X,y) slice instead of mismatched sampling
        start = rng.integers(0, max(1, X_stream.shape[0] - args.window_size))
        base = X_stream[start : start + args.window_size].copy()
        base_y = y_stream[start : start + args.window_size].copy()
        f1_clean = _f1(adapter, base, base_y)

        for k in args.n_responsible:
            responsible = sorted(rng.choice(n_features, size=k, replace=False).tolist())
            Xd = base.copy()
            for j in responsible:
                Xd[:, j] = Xd[:, j] + args.shift
            f1_drift = _f1(adapter, Xd, base_y)
            gnn_sc = np.asarray(gnn_scorer(adapter, Xd, base_y), dtype=float)
            psi_sc = np.asarray(psi_scorer(adapter, Xd, base_y), dtype=float)
            results[k]["recall_gnn"].append(_recall_at_k(gnn_sc, responsible, k))
            results[k]["recall_psi"].append(_recall_at_k(psi_sc, responsible, k))
            results[k]["conc_gnn"].append(_concentration(gnn_sc))
            results[k]["conc_psi"].append(_concentration(psi_sc))
            results[k]["f1_drop"].append(f1_clean - f1_drift)
        log.info("e11_seed_done", seed=seed)

    summary = {
        "experiment": "E11_diffuse_responsibility",
        "seeds": args.seeds,
        "shift": args.shift,
        "n_features": n_features,
        "per_k": {},
        "config": vars(args),
    }
    for k in args.n_responsible:
        r = results[k]
        summary["per_k"][str(k)] = {
            "recall_at_k_gnn": _ms(r["recall_gnn"]),
            "recall_at_k_psi": _ms(r["recall_psi"]),
            "concentration_gnn": _ms(r["conc_gnn"]),
            "concentration_psi": _ms(r["conc_psi"]),
            "f1_drop": _ms(r["f1_drop"]),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E11: diffuse vs concentrated responsibility ===")
    print(f"{'#resp':<8}{'recall@k GNN':<16}{'recall@k PSI':<16}{'conc GNN':<12}{'F1 drop':<10}")
    for k in args.n_responsible:
        s = summary["per_k"][str(k)]
        print(
            f"{k:<8}{s['recall_at_k_gnn'][0]:.3f}           {s['recall_at_k_psi'][0]:.3f}           "
            f"{s['concentration_gnn'][0]:.3f}       {s['f1_drop'][0]:+.3f}"
        )
    print(
        "\n(Concentration should fall as #responsible rises -> justifies the escalate-to-full guard.)"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
