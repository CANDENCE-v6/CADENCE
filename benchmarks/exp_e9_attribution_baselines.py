"""E9 — Stronger attribution baselines for H1 (reviewer #9).

The H1 comparison so far is PSI vs CDAG-structural vs CDAG+GNN. A reviewer
rightly asks: does the GNN beat *established* attribution methods, not just
PSI? We add two on the same per-feature root-cause ranking task:

  * grad      — input-gradient attribution: mean |d logit / d x_j| over the
                drifted window (Integrated-Gradients / saliency family).
  * perm      — counterfactual permutation: restore feature j to its baseline
                distribution and measure the F1 recovered; the feature whose
                restoration recovers most F1 is the drift cause (permutation-
                importance adapted to drift root-cause).

Metrics per scorer: top-1, MRR, AUROC (root-vs-rest), reported on the HARD
(contested) subset where the H1 question is actually decided, plus ALL.

Usage:
    python -m benchmarks.exp_e9_attribution_baselines --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sci_stats
from sklearn.metrics import f1_score, roc_auc_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import (
    CDAGResponsibilityScorer, GNNConfig, GNNResponsibilityScorer,
    GNNTrainConfig, generate_sandbox_dataset, train_gnn,
)
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import get_device
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.e9")

HARD = {"time_gradual", "v14_concept_shift"}


def _rank_of(scores, t):
    return int(np.where(np.argsort(-scores) == t)[0][0]) + 1


def _auroc(scores, t):
    lab = np.zeros_like(scores); lab[t] = 1
    try:
        return float(roc_auc_score(lab, scores))
    except ValueError:
        return float("nan")


def _ms(vs):
    vs = [v for v in vs if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0) if vs else (0.0, 0.0)


def _f1(adapter, X, y):
    p = adapter.predict_proba(X)
    return float(f1_score(y, (p >= adapter.decision_threshold).astype(int), zero_division=0))


def grad_scores(adapter, X):
    mod = adapter._module; dev = get_device()
    mod.eval()
    xb = torch.tensor(X[:2048].astype(np.float32), device=dev, requires_grad=True)
    logits, _ = mod(xb)
    mod.zero_grad(set_to_none=True)
    logits.abs().sum().backward()
    return xb.grad.detach().abs().mean(0).cpu().numpy()


def perm_scores(adapter, X, y, baseline_X, rng):
    """Restore each feature to baseline; score = F1 recovered."""
    cur = _f1(adapter, X, y)
    scores = np.zeros(X.shape[1])
    bcol = baseline_X[rng.integers(0, baseline_X.shape[0], size=X.shape[0])]
    for j in range(X.shape[1]):
        Xr = X.copy()
        Xr[:, j] = bcol[:, j]
        scores[j] = _f1(adapter, Xr, y) - cur
    return scores


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--windows-per-episode", type=int, default=3)
    p.add_argument("--gnn-samples", type=int, default=200)
    p.add_argument("--gnn-epochs", type=int, default=20)
    p.add_argument("--out", default="experiments/exp_e9_attribution_baselines.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n = ds.X_train.shape[0]; rng = np.random.default_rng(42); idx = np.arange(n); rng.shuffle(idx)
    npre = int(0.70 * n); nstr = int(0.25 * n)
    X_pre, y_pre = ds.X_train[idx[:npre]], ds.y_train[idx[:npre]]
    X_stream, y_stream = ds.X_train[idx[npre:npre+nstr]], ds.y_train[idx[npre:npre+nstr]]

    set_global_seed(42)
    adapter = FraudNet(FraudNetConfig(
        input_dim=ds.X_train.shape[1], hidden_dims=list(cfg.model.hidden_dims),
        dropout=cfg.model.dropout, lr=cfg.model.lr, batch_size=cfg.model.batch_size,
        max_epochs=cfg.model.max_epochs, early_stopping_patience=cfg.model.early_stopping_patience,
        class_weighted=cfg.model.class_weighted))
    adapter.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    bf1 = _f1(adapter, X_pre[-5000:], y_pre[-5000:])

    trigger = PSITrigger(baseline_X=X_pre, feature_names=ds.feature_names,
                         cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200))
    psi_scorer = PSIResponsibilityScorer(trigger=trigger)
    cdag_scorer = CDAGResponsibilityScorer.build(
        adapter, baseline_X=X_pre[:20000], feature_names=ds.feature_names, baseline_f1=bf1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters}, window_size=512)
    gnn_scorer = GNNResponsibilityScorer.build(
        adapter=adapter, baseline_X=X_pre[:20000], feature_names=ds.feature_names, baseline_f1=bf1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512, gnn_cfg=GNNConfig())
    samples = generate_sandbox_dataset(
        adapter=adapter, tap=gnn_scorer.tap, node_set=gnn_scorer.node_set,
        baseline_means=gnn_scorer.baseline_means, baseline_stds=gnn_scorer.baseline_stds,
        baseline_stream_X=X_stream, baseline_stream_y=y_stream, feature_names=ds.feature_names,
        n_samples=args.gnn_samples, window_size=512, seed=0, device=dev)
    train_gnn(gnn_scorer.model, samples, cfg=GNNTrainConfig(lr=3e-3, epochs=args.gnn_epochs), device=dev)
    log.info("setup_done", baseline_f1=round(bf1, 4), n_samples=len(samples))

    scorer_names = ["psi", "grad", "perm", "cdag_structural", "gnn_learned"]
    rows = []
    for scenario in build_default_scenarios(ds.feature_names):
        for seed in range(args.seeds):
            set_global_seed(seed)
            sX, sY, injector = make_baseline_stream(X_stream, y_stream, scenario, seed=seed)
            gt = injector.ground_truth().root_cause_feature_idx
            nuse = args.window_size * args.windows_per_episode
            wX, wY = sX[:nuse], sY[:nuse]
            all_scores = {
                "psi": psi_scorer(adapter, wX, wY),
                "grad": grad_scores(adapter, wX),
                "perm": perm_scores(adapter, wX, wY, X_pre, rng),
                "cdag_structural": cdag_scorer(adapter, wX, wY),
                "gnn_learned": gnn_scorer(adapter, wX, wY),
            }
            for name in scorer_names:
                sc = np.asarray(all_scores[name], dtype=float)
                r = _rank_of(sc, gt)
                rows.append({"scorer": name, "scenario": scenario.name, "seed": seed,
                             "subset": "hard" if scenario.name in HARD else "easy",
                             "rank": r, "top1": int(r == 1), "mrr": 1.0/r, "auroc": _auroc(sc, gt)})
            log.info("e9_cell", scenario=scenario.name, seed=seed)

    def agg(name, subset=None):
        r = [x for x in rows if x["scorer"] == name and (subset is None or x["subset"] == subset)]
        return {"n": len(r), "top1": _ms([x["top1"] for x in r]),
                "mrr": _ms([x["mrr"] for x in r]), "auroc": _ms([x["auroc"] for x in r])}

    def wilcox(a, b, subset, metric):
        keys = sorted({(x["scenario"], x["seed"]) for x in rows if x["subset"] == subset})
        av, bv = [], []
        for scen, sd in keys:
            ax = next(x for x in rows if x["scorer"] == a and x["scenario"] == scen and x["seed"] == sd)
            bx = next(x for x in rows if x["scorer"] == b and x["scenario"] == scen and x["seed"] == sd)
            if np.isnan(ax[metric]) or np.isnan(bx[metric]):
                continue
            av.append(ax[metric]); bv.append(bx[metric])
        try:
            _, pv = sci_stats.wilcoxon(av, bv, alternative="greater")
            return {"p": float(pv), "n": len(av)}
        except ValueError as e:
            return {"p": None, "err": str(e), "n": len(av)}

    summary = {
        "experiment": "E9_attribution_baselines", "seeds": args.seeds,
        "aggregates": {sub: {name: agg(name, sub) for name in scorer_names} for sub in ("hard", "easy")},
        "aggregate_all": {name: agg(name) for name in scorer_names},
        "wilcoxon_gnn_greater_hard": {
            b: {m: wilcox("gnn_learned", b, "hard", m) for m in ("mrr", "auroc")}
            for b in ("psi", "grad", "perm", "cdag_structural")},
        "rows": rows, "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E9: attribution baselines — HARD subset (contested) ===")
    print(f"{'scorer':<16}{'top1':<10}{'MRR':<10}{'AUROC':<10}")
    for name in scorer_names:
        a = summary["aggregates"]["hard"][name]
        print(f"{name:<16}{a['top1'][0]:.3f}     {a['mrr'][0]:.3f}     {a['auroc'][0]:.3f}")
    print("\nWilcoxon GNN > baseline (HARD):")
    for b, d in summary["wilcoxon_gnn_greater_hard"].items():
        print(f"  gnn>{b:<16} MRR p={d['mrr'].get('p')}  AUROC p={d['auroc'].get('p')}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
