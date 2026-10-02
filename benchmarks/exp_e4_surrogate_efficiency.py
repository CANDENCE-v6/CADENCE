"""E4b — Surrogate sample-efficiency (reviewer #4).

The surrogate is supposed to AMORTIZE expensive sandbox interventions. So:
how many interventions are needed to train a useful surrogate? We generate a
single pool of intervention triples once, then train a FRESH GNN+surrogate on
nested budgets {10,25,50,100,250,500} and record held-out MAE / R^2. A curve
that drops fast and plateaus is evidence of amortization.

Usage:
    python -m benchmarks.exp_e4_surrogate_efficiency
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.data.loaders import load_credit_card_fraud

from cadence.attribution import (
    GNNConfig,
    GNNResponsibilityScorer,
    JointTrainConfig,
    SurrogateConfig,
    build_surrogate_on_device,
    generate_intervention_dataset,
    train_joint,
)
from cadence.common.config import load_config
from cadence.common.device import get_device, log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed

log = get_logger("cadence.benchmarks.e4")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100, 250, 500])
    p.add_argument("--pool", type=int, default=550, help="total triples to generate once")
    p.add_argument("--candidates-per-drift", type=int, default=3)
    p.add_argument("--joint-epochs", type=int, default=25)
    p.add_argument("--out", default="experiments/exp_e4_surrogate_efficiency.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    log_device_info(dev)
    set_global_seed(42)

    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_pre = int(0.70 * n_train)
    X_pre, y_pre = ds.X_train[idx[:n_pre]], ds.y_train[idx[:n_pre]]
    X_stream = ds.X_train[idx[n_pre : n_pre + 25000]]
    y_stream = ds.y_train[idx[n_pre : n_pre + 25000]]

    adapter = FraudNet(
        FraudNetConfig(
            input_dim=ds.X_train.shape[1],
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
    baseline_state = adapter.state_dict()
    baseline_threshold = adapter.decision_threshold
    from sklearn.metrics import f1_score

    bp = adapter.predict_proba(X_pre[-5000:])
    baseline_f1 = float(
        f1_score(y_pre[-5000:], (bp >= baseline_threshold).astype(int), zero_division=0)
    )

    scorer = GNNResponsibilityScorer.build(
        adapter=adapter,
        baseline_X=X_pre[:20000],
        feature_names=ds.feature_names,
        baseline_f1=baseline_f1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
        gnn_cfg=GNNConfig(),
    )

    n_drifts = int(np.ceil(args.pool / args.candidates_per_drift))
    log.info("generating_pool", n_drifts=n_drifts, target_triples=args.pool)
    pool = generate_intervention_dataset(
        adapter=adapter,
        tap=scorer.tap,
        node_set=scorer.node_set,
        baseline_state=baseline_state,
        baseline_threshold=baseline_threshold,
        baseline_means=scorer.baseline_means,
        baseline_stds=scorer.baseline_stds,
        baseline_stream_X=X_stream,
        baseline_stream_y=y_stream,
        replay_X=X_pre,
        replay_y=y_pre,
        feature_names=ds.feature_names,
        n_drifts=n_drifts,
        candidates_per_drift=args.candidates_per_drift,
        finetune_epochs=2,
        window_size=512,
        windows_per_sample=3,
        seed=0,
        device=dev,
    )
    log.info("pool_ready", n_triples=len(pool))

    # Fresh GNN + surrogate initial states, reset before each budget.
    surrogate = build_surrogate_on_device(SurrogateConfig(embedding_dim=16))
    gnn_init = copy.deepcopy(scorer.model.state_dict())
    sur_init = copy.deepcopy(surrogate.state_dict())

    rows = []
    for k in args.budgets:
        if k > len(pool):
            continue
        scorer.model.load_state_dict(copy.deepcopy(gnn_init))
        surrogate.load_state_dict(copy.deepcopy(sur_init))
        summary = train_joint(
            scorer.model,
            surrogate,
            pool[:k],
            cfg=JointTrainConfig(lr=3e-3, epochs=args.joint_epochs),
            device=dev,
        )
        row = {
            "budget": k,
            "val_mae": float(summary["final_val_mae"]),
            "val_r2": float(summary["final_val_r2"]),
            "val_mse": float(summary["final_val_mse"]),
            "n_val": max(1, int(0.2 * k)),
        }
        rows.append(row)
        print(
            f"budget={k:>4}  val_MAE={row['val_mae']:.4f}  val_R2={row['val_r2']:+.3f}  (n_val~{row['n_val']})"
        )
        log.info("e4_point", **row)

    summary = {
        "experiment": "E4_surrogate_efficiency",
        "pool_size": len(pool),
        "candidates_per_drift": args.candidates_per_drift,
        "joint_epochs": args.joint_epochs,
        "rows": rows,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
