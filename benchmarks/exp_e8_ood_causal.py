"""E8 — Out-of-distribution causal generalization (reviewer #8 / P0).

The strongest challenge to the causal claim: attribution/prediction is only
shown on the same drift mechanisms it trained on. Here we test mechanism-held-
out generalization of the counterfactual surrogate:

  TRAIN : intervention triples from COVARIATE-shift drifts only.
  TEST-ID  : held-out COVARIATE-shift triples (in-distribution).
  TEST-OOD : CONCEPT-shift triples (an unseen mechanism family).

We report MAE / R^2 on ID vs OOD. A small ID-vs-OOD gap = the surrogate
generalizes across mechanisms; a large gap = it memorizes the training
mechanism (an honest negative). Either way this is the experiment the paper
currently lists only as a limitation.

Usage:
    python -m benchmarks.exp_e8_ood_causal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from cadence.adapters.neural import FraudNet, FraudNetConfig
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
from cadence.common.device import get_device
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud

log = get_logger("cadence.benchmarks.e8")


def _f1(adapter, X, y):
    p = adapter.predict_proba(X)
    return float(f1_score(y, (p >= adapter.decision_threshold).astype(int), zero_division=0))


def eval_pool(model, surrogate, samples, dev):
    model.eval()
    surrogate.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for s in samples:
            _, emb = model(s.data)
            ctx = torch.tensor([[s.pre_fix_f1, s.severity]], dtype=torch.float32, device=dev)
            pred = float(
                surrogate(emb[s.candidate_node_idx : s.candidate_node_idx + 1], ctx).item()
            )
            preds.append(pred)
            tgts.append(float(s.post_fix_f1))
    preds = np.array(preds)
    tgts = np.array(tgts)
    if preds.size == 0:
        return {"mae": None, "r2": None, "n": 0}
    mae = float(np.mean(np.abs(preds - tgts)))
    ss_res = float(np.sum((tgts - preds) ** 2))
    ss_tot = float(np.sum((tgts - tgts.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"mae": mae, "r2": r2, "n": int(preds.size)}


def _gen(
    adapter,
    scorer,
    bstate,
    bthr,
    X_stream,
    y_stream,
    X_pre,
    y_pre,
    names,
    n_drifts,
    mech_mix,
    seed,
    dev,
):
    return generate_intervention_dataset(
        adapter=adapter,
        tap=scorer.tap,
        node_set=scorer.node_set,
        baseline_state=bstate,
        baseline_threshold=bthr,
        baseline_means=scorer.baseline_means,
        baseline_stds=scorer.baseline_stds,
        baseline_stream_X=X_stream,
        baseline_stream_y=y_stream,
        replay_X=X_pre,
        replay_y=y_pre,
        feature_names=names,
        n_drifts=n_drifts,
        candidates_per_drift=3,
        finetune_epochs=2,
        window_size=512,
        windows_per_sample=3,
        mechanism_mix=mech_mix,
        seed=seed,
        device=dev,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-drifts", type=int, default=140)
    p.add_argument("--test-drifts", type=int, default=45)
    p.add_argument("--joint-epochs", type=int, default=25)
    p.add_argument("--out", default="experiments/exp_e8_ood_causal.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    npre = int(0.70 * n)
    nstr = int(0.25 * n)
    X_pre, y_pre = ds.X_train[idx[:npre]], ds.y_train[idx[:npre]]
    X_stream, y_stream = ds.X_train[idx[npre : npre + nstr]], ds.y_train[idx[npre : npre + nstr]]

    set_global_seed(42)
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
    bstate = adapter.state_dict()
    bthr = adapter.decision_threshold
    bf1 = _f1(adapter, X_pre[-5000:], y_pre[-5000:])

    scorer = GNNResponsibilityScorer.build(
        adapter=adapter,
        baseline_X=X_pre[:20000],
        feature_names=ds.feature_names,
        baseline_f1=bf1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
        gnn_cfg=GNNConfig(),
    )

    names = ds.feature_names
    log.info("gen_train_covariate")
    train_pool = _gen(
        adapter,
        scorer,
        bstate,
        bthr,
        X_stream,
        y_stream,
        X_pre,
        y_pre,
        names,
        args.train_drifts,
        (1.0, 0.0),
        0,
        dev,
    )
    log.info("gen_test_id_covariate")
    test_id = _gen(
        adapter,
        scorer,
        bstate,
        bthr,
        X_stream,
        y_stream,
        X_pre,
        y_pre,
        names,
        args.test_drifts,
        (1.0, 0.0),
        7777,
        dev,
    )
    log.info("gen_test_ood_concept")
    test_ood = _gen(
        adapter,
        scorer,
        bstate,
        bthr,
        X_stream,
        y_stream,
        X_pre,
        y_pre,
        names,
        args.test_drifts,
        (0.0, 1.0),
        8888,
        dev,
    )
    log.info("pools", train=len(train_pool), id=len(test_id), ood=len(test_ood))

    surrogate = build_surrogate_on_device(SurrogateConfig(embedding_dim=16))
    train_joint(
        scorer.model,
        surrogate,
        train_pool,
        cfg=JointTrainConfig(lr=3e-3, epochs=args.joint_epochs),
        device=dev,
    )

    id_res = eval_pool(scorer.model, surrogate, test_id, dev)
    ood_res = eval_pool(scorer.model, surrogate, test_ood, dev)

    summary = {
        "experiment": "E8_ood_causal_generalization",
        "train_mechanism": "covariate_shift",
        "n_train": len(train_pool),
        "test_id": {"mechanism": "covariate_shift", **id_res},
        "test_ood": {"mechanism": "concept_shift", **ood_res},
        "ood_mae_gap": (
            None
            if id_res["mae"] is None or ood_res["mae"] is None
            else ood_res["mae"] - id_res["mae"]
        ),
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E8: OOD causal generalization (mechanism-held-out surrogate) ===")
    print(f"Train: covariate-shift ({len(train_pool)} triples)")
    print(f"TEST-ID  (covariate): MAE={id_res['mae']}, R2={id_res['r2']}, n={id_res['n']}")
    print(f"TEST-OOD (concept)  : MAE={ood_res['mae']}, R2={ood_res['r2']}, n={ood_res['n']}")
    print(f"OOD MAE gap = {summary['ood_mae_gap']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
