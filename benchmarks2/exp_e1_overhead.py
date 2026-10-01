"""E1 — Total CADENCE overhead vs retraining cost (reviewer priority #3 / E1).

The reviewer's decisive question: CADENCE reports *retraining* compute saved,
but CADENCE itself spends compute on monitoring + CDAG + GNN + surrogate +
policy. If that overhead eats the retraining savings, the "cost-aware" thesis
collapses. This experiment MEASURES every component's wall-clock on the real
GPU, then composes the net-compute-saved equation:

    C_cadence  = N * (t_monitor + t_cdag + t_gnn + t_surrogate + t_policy)
                 + n_partial * t_partial + n_full * t_full
    C_reactive = N * t_monitor + N * t_full        # PSI -> full every window
    net_saved  = C_reactive - C_cadence

Component wall-times are measured over repeated calls on a representative
1024-row window; retrains are timed at the Elec2 protocol (finetune 2 epochs,
full 5 epochs). The action profile (N, n_partial, n_full) is taken from the
measured Elec2 run (R-Gate-E: CADENCE {5 no-op, 4 partial, 3 full} over 10
windows; reactive-full {0,0,10}).

Usage:
    python -m benchmarks.exp_e1_overhead
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import (
    GNNConfig,
    GNNResponsibilityScorer,
    SurrogateConfig,
    build_surrogate_on_device,
)
from cadence.attribution.gnn import cdag_to_pyg_data
from cadence.cdag import build_cdag_from_windows, compute_windowed_signals
from cadence.collector.drift_trigger import PSITrigger
from cadence.common.config import load_config
from cadence.common.device import get_device, log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud

log = get_logger("cadence.benchmarks.e1")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, reps: int, warmup: int = 2) -> float:
    """Median wall-seconds per call over `reps` (after warmup)."""
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--reps", type=int, default=25)
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--fullretrain-epochs", type=int, default=5)
    # Elec2 measured action profile (R-Gate-E-verified).
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--n-partial", type=int, default=4)
    p.add_argument("--n-full", type=int, default=3)
    p.add_argument("--out", default="experiments/exp_e1_overhead.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    log_device_info(dev)
    set_global_seed(42)

    # --- Pretrain FraudNet (same setup as Gate C) ---
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_pre = int(0.70 * n_train)
    X_pre, y_pre = ds.X_train[idx[:n_pre]], ds.y_train[idx[:n_pre]]
    X_stream = ds.X_train[idx[n_pre : n_pre + 20000]]
    y_stream = ds.y_train[idx[n_pre : n_pre + 20000]]

    adapter = FraudNet(FraudNetConfig(
        input_dim=ds.X_train.shape[1], hidden_dims=list(cfg.model.hidden_dims),
        dropout=cfg.model.dropout, lr=cfg.model.lr, batch_size=cfg.model.batch_size,
        max_epochs=cfg.model.max_epochs, early_stopping_patience=cfg.model.early_stopping_patience,
        class_weighted=cfg.model.class_weighted,
    ))
    adapter.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    baseline_state = adapter.state_dict()
    baseline_threshold = adapter.decision_threshold
    from sklearn.metrics import f1_score
    bprobs = adapter.predict_proba(X_pre[-5000:])
    baseline_f1 = float(f1_score(y_pre[-5000:], (bprobs >= baseline_threshold).astype(int), zero_division=0))
    log.info("pretrain_done", baseline_f1=round(baseline_f1, 4))

    # --- Build attribution scorer (tap + node_set + GNN) + untrained surrogate ---
    scorer = GNNResponsibilityScorer.build(
        adapter=adapter, baseline_X=X_pre[:20000], feature_names=ds.feature_names,
        baseline_f1=baseline_f1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512, gnn_cfg=GNNConfig(),
    )
    surrogate = build_surrogate_on_device(SurrogateConfig(embedding_dim=16))
    scorer.surrogate = surrogate

    trigger = PSITrigger(baseline_X=X_pre[:20000], feature_names=list(ds.feature_names))

    # Representative window.
    W = X_stream[: args.window_size]
    Wy = y_stream[: args.window_size]
    w_size = min(scorer.window_size, max(1, W.shape[0] // 4))

    # ---------- Time each per-window component ----------
    def do_monitor():
        trigger.psi_per_feature(W)

    def do_cdag():
        sig = compute_windowed_signals(
            W, Wy, scorer.tap, scorer.node_set, scorer.baseline_means,
            scorer.baseline_stds, window_f1=baseline_f1, window_size=w_size)
        build_cdag_from_windows(sig, pc_alpha=0.05, notears_lambda=0.05, notears_max_iter=25)

    # Precompute signals+cdag once for GNN/surrogate timing.
    sig = compute_windowed_signals(
        W, Wy, scorer.tap, scorer.node_set, scorer.baseline_means,
        scorer.baseline_stds, window_f1=baseline_f1, window_size=w_size)
    cdag = build_cdag_from_windows(sig, pc_alpha=0.05, notears_lambda=0.05, notears_max_iter=25)
    data = cdag_to_pyg_data(cdag.weights, sig[-1], device=dev)

    def do_gnn():
        scorer.model.eval()
        with torch.no_grad():
            data2 = cdag_to_pyg_data(cdag.weights, sig[-1], device=dev)
            scorer.model(data2)

    scorer.model.eval()
    with torch.no_grad():
        _, emb = scorer.model(data)

    def do_surrogate():
        surrogate.eval()
        with torch.no_grad():
            ctx = torch.tensor([[baseline_f1, 1.0]], dtype=torch.float32, device=dev)
            surrogate(emb[0:1], ctx)

    def do_policy():
        # rule policy: a few numpy ops on the observation vector
        obs = np.zeros(15, dtype=np.float32)
        _ = (obs[13] <= 1e-3, obs[14] > 0.5)

    t_monitor = _time(do_monitor, args.reps)
    t_cdag = _time(do_cdag, max(5, args.reps // 3))
    t_gnn = _time(do_gnn, args.reps)
    t_surrogate = _time(do_surrogate, args.reps)
    t_policy = _time(do_policy, args.reps)

    # ---------- Time retrains at the Elec2 protocol ----------
    replay_idx = rng.integers(0, X_pre.shape[0], size=2000)
    rX, rY = X_pre[replay_idx], y_pre[replay_idx]

    def do_partial():
        ad = adapter.clone()
        ad.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
        ad.decision_threshold = baseline_threshold
        fisher = ad.compute_fisher(rX, rY)
        theta = {k: v.clone() for k, v in baseline_state.items()}
        ad.partial_fit(W, Wy, layers_to_update=["layer1"], ewc_penalty=1000.0,
                       fisher=fisher, theta_star=theta, replay_X=rX, replay_y=rY,
                       max_epochs=args.finetune_epochs, finetune_lr=1e-4)
        del ad

    def do_full():
        ad = adapter.clone()
        ad.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
        ad.decision_threshold = baseline_threshold
        ad.cfg.max_epochs = args.fullretrain_epochs
        Xf = np.concatenate([W, rX], axis=0)
        yf = np.concatenate([Wy, rY], axis=0)
        ad.fit(Xf, yf)
        del ad

    t_partial = _time(do_partial, 3, warmup=1)
    t_full = _time(do_full, 3, warmup=1)

    # ---------- Compose the net-savings equation ----------
    N, npart, nfull = args.n_windows, args.n_partial, args.n_full
    overhead_per_window = t_monitor + t_cdag + t_gnn + t_surrogate + t_policy
    C_cadence = N * overhead_per_window + npart * t_partial + nfull * t_full
    C_reactive = N * t_monitor + N * t_full  # PSI monitor + full retrain every window
    net_saved = C_reactive - C_cadence
    pct_saved = 100.0 * net_saved / C_reactive if C_reactive > 0 else 0.0
    overhead_total = N * overhead_per_window
    overhead_frac_of_cadence = 100.0 * overhead_total / C_cadence if C_cadence > 0 else 0.0
    # retrain-only saving (what the paper currently reports) for comparison
    retrain_only_cadence = npart * t_partial + nfull * t_full
    retrain_only_reactive = N * t_full
    retrain_only_saved_pct = 100.0 * (retrain_only_reactive - retrain_only_cadence) / retrain_only_reactive

    comp = {
        "monitor_ms": t_monitor * 1e3, "cdag_ms": t_cdag * 1e3, "gnn_ms": t_gnn * 1e3,
        "surrogate_ms": t_surrogate * 1e3, "policy_ms": t_policy * 1e3,
        "partial_retrain_s": t_partial, "full_retrain_s": t_full,
    }
    summary = {
        "experiment": "E1_total_overhead",
        "hardware": "RTX 4070 Laptop",
        "window_size": args.window_size,
        "component_times": comp,
        "overhead_per_window_ms": overhead_per_window * 1e3,
        "action_profile": {"n_windows": N, "n_partial": npart, "n_full": nfull},
        "C_cadence_s": C_cadence, "C_reactive_full_s": C_reactive,
        "net_compute_saved_s": net_saved,
        "net_compute_saved_pct": pct_saved,
        "cadence_overhead_total_s": overhead_total,
        "overhead_as_pct_of_cadence_budget": overhead_frac_of_cadence,
        "retrain_only_saved_pct_reported_in_paper": retrain_only_saved_pct,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E1: Total CADENCE overhead vs retraining (measured, RTX 4070) ===")
    print(f"Per-window overhead components (ms):")
    print(f"  monitor(PSI)   : {comp['monitor_ms']:.3f}")
    print(f"  CDAG(PC+NOTEARS): {comp['cdag_ms']:.3f}")
    print(f"  GNN forward    : {comp['gnn_ms']:.3f}")
    print(f"  surrogate      : {comp['surrogate_ms']:.3f}")
    print(f"  policy         : {comp['policy_ms']:.3f}")
    print(f"  => overhead/window = {overhead_per_window*1e3:.3f} ms")
    print(f"Retrain costs (s): partial={t_partial:.3f}  full={t_full:.3f}")
    print(f"\nAction profile (Elec2): N={N}, partial={npart}, full={nfull}")
    print(f"C_cadence  = {C_cadence:.3f} s")
    print(f"C_reactive = {C_reactive:.3f} s")
    print(f"NET compute saved (incl. all overhead) = {net_saved:.3f} s ({pct_saved:.1f}%)")
    print(f"CADENCE's own overhead = {overhead_total*1e3:.1f} ms "
          f"= {overhead_frac_of_cadence:.4f}% of its compute budget")
    print(f"(retrain-only saving the paper reports: {retrain_only_saved_pct:.1f}%)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
