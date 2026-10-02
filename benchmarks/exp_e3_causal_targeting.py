"""E3 — Causal-targeting ablation for H3 (reviewer priority #2 / E3).

Question a reviewer rightly asks about H3: our earlier MNIST result shows
`partial-EWC < full retrain` on forgetting, but that proves *EWC helps*, not
that *causal targeting* helps. This experiment isolates the targeting axis.

Design (the ONLY thing that varies between arms is WHICH layer is retrained;
EWC penalty, Fisher, theta*, epochs, data are identical):

    Split-MNIST, Task A = {0,1} (protect), Task B = {2,3} (adapt).
    Model: MLP 784 -> 256 -> 128 -> 64 -> 1  (3 hidden layers => real choice).
    Per seed:
      1. Pretrain on Task A; snapshot theta* and Fisher(Task A).
      2. For each TARGETING method, clone the pretrained model and run an
         EWC-regularised partial_fit on Task B that trains ONLY the layer the
         method selects. Measure Task-A forgetting and Task-B F1.

    Targeting methods:
      * random   — pick a hidden layer uniformly at random.
      * psi      — input-feature drift (Task A vs B) attributes to the INPUT,
                   so the PSI-driven repair targets the input-facing layer1.
      * gradient — layer with the largest Task-B loss-gradient norm at theta*
                   (a standard gradient-attribution baseline).
      * cdag     — CADENCE's activation-cluster responsibility: the layer whose
                   ACTIVATION distribution drifts most between Task A and Task B
                   (the structural CDAG signal over internal nodes).
      * oracle   — try every single-layer target, report the one with the LOWEST
                   forgetting. The achievable ceiling; answers "is targeting the
                   bottleneck?".

H3-targeting is SUPPORTED iff `cdag` forgets less than `random`/`psi`/`gradient`
with paired Wilcoxon p<0.05. If it does not, we report that honestly.

Usage:
    python -m benchmarks.exp_e3_causal_targeting --seeds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sci_stats
from sklearn.metrics import f1_score

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.common.device import get_device
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.mnist_splits import load_split_mnist

log = get_logger("cadence.benchmarks.e3")

TARGETING_METHODS = ["random", "psi", "gradient", "cdag", "oracle"]


def _f1(adapter: FraudNet, X, y) -> float:
    if X.size == 0:
        return 0.0
    probs = adapter.predict_proba(X)
    preds = (probs >= adapter.decision_threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    qs = np.linspace(0, 100, bins + 1)
    cuts = np.percentile(expected, qs)
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=cuts)[0].astype(float) + 1e-6
    a = np.histogram(actual, bins=cuts)[0].astype(float) + 1e-6
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _select_layer(method, adapter, hidden_layers, theta_star, fisher, Xa, ya, Xb, yb, rng) -> str:
    """Return the hidden-layer name each targeting method chooses."""
    if method == "random":
        return hidden_layers[rng.integers(0, len(hidden_layers))]

    if method == "psi":
        # Input-feature drift attributes to the input -> input-facing layer.
        return hidden_layers[0]

    if method == "gradient":
        # Largest Task-B loss-gradient norm per layer at theta*.
        adapter.load_state_dict({k: v.clone() for k, v in theta_star.items()})
        mod = adapter._module
        mod.zero_grad(set_to_none=True)
        dev = get_device()
        xb = torch.from_numpy(Xb[:2048].astype(np.float32)).to(dev)
        yb_t = torch.from_numpy(yb[:2048].astype(np.float32)).to(dev)
        logits, _ = mod(xb)
        loss = nn.BCEWithLogitsLoss()(logits, yb_t)
        loss.backward()
        norms = {}
        for lname in hidden_layers:
            g = 0.0
            for pname, p in mod.named_parameters():
                if pname.split(".")[0] == lname and p.grad is not None:
                    g += float(p.grad.detach().pow(2).sum().item())
            norms[lname] = g**0.5
        mod.zero_grad(set_to_none=True)
        return max(norms, key=norms.get)

    if method == "cdag":
        # Activation-cluster responsibility: the layer whose activation
        # distribution shifts most between Task A and Task B (mean per-unit PSI).
        adapter.load_state_dict({k: v.clone() for k, v in theta_star.items()})
        _, tr_a = adapter.forward_with_traces(Xa[:2048])
        _, tr_b = adapter.forward_with_traces(Xb[:2048])
        drift = {}
        for lname in hidden_layers:
            a, b = tr_a[lname], tr_b[lname]
            # mean PSI across a sample of units in this layer's activation vector
            k = min(a.shape[1], 32)
            idx = np.linspace(0, a.shape[1] - 1, k).astype(int)
            drift[lname] = float(np.mean([_psi(a[:, j], b[:, j]) for j in idx]))
        return max(drift, key=drift.get)

    raise ValueError(method)


def _partial_retrain_forget(
    adapter_proto,
    baseline_state,
    threshold,
    hidden_layers,
    layer,
    fisher,
    theta_star,
    Xb,
    yb,
    Xa_test,
    ya_test,
    Xb_test,
    yb_test,
    pre_a_f1,
    ewc_penalty,
    epochs,
):
    """Clone -> EWC partial_fit on `layer` only -> return (forget, taskB_f1)."""
    ad = adapter_proto.clone()
    ad.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
    ad.decision_threshold = threshold
    ad.partial_fit(
        Xb,
        yb,
        layers_to_update=[layer],
        ewc_penalty=ewc_penalty,
        fisher=fisher,
        theta_star=theta_star,
        max_epochs=epochs,
        finetune_lr=1e-4,
    )
    post_a = _f1(ad, Xa_test, ya_test)
    post_b = _f1(ad, Xb_test, yb_test)
    del ad
    return (pre_a_f1 - post_a), post_b


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--task-a", type=int, nargs=2, default=[0, 1])
    p.add_argument("--task-b", type=int, nargs=2, default=[2, 3])
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--ewc-penalties", type=float, nargs="+", default=[0.0, 100.0, 1000.0, 5000.0])
    p.add_argument("--out", default="experiments/exp_e3_causal_targeting.json")
    args = p.parse_args(argv)

    log.info("e3_start", seeds=args.seeds, hidden=args.hidden, ewc=args.ewc_penalties)

    train_tasks = load_split_mnist(split="train", tasks=(tuple(args.task_a), tuple(args.task_b)))
    test_tasks = load_split_mnist(split="test", tasks=(tuple(args.task_a), tuple(args.task_b)))
    ta_tr, tb_tr = train_tasks
    ta_te, tb_te = test_tasks

    def _new_adapter() -> FraudNet:
        return FraudNet(
            FraudNetConfig(
                input_dim=784,
                hidden_dims=list(args.hidden),
                dropout=0.2,
                lr=1e-3,
                batch_size=128,
                max_epochs=15,
                early_stopping_patience=4,
                class_weighted=True,
            )
        )

    penalties = [float(x) for x in args.ewc_penalties]
    # forget[pen][method] = list over seeds ; same for taskb
    forget = {pen: {m: [] for m in TARGETING_METHODS} for pen in penalties}
    taskb = {pen: {m: [] for m in TARGETING_METHODS} for pen in penalties}
    chosen = {m: [] for m in ["random", "psi", "gradient", "cdag"]}
    pre_a_all = []

    for seed in range(args.seeds):
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        adapter = _new_adapter()
        adapter.fit(ta_tr.X, ta_tr.y, X_val=ta_te.X[:1500], y_val=ta_te.y[:1500])
        baseline_state = adapter.state_dict()
        threshold = adapter.decision_threshold
        pre_a = _f1(adapter, ta_te.X, ta_te.y)
        pre_a_all.append(pre_a)

        hidden_layers = list(adapter.tap_layer_names)  # e.g. [layer1, layer2, layer3]

        # theta* and Fisher on Task A (identical EWC target for every arm).
        theta_star = {k: v.clone() for k, v in baseline_state.items()}
        fisher = adapter.compute_fisher(ta_tr.X, ta_tr.y)

        # Layer choice per method is independent of the EWC penalty — pick once.
        picks = {}
        for m in ["random", "psi", "gradient", "cdag"]:
            picks[m] = _select_layer(
                m,
                adapter,
                hidden_layers,
                theta_star,
                fisher,
                ta_tr.X,
                ta_tr.y,
                tb_tr.X,
                tb_tr.y,
                rng,
            )
            chosen[m].append(picks[m])

        # For each penalty, run every hidden layer ONCE, then map methods to their pick.
        for pen in penalties:
            layer_res = {}
            for layer in hidden_layers:
                layer_res[layer] = _partial_retrain_forget(
                    adapter,
                    baseline_state,
                    threshold,
                    hidden_layers,
                    layer,
                    fisher,
                    theta_star,
                    tb_tr.X,
                    tb_tr.y,
                    ta_te.X,
                    ta_te.y,
                    tb_te.X,
                    tb_te.y,
                    pre_a,
                    pen,
                    args.finetune_epochs,
                )
            for m in ["random", "psi", "gradient", "cdag"]:
                f, b = layer_res[picks[m]]
                forget[pen][m].append(f)
                taskb[pen][m].append(b)
            # oracle = layer with lowest forgetting at this penalty
            best = min(layer_res.values(), key=lambda t: t[0])
            forget[pen]["oracle"].append(best[0])
            taskb[pen]["oracle"].append(best[1])

        print(
            f"[seed {seed}] pre_A={pre_a:.3f} picks=" + ",".join(f"{m}:{picks[m]}" for m in picks)
        )

        del adapter, fisher, theta_star
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _wilcoxon_less(a, b):
        try:
            stat, pv = sci_stats.wilcoxon(a, b, alternative="less")
            return {"stat": float(stat), "p": float(pv), "n": len(a)}
        except ValueError as e:
            return {"stat": None, "p": None, "err": str(e), "n": len(a)}

    per_penalty = {}
    for pen in penalties:
        per_penalty[str(pen)] = {
            "forgetting": {
                m: {
                    "mean": _mean_std(forget[pen][m])[0],
                    "std": _mean_std(forget[pen][m])[1],
                    "values": forget[pen][m],
                }
                for m in TARGETING_METHODS
            },
            "task_b_f1": {m: _mean_std(taskb[pen][m]) for m in TARGETING_METHODS},
            "wilcoxon_cdag_forgets_less": {
                "vs_random": _wilcoxon_less(forget[pen]["cdag"], forget[pen]["random"]),
                "vs_psi": _wilcoxon_less(forget[pen]["cdag"], forget[pen]["psi"]),
                "vs_gradient": _wilcoxon_less(forget[pen]["cdag"], forget[pen]["gradient"]),
            },
        }

    summary = {
        "experiment": "E3_causal_targeting",
        "seeds": args.seeds,
        "hidden": args.hidden,
        "ewc_penalties": penalties,
        "finetune_epochs": args.finetune_epochs,
        "pre_task_a_f1": _mean_std(pre_a_all),
        "layer_choices": {
            m: {layer: chosen[m].count(layer) for layer in set(chosen[m])} for m in chosen
        },
        "per_penalty": per_penalty,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E3 Causal-targeting ablation (Split-MNIST, identical EWC per row) ===")
    print(f"Pre Task-A F1: {summary['pre_task_a_f1'][0]:.4f} +/- {summary['pre_task_a_f1'][1]:.4f}")
    print("Layer picks: " + " | ".join(f"{m}:{summary['layer_choices'][m]}" for m in chosen))
    for pen in penalties:
        pp = per_penalty[str(pen)]
        print(f"\n--- EWC penalty = {pen} ---")
        print(f"{'method':<10}{'forgetting':<22}{'taskB_F1':<12}{'p(cdag<)':<10}")
        for m in TARGETING_METHODS:
            fm, fs = pp["forgetting"][m]["mean"], pp["forgetting"][m]["std"]
            tb = pp["task_b_f1"][m][0]
            pcell = ""
            if m in ("random", "psi", "gradient"):
                pcell = f"{pp['wilcoxon_cdag_forgets_less']['vs_' + m].get('p')}"
            print(f"{m:<10}{fm:+.4f} +/- {fs:.4f}    {tb:.4f}      {pcell}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
