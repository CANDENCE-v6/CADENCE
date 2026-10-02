"""E2 — Is PPO necessary? (reviewer #1).

The action space is only {no-op, partial, full}. A reviewer rightly asks: why
RL for a 3-action decision? This runs the SAME sandbox env / reward / cost /
SLA under five policies and compares them head-to-head:

  * no-op        : always action 0 (lower-bound floor)
  * always-partial: always action 1
  * always-full   : always action 2 (reactive-full analogue)
  * greedy-rule   : the hand rule (below-SLA -> partial if attribution is
                    concentrated else full; else no-op) -- a myopic non-learned policy
  * ppo           : the learned Stage-1 per-seed policy

If PPO does not beat the greedy rule on the reward / F1-vs-cost trade-off,
that is the honest answer: PPO is not necessary here. Reported, not hidden.

Usage:
    python -m benchmarks.exp_e2_ppo_necessity --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_contested_sla_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import GNNConfig, GNNResponsibilityScorer
from cadence.carbon.model import GridProfile, HardwareProfile
from cadence.common.config import load_config
from cadence.common.device import get_device, log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig

log = get_logger("cadence.benchmarks.e2")


def _rule(obs, sla):
    margin = float(obs[13])
    conc = float(obs[14])
    if margin <= 1e-3:
        return 0
    return 1 if conc > 0.5 else 2


def _mean(v):
    return float(statistics.fmean(v)) if v else 0.0


def _std(v):
    return float(statistics.pstdev(v)) if len(v) > 1 else 0.0


def evaluate_policy(action_fn, env, n_windows, sla):
    """Step the env under a policy; return per-episode metrics."""
    obs, _ = env.reset(seed=0)
    f1s, rewards, costs = [], [], []
    sla_viol = 0
    acts = {0: 0, 1: 0, 2: 0}
    for _ in range(n_windows):
        a = int(action_fn(obs))
        acts[a] += 1
        obs, reward, term, trunc, info = env.step(a)
        post = float(info.get("post_f1", env._current_f1))
        f1s.append(post)
        rewards.append(float(reward))
        costs.append(float(info.get("cost_gpu_hr", 0.0)))
        if post < sla:
            sla_viol += 1
        if term or trunc:
            break
    return {
        "mean_f1": _mean(f1s),
        "min_f1": min(f1s) if f1s else 0.0,
        "total_gpu_hr": float(sum(costs)),
        "sla_violations": sla_viol,
        "mean_reward": _mean(rewards),
        "actions": acts,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-windows", type=int, default=6)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--eval-window-size", type=int, default=2048)
    p.add_argument("--sla", type=float, default=0.80)
    p.add_argument("--out", default="experiments/exp_e2_ppo_necessity.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    log_device_info(dev)

    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_pre = int(0.70 * n_train)
    X_pre, y_pre = ds.X_train[idx[:n_pre]], ds.y_train[idx[:n_pre]]
    X_stream = ds.X_train[idx[n_pre : n_pre + 25000]]
    y_stream = ds.y_train[idx[n_pre : n_pre + 25000]]

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

    scenarios = build_contested_sla_scenarios(ds.feature_names)
    sandbox_cfg = SandboxConfig(
        window_size=args.eval_window_size,
        max_windows_per_episode=args.n_windows,
        finetune_epochs=3,
        fullretrain_epochs=8,
        replay_buffer_size=5000,
        sla_target=args.sla,
        ewc_penalty=1000.0,
        ewc_fisher_sample_size=2000,
        ewc_cache_fisher=True,
        hardware=HardwareProfile(),
        grid=GridProfile(),
    )

    # Load PPO once per seed (per-seed Stage-1 policies preferred).
    def load_ppo(seed):
        for path in (
            f"experiments/rso_ppo_phase_a_seed{seed}.zip",
            "experiments/rso_ppo_phase_a.zip",
        ):
            if os.path.exists(path):
                try:
                    from stable_baselines3 import PPO

                    m = PPO.load(path, custom_objects={"lr_schedule": lambda _: 0.0})
                    return m, path
                except Exception as e:  # noqa: BLE001
                    log.warning("ppo_load_failed", path=path, err=str(e))
        return None, None

    policies = ["no_op", "always_partial", "always_full", "greedy_rule", "ppo"]
    results = {pol: [] for pol in policies}

    for scenario in scenarios:
        for seed in range(args.seeds):
            set_global_seed(seed)
            stream = make_baseline_stream(X_stream, y_stream, scenario, seed=seed)
            sX, sY = stream[0], stream[1]
            ppo, ppo_path = load_ppo(seed)

            def make_env(sX=sX, sY=sY):
                fresh = adapter.clone()
                fresh.load_state_dict({k: v.clone() for k, v in baseline_state.items()})
                fresh.decision_threshold = baseline_threshold
                return RetrainingSandboxEnv(
                    initial_adapter=fresh,
                    historical_X=X_pre,
                    historical_y=y_pre,
                    drifted_stream_X=sX,
                    drifted_stream_y=sY,
                    responsibility_scorer=scorer,
                    cfg=sandbox_cfg,
                )

            action_fns = {
                "no_op": lambda o: 0,
                "always_partial": lambda o: 1,
                "always_full": lambda o: 2,
                "greedy_rule": lambda o: _rule(o, args.sla),
            }
            if ppo is not None:
                action_fns["ppo"] = lambda o, _m=ppo: int(
                    np.asarray(_m.predict(o, deterministic=True)[0]).flatten()[0]
                )

            for pol in policies:
                if pol not in action_fns:
                    continue
                out = evaluate_policy(action_fns[pol], make_env(), args.n_windows, args.sla)
                out.update({"scenario": scenario.name, "seed": seed})
                results[pol].append(out)
            log.info("e2_cell_done", scenario=scenario.name, seed=seed, ppo=ppo_path)

    def agg(pol):
        outs = results[pol]
        if not outs:
            return None
        return {
            "mean_f1": _mean([o["mean_f1"] for o in outs]),
            "mean_f1_std": _std([o["mean_f1"] for o in outs]),
            "total_gpu_hr": _mean([o["total_gpu_hr"] for o in outs]),
            "sla_violations": _mean([o["sla_violations"] for o in outs]),
            "mean_reward": _mean([o["mean_reward"] for o in outs]),
            "n": len(outs),
        }

    summary = {
        "experiment": "E2_ppo_necessity",
        "sla": args.sla,
        "scenarios": [s.name for s in scenarios],
        "seeds": args.seeds,
        "aggregate": {pol: agg(pol) for pol in policies},
        "raw": results,
        "config": vars(args),
    }

    # Paired: PPO vs greedy on mean_reward and mean_f1 (same (scenario,seed) order).
    try:
        from scipy import stats as sci

        ppo_r = [o["mean_reward"] for o in results["ppo"]]
        gr_r = [o["mean_reward"] for o in results["greedy_rule"]]
        if len(ppo_r) == len(gr_r) and len(ppo_r) >= 2:
            _, p_rew = sci.wilcoxon(ppo_r, gr_r)
            summary["wilcoxon_ppo_vs_greedy_reward_p"] = float(p_rew)
    except Exception as e:  # noqa: BLE001
        summary["wilcoxon_err"] = str(e)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== E2: PPO necessity (same env/reward/cost/SLA, contested scenarios) ===")
    print(f"{'policy':<16}{'mean_F1':<12}{'gpu_hr':<14}{'SLA_viol':<10}{'reward':<10}")
    for pol in policies:
        a = summary["aggregate"][pol]
        if a is None:
            print(f"{pol:<16}(no PPO checkpoint loaded)")
            continue
        print(
            f"{pol:<16}{a['mean_f1']:.4f}      {a['total_gpu_hr']:.3e}   {a['sla_violations']:.2f}      {a['mean_reward']:+.4f}"
        )
    if "wilcoxon_ppo_vs_greedy_reward_p" in summary:
        print(
            f"\nWilcoxon PPO vs greedy (reward): p={summary['wilcoxon_ppo_vs_greedy_reward_p']:.4f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
