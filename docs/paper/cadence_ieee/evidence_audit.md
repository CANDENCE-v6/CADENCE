# CADENCE — Reviewer-Priority Evidence Audit

Honest map of the 20 reviewer priorities (+ the E1–E10 experiment list) to
**what we actually have (with the measured proof)** vs **what we do not**.
"Proof" = an entry in `docs/results.md` (the measured ledger). Nothing here
is aspirational; unrun experiments are marked ❌ and appear in the paper's
Limitations section, never as results.

Legend: ✅ HAVE (measured) · 🟡 PARTIAL · ❌ DON'T HAVE (honest gap) ·
✍️ WRITING (formalization/scholarship, no experiment needed)

| # | Reviewer ask | Status | Proof / where in paper |
|---|---|---|---|
| P0 / E2 | Causal attribution generalizes beyond synthetic drift (train mech A/B/C → test D/E/F) | ❌ | No OOD-mechanism experiment. Robustness edge-dropout (R-Gate-F) is a *degraded-signal* proxy, not OOD. → **Limitations §8, bullet 1** |
| P1 | Related-work **comparison matrix**, one citation per row | ✅✍️ | Built from `docs/prior_art.md`. → **Table I (§2)** |
| P2 | Define what "causal" means (recovery vs attribution vs counterfactual) | ✅✍️ | Formalized as ranking/prediction of `R_i`, *not* graph recovery. → **§3.1, Eq. (1)** |
| P3 / E3 | Surrogate accuracy + **sample-efficiency** | ✅ **DONE** | R²=0.888 held-out (R-Gate-C) + R-E4 curve: R²≈0.9 by ~100 interventions. Seen/unseen split still open. → **§6.2** |
| P4 / E4 | **PPO necessity** vs greedy/always-*/bandit | ✅ **DONE (null)** | R-E2: PPO vs no-op/always-partial/always-full/greedy on same env — **competitive but NOT superior** (p=0.29 vs greedy). → **§6 subsec + §8** |
| P5 | **CMDP formalization** (s, a, P, c, g, objective) | ✅✍️ | Full CMDP + augmented-Lagrangian. → **§3.2, Eq. (2)–(4)** |
| P6 / E1 | Stronger attribution baselines + oracles | 🟡 | 3 scorers measured: PSI, structural, GNN (R-Gate-B-n10). Missing: gradient/permutation attribution, oracle. → **§6.1; gap in §8** |
| P7 | H1 per-scenario / easy-hard breakdown (anti-ceiling) | ✅ | R-Gate-B-n10-subset (EASY vs HARD). → **§6.1, Fig. 2, Table II** |
| P8 | Confidence intervals, not just p-values | 🟡 | 95% bootstrap CI on Elec2 (R-Gate-E-verified); std elsewhere. → **§6.3 (CI shown)** |
| P9 / E5 | Full Pareto frontier (F1 vs compute over w) | 🟡 | R-4b 3×3 sweep (bang-bang) + Elec2 point (R-Gate-E). Not a clean w-swept paper-scale curve. → **§6.3; gap in §8** |
| P10 | H2 on >1 real dataset | 🟡 | Elec2 = real result; Airlines = masked drift (honest null), R-Gate-E. → **§6.3 + failure §6.9** |
| P11 / E7 | H3 isolates **causal targeting** (random/PSI/CDAG partial, all +EWC) | ✅ **DONE (null)** | R-E3: ran it (n=10, EWC swept). CDAG targeting is **tied** with random/gradient on forgetting (p≈0.5–0.9); forgetting win is EWC's, not targeting's. → **§6.4 + §8** |
| P12 | Model generality phrased precisely | ✅ | R-Gate-F (tree), -text, -tree-sla045. → **§6.5, Table IV** |
| P13 | Carbon: measured kWh/CO₂e | ❌ | Closed-form hardware model only (R-2/R-3); no metered kWh. → **Softened to compute; §8, bullet 7** |
| P14 | "Production" claim / rename | ✅ | Retitled **"Continually Deployed"**; scope stated. → **Title + §1** |
| P15 / E6 | **Total CADENCE overhead** (net compute saved) | ✅ **DONE (bounded)** | R-E1: measured. CDAG=374ms/window dominates; **net −78.5% on tiny FraudNet** (overhead eats the saving); break-even full-retrain ≈1.3s → net-positive only on large models. → **§6.4 (new subsec) + §8** |
| P16 / E10 | Graph-construction cost vs scale | ✅ **DONE (bound)** | R-E5: 0.30s@30 → 5.2s@100 → 127s@300 nodes, ≈O(n²·⁶); impractical >100 nodes. → **§8 (measured limitation)** |
| P17 | "Online" is misleading | ✅✍️ | Reframed as **"online-triggered"** rolling-window rebuild-on-alert. → **§8, bullet 8** |
| P18 | Concept-flip limitation → quantitative | 🟡 | v14_concept_shift collapses to F1≈0.01 (R-4, R-Gate-C); not a magnitude sweep. → **§6.9 failure (2)** |
| P19 | Diffuse-responsibility guard experiment | ❌ | Guard exists in code; no concentrated/diffuse/unrelated experiment. → **§8 (planned)** |
| P20 | Failure cases | ✅ | 4 measured failures (PSI saturation, concept flip, masked drift, single-task H3). → **§6.9** |

## The three headline hypotheses — exact status

| Hyp. | Claim | Verdict | Proof |
|---|---|---|---|
| **H1** | Causal/GNN attribution beats correlational PSI | ✅ **SUPPORTED on contested subset** (MRR & AUROC, p<0.001 vs PSI, n=10) | R-Gate-B-n10, R-Gate-B-n10-subset |
| **H2** | RSO cuts retraining compute at ≈equal quality | 🟡 **PRACTICAL** (Elec2: 69.1% compute for −1.66 F1, CI [−.025,−.009], n=3, rule policy). **STRONG form pending Stage-2 Colab** | R-Gate-E, R-Gate-E-verified; policy-health = R-Gate-A-stage1-pass |
| **H3** | Targeted partial retrain reduces forgetting | ✅ **SUPPORTED on multi-task** (MNIST, −0.03 vs +0.40, p<0.001, n=10, incl. fair replay baseline); ❌ **negative on single-task Fraud** (reported) | R-Gate-F-mnist-n10, R-Gate-D |

## What makes this defensible to a strict reviewer

1. **Every claim is scoped to its evidence.** H1/H3 strong where proven; H2
   explicitly PRACTICAL not STRONG. No "beats on average" over-claims.
2. **Pre-registration + honest negatives** (single-task H3, bang-bang PPO,
   masked-drift Airlines/Yelp, unrecoverable concept flip) — reviewers
   trust papers that report their own failures.
3. **A policy-health diagnostic** (Stage-1, 7/8 gates) — proves the RL
   policy did not silently collapse, a check most RL-for-systems papers skip.
4. **The Limitations section names every gap above before the reviewer can**,
   each with a pre-registered confirmatory experiment.

## Update (2026-09-05): E1 and E3 now run — both returned honest boundaries

- **E1 (total overhead) — DONE.** The load-bearing experiment. Result:
  CDAG construction (374 ms/window) dominates overhead; on the tiny FraudNet
  CADENCE is **net −78.5%** (overhead eats the 44.7% retrain saving). Net
  saving is positive only when a full retrain costs **≳1.3 s** (i.e. real
  models). This bounds the H2 claim honestly rather than invalidating it, and
  is now a paper subsection (§6.4) + a resolved limitation.
- **E3 (causal-targeting) — DONE (null).** Under identical EWC, CDAG targeting
  is **tied** with random/gradient on forgetting; the H3 win is EWC's. Paper
  now claims only EWC-partial < full, and reports the null.

## Remaining load-bearing before a top main-track submission

1. **E4 / #7 — H2 at n≥10 with the learned policy** (Stage 2). Blocked on
   compute; run locally on the 4070 (reduced-scale pre-reg amendment) since
   no A100.
2. **E2 / #1 — PPO necessity** (greedy/bandit/supervised vs PPO). Not yet run.
3. **E5 / #4 — OOD causal generalization.** Strengthens the causal claim most.

The overhead result (E1) also raises a concrete engineering priority the paper
now names: build the CDAG only on trigger-alerts and cache/optimize discovery,
so the framework is net-positive on a wider range of model sizes.
