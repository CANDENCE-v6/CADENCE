# CADENCE — Patent Prosecution Strategy Memo

**Re:** *System and Method for Causal Attribution-Based Selective Retraining of Deployed Machine-Learning Models*
**Source of record:** `CADENCE-Patent-Invention-Disclosure.docx` (the version present in this repository; the prompt referenced a `(1)` copy that is not present — confirm they are identical before relying on this memo).
**Scope of this memo:** patentability strengthening, claim architecture, inventive-step defense, disclosure-gap identification. This is drafting/strategy analysis, **not** a legal opinion; a registered patent agent must finalize claims and complete the prior-art search. Every technical statement below is taken from the disclosure; where a fact needed for patentability is absent it is marked **[DISCLOSURE GAP — DO NOT INVENT]**.

---

## PART 1 — EXECUTIVE PATENTABILITY DIAGNOSIS

**Current weaknesses.**
1. Each named building block (drift detection, causal root-cause, selective weight update, EWC forgetting control, cost-aware retraining, shadow/canary, GNN attribution) is independently known — including in granted patents (Dell US11928011B2 does causal drift root-cause + data tagging + retrain + A/B/canary). Novelty therefore *cannot* rest on the component list.
2. The full-retrain predicted-recovery term `R̂_full` is used by the optimizer but its derivation is not disclosed (Critical Issue #6). This is a live §112/enablement and clarity exposure.
3. Model-agnosticism (neural / GBDT / linear / transformer) is asserted, but only the feed-forward-neural embodiment is enabled to best-mode depth.
4. "Causal" language risks over-claiming; the spec itself disclaims unique-graph identification.
5. Prior-art analysis is preliminary (abstract-level); no reference has been read at claim level, so no anticipation position is verified.

**Strongest inventive core.** The **surrogate-amortized, responsibility×recovery-coupled selection of a concrete parameter-scope repair, executed as a constrained no-op/partial/full decision inside a validated, reversible closed loop.** In one line: *predict each candidate repair's benefit cheaply (no per-candidate retrain), fuse it with node responsibility to pick which subnetwork to retrain, and gate the update by cost/SLA feasibility with isolated validation and rollback.*

**Biggest obviousness attack.** Dell US11928011B2 (causal drift root-cause + retrain + canary) **+** Nvidia EP3745318A1 (selective weight update) **+** EWC (Kirkpatrick 2017) **+** cost-aware retraining. The examiner will argue "known causal drift diagnosis, applied to known selective/forgetting-safe retraining, under known cost budgeting."

**Biggest support problem.** `R̂_full` derivation (Issue #6) and multi-model-family enablement (Issue #11).

**Biggest claim-drafting problem.** The independent claims must make the **interaction** — responsibility×recovery → mapped parameter scope → feasibility-gated action — the load-bearing limitation, not a post-hoc list. The earlier optimizer inconsistency (least-cost-feasible vs objective `J`) is reconciled in Part 5.

**Recommended filing strategy.** File an **India provisional first** to secure priority while the disclosure gaps (esp. `R̂_full`, model-family enablement) are closed, then a **PCT** within 12 months built on the reconciled claim set in Part 5. Do **not** file the current document as a complete specification.

---

## PART 2 — INVENTIVE CORE

The patentable core is a **closed-loop model-maintenance control architecture** in which a diagnosis of degradation is converted, via a *learned prediction of repair benefit*, into a *concrete, minimal, reversible modification of internal model parameters*. Two couplings distinguish it from the prior art. First, a **single graph unifies external input-feature behavior and internal model-state behavior**, so responsibility is scored over both the data and the model's own activations. Second — and most importantly — **a learned recovery surrogate predicts the performance obtainable from each candidate repair without running a separate retrain per candidate, and this predicted recovery is fused multiplicatively with node responsibility (`i* = argmax_i r_i·R̂_i`) to choose *which subnetwork* to retrain.** The chosen node is mapped to a concrete trainable parameter set, and the system then selects the least-cost action (no-op / partial / full) whose *predicted* post-repair performance satisfies an SLA, updates only that parameter set under a forgetting penalty, validates in an isolated path, and promotes or rolls back.

The non-obviousness lives in the **glue**: prior causal-drift systems tag *data* and retrain; prior selective-update systems pick weights for *training efficiency*; prior cost-aware systems decide *timing*. None predict per-candidate repair benefit and use it, jointly with responsibility, to select a *parameter scope* under a feasibility constraint.

**CORE LIMITATIONS (must appear in an independent claim).**
- Unified external-feature + internal-model-state attribution graph.
- Per-node responsibility value from graph message passing.
- Learned recovery surrogate producing predicted recovery **without a per-candidate retrain**.
- Target selection **coupling responsibility and predicted recovery** (`argmax r_i·R̂_i`).
- Mapping target node → concrete trainable parameter set.
- Constrained action selection (no-op/partial/full) by **least-cost predicted-feasible action** under an SLA.
- Selective update of only the mapped parameter set with the remainder frozen, under a parameter-deviation (importance-weighted) penalty.

**SUPPORTING LIMITATIONS (strong dependents; optionally in a closed-loop independent claim).**
- Isolated shadow/canary validation with promotion/rollback.
- Realized-recovery recalibration of the surrogate.
- Attribution-concentration escalation guard.
- PC-skeleton + continuous-relaxation graph construction; clustered activation nodes.

**OPTIONAL LIMITATIONS (narrowing dependents).**
- Specific optimizer instances (PPO/bandit/greedy); dual-variable SLA enforcement.
- OOD guard; trigger AND/OR logic; rolling-window rebuild; model-family specializations; telemetry.

---

## PART 3 — REVISED INVENTION SUMMARY

A computer-implemented system maintains a deployed machine-learning model that continues to serve predictions. Upon detecting degradation of a performance metric, the system builds a single attribution graph whose nodes represent both external input-feature statistics and internal model-state representations of the deployed model, together with a performance node, and scores each node with a graph message-passing model to obtain a node embedding and a non-negative responsibility value. A learned recovery surrogate, operating on the node embeddings, predicts the post-repair performance — and hence the predicted recovery — of each of several candidate repairs, **without executing a separate retrain of the deployed model for each candidate**. The system selects a target node that maximizes the product of responsibility and predicted recovery, maps that node to a concrete trainable parameter set of a subnetwork, and selects a maintenance action from {no-operation, partial retrain of the mapped set, full retrain}: among the actions whose *predicted* post-repair performance satisfies a service-level constraint, it chooses the one of least estimated compute cost. For a partial retrain it updates only the mapped parameter set, freezing the remainder, under an importance-weighted parameter-deviation penalty that limits loss of previously retained competence. The resulting model is validated in a path isolated from live serving and is promoted or rolled back; realized post-repair performance is fed back to recalibrate the recovery surrogate. The architecture thereby reduces the parameters updated and the compute expended per adaptation while protecting prior competence and the live serving path.

---

## PART 4 — REVISED DETAILED DESCRIPTION (deltas only)

The disclosure's §3.4.1–§3.4.14 are largely adequate after the consistency fixes already applied (recovery defined once as `R̂_i = P̂_i − Perf_current`; target rule `i* = argmax r_i·R̂_i` consistent across spec/algorithm/claims; least-cost-feasible selection). The following **deltas** are required.

**(4.1) Recovery surrogate — retain and make prominent.** Keep the operational definitions: candidate intervention = apply mapping Φ to a node → obtain a target parameter set → perform a *bounded* partial retrain of that set; sandbox tuple = (graph, candidate node, pre-repair performance, measured post-repair performance, mechanism); surrogate `g_θ(e_i,[Perf_current,severity]) → P̂_i`; predicted recovery `R̂_i = P̂_i − Perf_current`; trained by MSE to measured post-repair performance; validated on a held-out (preferably mechanism-held-out) split; uncertainty by ensemble/MC-dropout; OOD guard by embedding distance vs a percentile threshold; recalibration from realized post-repair performance. **Amortization statement to add explicitly:** "At decision time, predicted recoveries for all candidate nodes are obtained by a single forward pass of the graph message-passing model and the surrogate, and no candidate requires a separate retrain of the deployed model; the sandbox retrains occur only during offline surrogate training."

**(4.2) `R̂_full` — [DISCLOSURE GAP — DO NOT INVENT].** The optimizer uses `dPerf(full) = R̂_full`, but the disclosure does not state how the predicted full-retrain recovery is obtained. The claims must not require an undisclosed mechanism. **Required inventor input (choose the one actually implemented):** (a) `R̂_full` is a second surrogate head / separate surrogate trained on sandbox *full-retrain* tuples; or (b) `R̂_full` is a fixed or historical constant estimate (e.g., a running mean of realized full-retrain recoveries); or (c) the full-retrain action is treated as always-predicted-feasible (upper-bounding recovery) so `R̂_full` is not predicted at all. Until supplied, Part 5 Claim 1 is drafted to **not depend** on `R̂_full` (the full-retrain action is the fallback when no partial action is predicted-feasible), and `R̂_full` appears only in a dependent claim conditioned on inventor confirmation.

**(4.3) Model-family enablement — partial [DISCLOSURE GAP].** The neural embodiment is enabled to best-mode depth. For GBDT ("leaf/path co-activation statistics"), linear ("per-feature contribution statistics"), and transformer ("per-head token-representation clusters"), the disclosure asserts the internal-state analogue but does **not** enable the node→trainable-parameter-set mapping or the selective/forgetting-safe update for these families. **Required inventor input:** for each non-neural family actually reduced to practice, the concrete (i) internal-state feature, (ii) node→component→parameter mapping, and (iii) partial-update mechanism, or an explicit statement that the family degrades to full retrain (already disclosed for trees/linear). Draft independent claims around the **neural** embodiment plus a genuinely-supported model-agnostic monitoring/attribution layer; keep other families as dependents flagged to enablement.

**(4.4) "Causal" terminology — scope consistently.** Adopt throughout: "causal-**structure-informed** attribution graph" and "responsibility value"; reserve "counterfactual" for the surrogate, defined operationally as *predicting the model's post-repair performance under a simulated repair intervention, learned from sandbox interventions* — **not** as identification of a causal effect in the field. Remove any phrasing implying recovery of a unique/true causal DAG (the spec already disclaims this in §3.4.3; make the claims match by using "causal-structure-informed procedure").

---

## PART 5 — REVISED CLAIM SET (proposed; ~24 claims)

> Draft for finalization by a registered patent agent. Independent claims 1/2/3 are technically consistent and share the same inventive architecture. `R̂_full` is deliberately kept out of the independent claims (see 4.2).

**Claim 1 (method).** A computer-implemented method for maintaining a deployed machine-learning model while the deployed model continues to serve predictions, the method comprising:
(a) receiving, through an adapter interface to the deployed model, external input-feature statistics of an input window and internal model-state representations obtained from the deployed model for the input window;
(b) detecting a degradation of a performance metric of the deployed model;
(c) responsive to the detecting, constructing a single attribution graph comprising external-feature nodes derived from the external input-feature statistics, internal-state nodes derived from the internal model-state representations, and a performance node, and determining weighted directed relationships among the nodes by a causal-structure-informed procedure;
(d) generating, by a graph message-passing model applied to the attribution graph, for each of a plurality of the nodes, a node embedding and a non-negative responsibility value quantifying a contribution of the node to the detected degradation;
(e) for each of a plurality of candidate repairs, each candidate repair associated with a respective node, generating, by a learned recovery surrogate applied to the node embedding of the respective node, a predicted post-repair performance and deriving a predicted recovery therefrom, wherein the predicted recoveries for the plurality of candidate repairs are generated without executing a separate retrain of the deployed model for each candidate repair;
(f) selecting a target node as a node that maximizes a product of the responsibility value of the node and the predicted recovery of its candidate repair;
(g) mapping the target node to a target parameter set of a subnetwork of the deployed model;
(h) selecting a maintenance action from a maintenance action space comprising a no-operation action, a partial-retrain action of the target parameter set, and a full-retrain action, wherein each action of the maintenance action space has a predicted post-repair performance, an action is predicted-feasible when its predicted post-repair performance satisfies a service-level constraint, and the selected maintenance action is an action of least estimated compute cost among the predicted-feasible actions, the estimated compute cost of an action being a function of a number of parameters updated by the action, and the full-retrain action being selected when no other action is predicted-feasible;
(i) responsive to selecting the partial-retrain action, updating only the target parameter set while freezing the remaining parameters of the deployed model, the updating being regularized by a parameter-deviation penalty that weights a deviation of a parameter from a stored prior value of the parameter by a per-parameter importance estimate; and
(j) validating a resulting model in an evaluation path isolated from a live serving path of the deployed model, and promoting the resulting model when it satisfies the service-level constraint and otherwise restoring a prior snapshot of the deployed model.

**Claim 2 (system).** A system for maintaining a deployed machine-learning model that continues to serve predictions, the system comprising one or more processors and memory storing instructions that configure the system to provide: an adapter interface exposing a scoring operation, an activation-tap operation returning internal model-state representations, a partial-fit operation accepting a target-parameter-set descriptor, a full-fit operation, and snapshot and restore operations; a graph generator configured to construct a single attribution graph comprising external-feature nodes, internal-state nodes derived from the activation-tap operation, and a performance node; a responsibility scorer comprising a graph message-passing model configured to output, per node, a node embedding and a non-negative responsibility value; a recovery surrogate configured to output, from a node embedding and without a per-candidate retrain, a predicted post-repair performance from which a predicted recovery is derived; a target selector configured to select a target node maximizing a product of a node's responsibility value and predicted recovery; a mapper configured to map the target node to a target parameter set via the partial-fit descriptor; a retraining-scope optimizer configured to select, from a no-operation action, a partial-retrain action of the target parameter set and a full-retrain action, a least-cost action among those whose predicted post-repair performance satisfies a service-level constraint; a forgetting-constrained executor configured to update, via the partial-fit operation, only the target parameter set while freezing remaining parameters under a parameter-deviation penalty weighted by a per-parameter importance estimate; a validation module configured to evaluate a resulting model in a path isolated from a live serving path; and a model registry configured to promote the resulting model or restore a prior snapshot.

**Claim 3 (CRM).** A non-transitory computer-readable medium storing instructions that, when executed by one or more processors, cause the processors to: construct a single attribution graph over external-feature nodes and internal model-state nodes of a deployed model and a performance node; generate, by a graph message-passing model, a per-node responsibility value and node embedding; generate, by a recovery surrogate applied to the node embeddings and without a per-candidate retrain, a per-candidate predicted recovery; select a target node maximizing a product of responsibility value and predicted recovery; map the target node to a target parameter set of a subnetwork; select, from a no-operation, a partial-retrain of the target parameter set, and a full-retrain action, a least-cost action among those whose predicted post-repair performance satisfies a service-level constraint; for the partial-retrain action, update only the target parameter set while freezing remaining parameters under a per-parameter-importance-weighted deviation penalty; and validate the resulting model in an isolated path and promote it or restore a prior snapshot.

**Dependent claims.**
4. The method of claim 1, wherein the internal-state nodes are derived by clustering internal post-activation representations of one or more tapped layers of the deployed model.
5. The method of claim 1, wherein constructing the graph comprises estimating a skeleton by a constraint-based conditional-independence procedure and estimating edge weights by a continuous acyclicity-regularized optimization, the edge weights being masked by the skeleton.
6. The method of claim 1, wherein the responsibility value of a node is generated by neighborhood message passing such that the responsibility value is determined even in an absence of a direct edge from the node to the performance node.
7. The method of claim 1, wherein the recovery surrogate maps the node embedding and a context vector comprising the current performance and a severity to the predicted post-repair performance.
8. The method of claim 7, wherein the recovery surrogate is trained on tuples generated by injecting a drift, performing a candidate repair mapped from a sampled node, and recording a measured post-repair performance, and the predicted recoveries at decision time are obtained by a single forward pass over the plurality of candidate nodes.
9. The method of claim 1, further comprising estimating an uncertainty of the predicted recovery, detecting that a candidate repair lies outside a training distribution of the recovery surrogate by an embedding-distance test, and biasing the selection away from the candidate repair responsive thereto.
10. The method of claim 1, wherein the service-level constraint is enforced by a dual variable updated toward a target service-level-violation rate, and the actions are ranked by an objective that rewards predicted performance change and penalizes estimated compute cost.
11. The method of claim 1, further comprising computing an attribution-concentration measure equal to a ratio of a maximum of the responsibility values to a sum of the responsibility values, and converting the partial-retrain action to the full-retrain action when the measure is below a threshold.
12. The method of claim 1, wherein mapping the target node comprises mapping an internal-state node to a component from which its representation was tapped and mapping the component to the target parameter set, and mapping a feature node to an input-facing component.
13. The method of claim 12, further comprising, when a plurality of nodes are responsible, forming the target parameter set as a union of the parameter sets of the plurality of nodes, and escalating to the full-retrain action when the union spans substantially all parameters of the deployed model.
14. The method of claim 1, wherein the per-parameter importance estimate is a diagonal Fisher information computed over retained-competence data and stored with a model snapshot, and updating comprises masking a gradient of any parameter outside the target parameter set to zero.
15. The method of claim 1, further comprising updating the recovery surrogate using a realized post-repair performance of the promoted model.
16. The method of claim 1, further comprising emitting a machine-readable telemetry record comprising the target node, an attribution-concentration measure, the selected maintenance action, the predicted recovery, and a realized post-repair performance.
17. The method of claim 1, wherein the deployed model is a neural network and an internal-state node is derived from clustered hidden-layer activations of a tapped layer, and the target parameter set comprises weights of the tapped layer.
18. The method of claim 1, wherein the deployed model is a gradient-boosted-tree ensemble and, responsive to an absence of an independently-retrainable subnetwork, the partial-retrain action degrades to the full-retrain action.
19. The method of claim 1, wherein the deployed model is a linear or logistic model and an internal-state node is derived from per-feature contribution statistics.
20. The method of claim 1, wherein detecting the degradation comprises one of (i) a disjunction and (ii) a conjunction of a distributional-shift condition and a delayed-label performance condition.
21. The method of claim 1, wherein the attribution graph is reconstructed on the detecting over rolling windows rather than continuously, and a node count is bounded by a selection over candidate features and internal-state clusters.
22. The method of claim 1, wherein the maintenance action space further comprises a partial-retrain action defining a fraction of the subnetwork to be updated.
23. The method of claim 1, wherein the predicted recovery is a difference between the predicted post-repair performance and the current performance, and the full-retrain action is assigned a full-retrain recovery estimate obtained from a running statistic of realized full-retrain recoveries. *(Conditioned on inventor confirmation of the `R̂_full` mechanism — see 4.2; delete or amend if not implemented.)*
24. The system of claim 2, wherein the recovery surrogate is trained jointly with the graph message-passing model by back-propagating a regression loss through the surrogate into the graph message-passing model.

---

## PART 6 — CLAIM-BY-CLAIM SUPPORT MATRIX

| Claim | Key limitation | Spec support | Strength | Issue / action |
|---|---|---|---|---|
| 1(c) | unified external+internal graph | §3.4.3, Fig.5 | Strong | — |
| 1(d) | per-node responsibility via GNN | §3.4.4 | Strong | — |
| 1(e) | recovery surrogate, no per-candidate retrain | §3.4.5 | Strong (add explicit amortization sentence, 4.1) | Moderate→Strong |
| 1(f) | target = argmax r·R̂ | §3.4.6 | Strong | — |
| 1(g) | node→parameter-set mapping | §3.4.7, Fig.6 | Strong | — |
| 1(h) | least-cost predicted-feasible action | §3.4.6 | Strong; `R̂_full` kept out | — |
| 1(i) | selective update + deviation penalty | §3.4.8 | Strong | — |
| 1(j) | isolated validation + promote/rollback | §3.4.8 | Strong | — |
| 4–6 | clustering / PC+NOTEARS / message passing | §3.4.3–4, §3.4.14 | Strong | — |
| 7–8 | surrogate inputs / training / single-pass | §3.4.5 | Strong | — |
| 9 | uncertainty + OOD guard | §3.4.5 | Moderate | metric at "preferred embodiment" level — OK |
| 10 | dual-variable SLA + objective | §3.4.6 | Strong | — |
| 11 | concentration guard | §3.4.9 | Strong | — |
| 12–13 | node→component→param, multi-node union | §3.4.7 | Strong | — |
| 14 | diagonal Fisher + gradient masking | §3.4.8 | Strong | — |
| 15 | surrogate recalibration | §3.4.5/§3.4.8 | Strong | — |
| 16 | telemetry | §3.4.9 | Strong | — |
| 17 | neural embodiment | §3.4.10/§3.4.14 | Strong | best mode |
| 18–19 | tree / linear fallback | §3.4.10 | **Weak** | enablement gap for partial update; degrade-to-full is supported |
| 20–22 | trigger / rolling window / fractional scope | §3.4.2/§3.4.6/§3.4.13 | Strong/Moderate | — |
| 23 | `R̂_full` running-statistic | §3.4.6 | **Gap** | requires inventor confirmation (4.2) |
| 24 | joint surrogate+GNN training | §3.4.5 | Strong | — |

---

## PART 7 — PRIOR-ART / INVENTIVE-STEP ATTACK (hostile examiner)

**Attack A (primary).** *US11928011B2 (Dell — causal drift remediation: causal root-cause + tag data + automated retrain + A/B/canary)* in view of *EP3745318A1 (Nvidia — selective weight updates)* and *Kirkpatrick 2017 (EWC)*: "It is obvious to take Dell's causal drift root-cause and, instead of retraining on tagged data, apply Nvidia's selective weight update with EWC to retrain only the implicated part, under a known cost budget."
**Defense.** The claimed method does not select the repair by causal root-cause alone; it selects the target by **the product of responsibility and a *learned predicted recovery* of a candidate repair obtained without a per-candidate retrain**, then maps that target to a **concrete parameter set** and gates the action by **predicted-feasibility against an SLA**. Dell tags *data* and retrains; it does not predict per-candidate repair benefit nor select a parameter scope by `r·R̂`. Nvidia's selective update is driven by training-time weight-usage, not by a drift attribution or a recovery prediction. Combining them yields "retrain the implicated weights," not "predict which subnetwork's repair will most recover performance and is feasible under cost/SLA." The **recovery-prediction-coupled scope selection** (1(e)-(h)) is the missing, non-obvious link, and it solves a technical problem the combination does not address: choosing the minimal parameter modification *expected to restore the metric* without trying each candidate.

**Attack B.** *Oracle US20230139718A1 (automated drift detection→retrain)* + *US12518197B2 (IBM — incremental learning without forgetting)*: "obvious to retrain incrementally on drift."
**Defense.** Both decide *timing/whether* and provide an update *mechanism*; neither attributes over internal model state, predicts candidate recovery, nor selects a parameter scope by `r·R̂` under feasibility. No motivation to introduce the surrogate-coupled scope selection.

**Attack C.** *Graph-attribution / SHAP (EP4046087A1)* + drift detection: "graph attribution identifies the responsible feature; retrain accordingly."
**Defense.** Attribution here is over **internal model state as well as inputs**, and — decisively — is fused with a **predicted repair recovery** to choose a **model-parameter scope**, not merely to explain a prediction. Attribution alone does not teach recovery-coupled scope selection or the constrained action decision.

**Residual risk.** If any single reference already predicts per-candidate repair recovery and selects a parameter scope accordingly, Claim 1 is in jeopardy — **this must be checked at claim level** (Parts 8–9).

---

## PART 8 — ANTICIPATION ANALYSIS

| Reference (title, assignee) | Single-reference anticipation of revised Claim 1? |
|---|---|
| US10762444B2 — real-time drift detection (Quickpath) | **NOT VERIFIED.** Abstract covers drift detection only; no indication of recovery surrogate or scope selection. Full read required. |
| US11928011B2 — causal drift remediation (Dell) | **NOT VERIFIED — closest reference; full claim-level read REQUIRED.** Abstract shows causal root-cause + data tagging + retrain + canary, but not `r·R̂` scope selection or a per-candidate recovery surrogate. |
| US12518197B2 — incremental learning without forgetting (IBM) | **NOT VERIFIED.** Appears to be an update mechanism, not a diagnosis→scope loop. |
| US20230139718A1 — automated dataset drift detection (Oracle) | **NOT VERIFIED.** Timing decision; no scope-selection surrogate indicated. |
| EP3745318A1 — selective weight updates (Nvidia) | **NOT VERIFIED.** Training-efficiency weight selection; no drift attribution/recovery indicated. |
| EP4046087A1 — ML interpretability / SHAP (Kinaxis) | **NOT VERIFIED.** Prediction attribution; no repair loop indicated. |

No anticipation is asserted; each requires a full-text, claim-level review by a registered patent agent before any position is taken.

---

## PART 9 — CLAIM CHART (revised Claim 1)

| Claim-1 element | Candidate reference | Disclosed? | Evidence/location | Why it matters | Obviousness combo | CADENCE distinguishing interaction |
|---|---|---|---|---|---|---|
| (a) external + internal state via adapter | US11928011; EP4046087 | NOT VERIFIED | abstracts only | grounds attribution in model state, not inputs alone | Dell(data)+SHAP(pred) | internal-state nodes tapped from the model, unified with features |
| (b) degradation detection | US10762444; US20230139718 | Likely (abstract) | drift-detection abstracts | trigger | any drift detector | not relied upon for novelty |
| (c) unified attribution graph | US11928011 (causal graph over data) | NOT VERIFIED | abstract | localizes cause across data+model | Dell + graph-attribution | nodes include internal activations + performance node |
| (d) responsibility via message passing | EP4046087; graph-attribution | NOT VERIFIED | abstract | per-node blame | attribution refs | message passing over the unified graph |
| (e) predicted recovery, no per-candidate retrain | — | NOT VERIFIED (none identified) | — | **amortizes intervention cost** | — | **learned surrogate predicts benefit without retraining each candidate** |
| (f) target = argmax r·R̂ | — | NOT VERIFIED (none identified) | — | **couples diagnosis to benefit** | — | **multiplicative responsibility×recovery selection** |
| (g) node→parameter-set map | EP3745318 (weights) | NOT VERIFIED | abstract | converts attribution to a concrete edit | Nvidia selective weights | scope chosen by (f), not by weight-usage |
| (h) least-cost predicted-feasible action | cost-aware retraining | NOT VERIFIED | — | minimal corrective scope | Dell + cost-aware | feasibility by *predicted* post-repair perf vs SLA |
| (i) selective update + deviation penalty | EP3745318 + EWC | NOT VERIFIED | abstracts | targeted, forgetting-safe edit | Nvidia + Kirkpatrick | applied only to the (f)-selected scope |
| (j) isolated validation + rollback | US11928011 (A/B/canary) | NOT VERIFIED | abstract | reversible, serving-safe | Dell canary | gates the (f)-(i) targeted repair |

Elements **(e)** and **(f)** are the ones for which no candidate reference has been identified even at the abstract level — the strongest points of novelty, pending full search.

---

## PART 10 — TECHNICAL EFFECT (source-supported)

The claimed architecture produces effects on the operation of the computing system, not merely a better prediction:
- **Reduced parameter-update and floating-point operations per adaptation**, because only the mapped target parameter set is updated while the remainder is frozen (§3.4.8; disclosure reports ~66% and ~69% lower retraining compute in the described scenarios — to be verified against the ledger).
- **Reduced training-data/compute footprint of the decision**, because candidate repair benefits are obtained from a single surrogate forward pass rather than a retrain per candidate (§3.4.5).
- **Controlled, reversible modification of model state** with the **live serving path uninterrupted**, via isolated validation and snapshot promotion/rollback (§3.4.8).
- **Constrained compute allocation** across candidate corrective actions under an explicit cost/SLA feasibility rule (§3.4.6).
The technical problem — adapting a deployed model at minimal compute without degrading either the served metric or previously retained competence — is solved by technical means (surrogate-amortized scope selection + selective, forgetting-constrained, validated update), which is the basis for the technical-contribution (India §3(k) / EPO) assessment.

---

## PART 11 — DISCLOSURE GAPS (prioritized; DO NOT INVENT)

**CRITICAL**
1. **`R̂_full` derivation** — how predicted full-retrain recovery is obtained (see 4.2). Needed to support Claim 23 and any objective using `R̂_full`.
2. **Prior-art claim-level read** of US11928011B2 (Dell) and the other five references — required before asserting any distinction; currently abstract-level only.

**HIGH**
3. **Non-neural enablement** — concrete node→parameter mapping and partial-update mechanism for GBDT / linear / transformer, or explicit degrade-to-full (trees/linear already degrade). Governs Claims 18–19 and any "model-agnostic" breadth.
4. **Public-disclosure dates** (see Part 17-equivalent below) — any prior public disclosure (paper, preprint, thesis, GitHub, demo) with dates; determines novelty-grace and filing urgency.

**MEDIUM**
5. **Exact best-mode numerical values** — confirm the §3.4.14 values (cluster counts, α, L1, EWC coefficient, epochs) match the actual implementation.
6. **Uncertainty/OOD metric** actually used at runtime (embedding distance type and threshold).
7. **Results provenance** — confirm each reported number (MRR, AUROC, R², ~69%/~66% compute, forgetting) against the versioned ledger; keep the ledger qualification in the spec.

**LOW**
8. Telemetry schema specifics; adapter operation signatures; registry/versioning details.

---

## PART 12 — FINAL RED-TEAM REVIEW (hostile examiner scores, revised claim set)

| Criterion | Score /10 | Note |
|---|---|---|
| Novelty | 7 | (e)/(f) show no identified anticipation, but full search pending |
| Inventive step | 6.5 | Strong interaction defense; Dell+Nvidia+EWC combo is the real threat |
| Clarity | 8 | Claim 1 is a coherent chain; `R̂_full` removed from independents |
| Support (§112 / §10) | 7 | Strong for neural; weak for non-neural families; `R̂_full` gap |
| Enablement | 6.5 | Neural best-mode good; non-neural under-enabled |
| Claim construction | 8 | Consistent 1/2/3; explicit dependencies; layered tree |
| Technical effect | 8 | Concrete, source-supported computing effects |
| **Overall patentability** | **7** | Credible after gaps closed and search done |

**FINAL VERDICT: File after major revision.** Specifically: (i) close the `R̂_full` and non-neural-enablement gaps or narrow the claims to the neural embodiment; (ii) complete a professional claim-level prior-art search (esp. Dell US11928011B2); (iii) verify all results against the ledger. With (i)–(iii), the reconciled claim set in Part 5 is filing-grade. **Do not file the current document as a complete specification**; an **India provisional now** (to secure priority) followed by a **PCT** on the revised set is the recommended sequence. This is drafting strategy, not legal advice — a registered patent agent must sign off.

---

## PART 13 — CHANGE LOG

| Original weakness | Change made | Why it helps | Residual risk |
|---|---|---|---|
| Novelty rested on component list | Recast core as responsibility×recovery-coupled scope selection in a validated closed loop | Moves novelty to a non-obvious interaction | Depends on full search |
| `i*=argmax r·R̂` buried in spec | Elevated to Claim 1(f) | Captures the strongest differentiator | — |
| Optimizer inconsistency (least-cost vs J) | Reconciled: least-cost among predicted-feasible; J as ranking in dep. Claim 10 | One coherent decision rule | — |
| `R̂_full` unsupported in decision | Removed from independents; full-retrain = fallback when nothing feasible; `R̂_full` only in Claim 23 gated on inventor input | Eliminates a §112 exposure | Claim 23 deletable if not implemented |
| Over-claimed "causal" | "causal-structure-informed"; "counterfactual" defined operationally | Matches the disclaimer; avoids overclaim | — |
| "Model-agnostic" over-broad | Neural independent embodiment; tree/linear as degrade-to-full dependents; enablement gap flagged | Aligns breadth with enablement | Non-neural breadth limited until disclosed |
| Validation/rollback under-claimed | Added as Claim 1(j) (closed loop) | Strengthens inventive-step (control architecture) | Slightly narrows Claim 1 |
| Prior art asserted loosely | All anticipation marked NOT VERIFIED; claim chart flags unread refs | Avoids unsupported certainty | Requires the actual search |
| Public-disclosure blank | Flagged CRITICAL/HIGH with checklist | Protects novelty/priority | Inventor must supply dates |

### Public-disclosure checklist (complete before filing; dates matter for novelty/priority)
conference submission · paper/preprint · thesis · poster · oral presentation · GitHub/public repo · project website · live demo · social-media post · university repository · patent filing · any third-party demonstration. **Record the earliest public date for each; if any exists, filing may be time-critical.**
