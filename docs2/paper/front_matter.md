# CADENCE — Paper Front Matter (title, objectives, abstract, introduction)

> Grounded in the project's *measured* results. H1 (causal attribution) and H3
> (forgetting) are supported at 10-seed paper scale; the RSO / H2 result is
> framed as a controllable cost/quality trade-off (measured 69% compute / 1.7%
> F1 on Elec2), not strict Pareto dominance, which the confirmatory Stage-2 run
> has not yet established. Do not strengthen the H2 wording until R-Gate-A-final
> lands.

---

## Title

**CADENCE: Causal Attribution-Driven, Cost-Aware Continual Retraining for Production Machine Learning Systems**

*Alternative short title (for slides / camera-ready running head):*
CADENCE: Knowing *Why* a Model Broke, and *How Much* to Retrain

---

## Objectives

**Objective 1 — Causal, model-internals-aware drift attribution.**
Design and evaluate an online attribution mechanism, the *Causal Drift
Attribution Graph* (CDAG), that identifies **which** upstream factor is
responsible for a deployed model's performance degradation — not merely
**that** a distribution shifted. The CDAG spans a unified node set of external
input-feature statistics **and** internal activation-cluster statistics of the
deployed model, and a graph neural network learns a per-node responsibility
score. The objective is to demonstrate that this causal attribution localizes
the true root cause more precisely than correlational drift detectors (PSI /
KS-test) on drift scenarios where the correlational signal is ambiguous.

**Objective 2 — Cost- and carbon-aware retraining-scope optimization.**
Formulate the decision of **how much** of a deployed model to retrain — none,
a targeted subnetwork, or the whole model — as a constrained reinforcement-
learning problem. A policy over the discrete action set {no-op, partial retrain,
full retrain} is trained with an augmented-Lagrangian reward that jointly
penalizes compute cost and carbon cost subject to a service-level-agreement
(SLA) recovery constraint. The objective is to quantify the achievable
cost/quality trade-off on real concept-drift data and to show that a single
learned policy exposes an operator-tunable knob that fixed-rule baselines
(periodic and reactive-full retraining) cannot.

**Objective 3 — Forgetting-safe, model-agnostic targeted retraining.**
Close the loop with an execution layer that applies Elastic Weight
Consolidation (EWC)-regularized partial retraining to the attributed
subnetwork, guarded by shadow/canary validation and rollback. The objective is
twofold: (i) show that causally-targeted partial retraining reduces
catastrophic forgetting relative to naive full retraining on a genuinely
multi-task benchmark, and (ii) demonstrate that the entire attribution →
decision → retraining loop is model-agnostic, operating unchanged across
neural-network, gradient-boosted-tree, and linear/text production models
through a common adapter interface.

---

## Abstract

Every deployed machine-learning model degrades as the data it now sees drifts
away from the data it was trained on. In practice, teams respond either by
retraining on a fixed calendar — wasting large amounts of compute and carbon on
models that have not degraded — or by waiting for a human to notice a metric
drop, by which point real damage has occurred. Crucially, existing production
monitors are *correlational*: they report **that** a feature distribution
shifted, not **why** the model broke or **how much** of it must be repaired.
We present **CADENCE**, a closed-loop system that couples causal drift
attribution to a cost-aware retraining-scope policy. CADENCE builds a *Causal
Drift Attribution Graph* over a unified node set of external input features and
internal model activation clusters, and a graph neural network scores each
node's responsibility for the observed degradation; a counterfactual surrogate,
trained offline on sandboxed interventions, predicts the recovery obtainable
from repairing each node without performing costly ground-truth retrains. These
responsibility scores drive a constrained reinforcement-learning policy that
chooses among no-op, EWC-regularized partial retraining, and full retraining to
minimize compute and carbon cost subject to an SLA-recovery constraint; the
chosen fix is validated in shadow mode before promotion. Across neural-network,
tree-ensemble, and text production models, we find that (i) the learned
attribution localizes the true injected root cause significantly more precisely
than PSI/KS-test baselines on hard concept-shift and gradual-drift scenarios
(mean-reciprocal-rank and AUROC gains, *p* < 0.001 over ten seeds); (ii) on a
real concept-drift benchmark the policy delivers a controllable cost/quality
trade-off, cutting retraining compute by roughly 69% for a 1.7-point F1
reduction versus a reactive full-retrain baseline; and (iii) causally-targeted
partial retraining reduces catastrophic forgetting on unrelated tasks relative
to naive full retraining (*p* < 0.001). CADENCE reframes model maintenance from
a fixed schedule into a self-healing loop that acts only when, where, and as
much as the evidence warrants.

---

## 1. Introduction

Machine-learning models are increasingly deployed as long-lived services —
fraud detectors, credit-risk scorers, recommendation rankers, churn predictors
— whose accuracy silently erodes as the production data-generating process
drifts away from the training distribution. Industry surveys repeatedly rank
this *concept and data drift* among the leading causes of production-ML failure.
The dominant operational responses are both unsatisfying. A **fixed-schedule**
policy retrains every model on a calendar cadence regardless of need, spending
substantial compute — and, at scale, non-trivial carbon — on models that have
not degraded. A **reactive** policy waits until a downstream key metric visibly
drops and then retrains everything from scratch, incurring both the delay-cost
of degraded predictions and the full compute cost of a from-scratch retrain.
Neither policy reasons about *whether*, *where*, or *how much* a given model
actually needs to change.

A second, deeper limitation cuts across today's drift-monitoring tooling.
Production monitors — whether the PSI and KS-test detectors built into
commercial observability platforms or their academic antecedents — are
fundamentally **correlational**. They raise an alarm when an input feature's
distribution shifts, but they cannot say whether that shift is *causally
responsible* for the model's degradation, nor which internal part of the model
the shift actually broke. This yields two failure modes at once: false alarms
that trigger unnecessary retraining when a shifted-but-irrelevant feature moves,
and uninformative alarms that tell an engineer a distribution changed without
telling them what to fix. The gap is the absence of a system that answers the
two operational questions that matter — *why did the model break?* and *how much
of it must be repaired?* — and that couples those answers into an automated,
cost-aware repair.

We address this gap with **CADENCE**, a closed-loop, self-healing maintenance
system for production models. CADENCE's core is the *Causal Drift Attribution
Graph* (CDAG): a graph whose nodes span both external input-feature statistics
and internal activation-cluster statistics of the deployed model, with edges
inferred online from streaming inference telemetry via a hybrid constraint-based
and continuous-optimization causal-discovery procedure. A graph neural network
embeds each node in the context of its causal neighborhood, and a counterfactual
surrogate — trained offline on sandboxed drift-and-repair experiments — maps
each embedding to the model recovery that repairing that node would yield,
producing a ranked *responsibility score* without performing an expensive
ground-truth retrain to test each hypothesis. These scores feed a constrained
reinforcement-learning policy, the *Retraining Scope Optimizer*, which chooses
among doing nothing, retraining only the implicated subnetwork under an
Elastic-Weight-Consolidation penalty, or retraining the full model, so as to
minimize compute and carbon cost subject to an SLA-recovery constraint enforced
through an augmented-Lagrangian dual variable. The selected action is applied by
an executor that validates the candidate model in shadow mode and promotes or
rolls it back, feeding the realized outcome back to recalibrate the surrogate.

Two design choices give CADENCE its generality and its honesty. First, the
production model is reached only through a narrow **adapter interface** — score,
tap activations, partial-fit, full-fit — so the identical attribution-and-
decision loop operates unchanged over a feed-forward neural network, a
gradient-boosted-tree ensemble, and a linear text classifier; where a model
class cannot support targeted partial retraining, the executor degrades
gracefully to a full retrain. Second, attribution is validated where ground
truth exists — on **synthetic drift** whose injected root cause is known — while
end-to-end value is validated on **real drift** where the cause is unknown and
the system is judged by the outcome it produces (performance recovered, compute
saved). This separation lets us make a precise attribution claim without
overclaiming on data where no answer key exists.

We evaluate CADENCE across public tabular, streaming, and image benchmarks. On
synthetic drift with a known injected cause, the learned CDAG attribution
localizes the true root cause significantly more precisely than PSI/KS-test
baselines on the hard scenarios where correlational signal is ambiguous —
concept-shift and gradual drift — with mean-reciprocal-rank and AUROC gains
significant at *p* < 0.001 over ten seeds, while matching the (already-saturated)
baseline on easy abrupt shifts. On the Electricity real-drift benchmark, the
Retraining Scope Optimizer yields a controllable cost/quality trade-off,
reducing retraining compute by approximately 69% for a 1.7-point F1 reduction
relative to a reactive full-retrain baseline — a knob that fixed-rule baselines
do not expose. And on a genuinely multi-task Split-MNIST benchmark, EWC-
regularized partial retraining reduces forgetting on the preserved task from a
roughly 40-point F1 loss under naive full retraining to a negligible change,
significant at *p* < 0.001 and robust to a stronger full-retrain-with-replay
baseline.

**Contributions.** This paper makes three contributions. **(1)** A unified,
online causal attribution mechanism (the CDAG plus a jointly-trained GNN and
counterfactual surrogate) that reaches into model internals and identifies the
root cause of production degradation more precisely than correlational
detectors. **(2)** A formulation of retraining *scope* selection as a
compute- and carbon-aware constrained RL problem, and evidence that a single
learned policy provides an operator-tunable cost/quality trade-off unavailable
to fixed rules. **(3)** A model-agnostic, forgetting-safe execution loop —
EWC-targeted partial retraining with shadow validation and rollback — together
with an experimental protocol that validates attribution on synthetic
ground-truth and end-to-end value on real drift, and an open, reproducible
implementation with a resumable experiment harness and pre-registered success
criteria. The remainder of the paper details the CDAG construction (§3), the
attribution and surrogate models (§4), the Retraining Scope Optimizer (§5), the
execution and validation loop (§6), and the experimental evaluation (§7).
