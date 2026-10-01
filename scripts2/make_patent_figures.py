#!/usr/bin/env python
"""Patent figures for CADENCE — architecture block diagram, Level-0 / Level-1
data-flow diagrams (Gane-Sarson style), and the runtime decision pipeline.

Pure matplotlib; clean, print-ready PNGs (300 dpi, white ground).
Output: docs/paper/patent_figures/*.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path("docs/paper/patent_figures")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
C_PROC = "#dce9f7"     # process (blue)
C_ENT = "#efe3cf"      # external entity (tan)
C_STORE = "#e6f0da"    # data store (green)
C_DEC = "#fbe3cf"      # decision (orange)
C_EXEC = "#f6dede"     # execution (red)
ACCENT = "#2e5a88"     # accent text


def _proc(ax, x, y, w, h, num, title, face=C_PROC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                 lw=1.4, edgecolor=INK, facecolor=face, zorder=2))
    ax.text(x + 0.14, y + h - 0.16, num, fontsize=8.5, fontweight="bold", color=INK, zorder=3, va="top")
    ax.text(x + w / 2, y + h / 2 - 0.02, title, ha="center", va="center", fontsize=8.4,
            fontweight="bold", color=INK, zorder=3)


def _entity(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, lw=1.6, edgecolor=INK, facecolor=C_ENT, zorder=2))
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=INK, zorder=3)


def _store(ax, x, y, w, h, tag, title):
    ax.add_patch(Rectangle((x, y), w, h, lw=0, facecolor=C_STORE, zorder=2))
    ax.plot([x, x + w], [y, y], color=INK, lw=1.4, zorder=3)
    ax.plot([x, x + w], [y + h, y + h], color=INK, lw=1.4, zorder=3)
    ax.plot([x, x], [y, y + h], color=INK, lw=1.4, zorder=3)
    ax.text(x + 0.13, y + h / 2, tag, fontsize=8, fontweight="bold", va="center", color=INK, zorder=4)
    ax.text(x + w / 2 + 0.12, y + h / 2, title, ha="center", va="center", fontsize=8.2, color=INK, zorder=4)


def _arrow(ax, p0, p1, label=None, rad=0.0, lp=0.5, color=INK, dxy=(0, 0.12), fs=7.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=1.3,
                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx = p0[0] + lp * (p1[0] - p0[0]) + dxy[0]
        my = p0[1] + lp * (p1[1] - p0[1]) + dxy[1]
        ax.text(mx, my, label, fontsize=fs, color="#333", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def fig_context():
    fig, ax = plt.subplots(figsize=(9.2, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.1, 5.75, "FIG. 2 — Level-0 (Context) Data-Flow Diagram", fontsize=11, fontweight="bold")
    _entity(ax, 0.4, 2.9, 2.5, 1.3, "Deployed\nProduction Model")
    _entity(ax, 9.1, 2.9, 2.5, 1.3, "MLOps Operator /\nDashboard")
    _proc(ax, 4.7, 2.55, 2.6, 1.9, "0", "CADENCE\nCausal-Repair\nLoop", C_PROC)
    _arrow(ax, (2.9, 3.9), (4.7, 3.9), "inference telemetry\n(features + activations)", dxy=(0, 0.35))
    _arrow(ax, (4.7, 3.1), (2.9, 3.1), "model-update\ncontrol signal", dxy=(0, -0.35))
    _arrow(ax, (7.3, 3.9), (9.1, 3.9), "health telemetry\n(cause, confidence)", dxy=(0, 0.35))
    _arrow(ax, (9.1, 3.1), (7.3, 3.1), "SLA / cost policy", dxy=(0, -0.3))
    fig.savefig(OUT / "fig_dfd_level0.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _pbox(ax, x, y, w, h, title, face=C_PROC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                 lw=1.4, edgecolor=INK, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2,
            fontweight="bold", color=INK, zorder=3)


def fig_level1():
    fig, ax = plt.subplots(figsize=(11.4, 6.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 9.5); ax.axis("off")
    ax.text(0.1, 9.1, "FIG. 3 — Level-1 Data-Flow Diagram (CADENCE internal processes)", fontsize=11.5, fontweight="bold")
    # top row (L->R): entity, 1, 2, 3
    _entity(ax, 0.3, 6.7, 2.2, 1.3, "Deployed\nModel")
    _pbox(ax, 3.4, 6.7, 2.7, 1.3, "1. Monitor &\nDrift Trigger")
    _pbox(ax, 7.0, 6.7, 2.7, 1.3, "2. CDAG\nConstruction")
    _pbox(ax, 10.6, 6.7, 3.1, 1.3, "3. Responsibility Scoring\n(GNN + Surrogate)")
    # bottom row (R->L): 4, 5, entity(update) at far left
    _pbox(ax, 10.6, 4.0, 3.1, 1.3, "4. Retraining-Scope\nOptimizer (RSO)")
    _pbox(ax, 7.0, 4.0, 2.7, 1.3, "5. Guarded Executor\n& Shadow Validation")
    _entity(ax, 0.3, 4.05, 2.2, 1.2, "Deployed\nModel (updated)")
    # data stores: D1 under proc1, D3 under proc5, D2 under proc4 (no crossings)
    _store(ax, 3.5, 1.5, 2.9, 0.72, "D1", "Baseline stats")
    _store(ax, 6.9, 1.5, 3.0, 0.72, "D3", "Model registry")
    _store(ax, 10.6, 1.5, 3.1, 0.72, "D2", "Intervention set")
    # top-row flows
    _arrow(ax, (2.5, 7.35), (3.4, 7.35), "windows +\nactivations", dxy=(0, 0.42), fs=6.6)
    _arrow(ax, (6.1, 7.35), (7.0, 7.35), "alert +\nsignals", dxy=(0, 0.42), fs=6.6)
    _arrow(ax, (9.7, 7.35), (10.6, 7.35), "CDAG", dxy=(0, 0.3), fs=6.8)
    # 3 -> 4 (down)
    _arrow(ax, (12.15, 6.7), (12.15, 5.3), "responsibility +\npredicted benefit", dxy=(1.4, 0), fs=6.6)
    # 4 -> 5 (left)
    _arrow(ax, (10.6, 4.65), (9.7, 4.65), "chosen\nscope", dxy=(0, 0.0), fs=6.2)
    # 5 -> entity update (far left)
    _arrow(ax, (7.0, 4.65), (2.5, 4.65), "promote / rollback", dxy=(0, 0.34), fs=6.6)
    # vertical data-store flows (straight down, no crossings)
    _arrow(ax, (4.75, 6.7), (4.75, 2.22), "PSI baseline", dxy=(-0.85, 0), fs=6.4)
    _arrow(ax, (8.3, 4.0), (8.3, 2.22), "snapshots", dxy=(-0.75, 0), fs=6.4)
    _arrow(ax, (12.15, 4.0), (12.15, 2.22), "interventions", dxy=(1.0, 0), fs=6.4)
    fig.savefig(OUT / "fig_dfd_level1.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 8.6)); ax.set_xlim(0, 8.4); ax.set_ylim(0, 12); ax.axis("off")
    ax.text(0.1, 11.6, "FIG. 4 — Runtime Cause-Gating Decision Pipeline", fontsize=11, fontweight="bold")
    steps = [
        (10.3, "Monitor window: PSI + delayed-label F1", C_PROC),
        (9.1, "Drift / SLA breach detected?", C_DEC),
        (7.9, "Build CDAG over features + activation clusters", C_PROC),
        (6.7, "Score node responsibility (GNN) +\npredict repair benefit (surrogate)", C_PROC),
        (5.5, "Attribution concentrated?", C_DEC),
        (4.3, "RSO selects scope under cost/SLA\n{no-op | partial (EWC) | full}", C_PROC),
        (3.1, "Execute + shadow/canary validate", C_PROC),
        (1.9, "Post-fix F1 >= SLA?", C_DEC),
        (0.7, "Promote model  /  Rollback + flag", C_STORE),
    ]
    xs = 0.7
    boxes = []
    for (y, txt, face) in steps:
        h = 0.82
        ax.add_patch(FancyBboxPatch((xs, y), 4.6, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                     lw=1.4, edgecolor=INK, facecolor=face, zorder=2))
        ax.text(xs + 2.3, y + h / 2, txt, ha="center", va="center", fontsize=8.0, color=INK, zorder=3)
        boxes.append((y, h))
    for i in range(len(steps) - 1):
        y0 = boxes[i][0]; y1 = boxes[i + 1][0] + boxes[i + 1][1]
        _arrow(ax, (3.0, y0), (3.0, y1))
    # side branch labels to the RIGHT of the decision boxes (clear of boxes, no clipping/overlap)
    def _branch(yy, text):
        ax.add_patch(FancyArrowPatch((5.3, yy + 0.41), (6.05, yy + 0.41), arrowstyle="-|>",
                     mutation_scale=10, lw=1.1, color="#a33", zorder=3))
        ax.text(6.2, yy + 0.41, text, fontsize=7.6, color="#a33", ha="left", va="center", zorder=3)
    _branch(9.1, "no → no-op")
    _branch(5.5, "diffuse → full")
    _branch(1.9, "fail → rollback")
    fig.savefig(OUT / "fig_decision_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _refbox(ax, x, y, w, h, ref, title, face=C_PROC, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                 lw=1.4, edgecolor=INK, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2 + 0.09, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + 0.17, ref, ha="center", va="center", fontsize=7.8, color="#a11", zorder=3)


def fig_arch_numbered():
    """FIG. 1 — patent block diagram with reference numerals."""
    fig, ax = plt.subplots(figsize=(9.6, 6.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.1, 9.6, "FIG. 1 — CADENCE System Architecture (with reference numerals)", fontsize=11.5, fontweight="bold")
    _refbox(ax, 0.4, 7.6, 2.7, 1.5, "100", "Deployed\nProduction Model", C_ENT)
    _refbox(ax, 0.5, 5.0, 2.5, 1.3, "105", "Adapter Interface\n(score/tap/fit)", C_STORE)
    _refbox(ax, 4.0, 7.7, 2.6, 1.3, "110", "Streaming\nCollector")
    _refbox(ax, 7.2, 7.7, 2.6, 1.3, "120", "Drift & SLA\nTrigger")
    _refbox(ax, 10.6, 7.7, 3.8, 1.3, "130", "CDAG Generator\n(features + activations)")
    _refbox(ax, 10.6, 5.2, 3.8, 1.3, "140", "Responsibility Scorer\n(GNN)")
    _refbox(ax, 10.6, 2.9, 3.8, 1.3, "150", "Counterfactual\nRecovery Surrogate")
    _refbox(ax, 6.6, 2.9, 3.2, 1.3, "160", "Retraining-Scope\nOptimizer (RSO)", C_DEC)
    _refbox(ax, 3.0, 2.9, 3.0, 1.3, "170", "Forgetting-Safe\nExecutor (EWC)", C_EXEC)
    _refbox(ax, 3.0, 0.7, 3.0, 1.3, "180", "Shadow / Canary\nValidation", C_EXEC)
    _refbox(ax, 6.6, 0.7, 3.2, 1.3, "190", "Model Registry /\nSnapshots", C_STORE)
    _refbox(ax, 10.6, 0.7, 3.8, 1.3, "195", "Health-Telemetry\nOutput", C_STORE)
    a = lambda p0, p1, rad=0: ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                                           lw=1.3, color=INK, connectionstyle=f"arc3,rad={rad}", zorder=1))
    a((3.1, 8.35), (4.0, 8.35)); a((6.6, 8.35), (7.2, 8.35)); a((9.8, 8.35), (10.6, 8.35))
    a((12.5, 7.7), (12.5, 6.5)); a((12.5, 5.2), (12.5, 4.2)); a((10.6, 3.55), (9.8, 3.55))
    a((6.6, 3.55), (6.0, 3.55)); a((4.5, 2.9), (4.5, 2.0)); a((6.0, 1.35), (6.6, 1.35))
    a((3.0, 5.0), (1.75, 5.0)); a((1.75, 6.3), (1.75, 7.6))  # executor -> adapter -> model
    a((9.8, 1.35), (10.6, 1.35))
    ax.text(1.0, 6.5, "control\nsignal", fontsize=6.6, color="#555")
    fig.savefig(OUT / "fig_arch_numbered.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_cdag():
    """FIG. 5 — CDAG node set."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(0.1, 6.7, "FIG. 5 — Causal Drift-Attribution Graph (CDAG) node set", fontsize=11.5, fontweight="bold")
    import numpy as np
    feats = [(1.2, 5.4, "F1\n(210)"), (1.2, 4.2, "F2"), (1.2, 3.0, "..."), (1.2, 1.8, "Fn (210)")]
    for x, y, t in feats:
        ax.add_patch(plt.Circle((x, y), 0.42, fc=C_ENT, ec=INK, lw=1.3, zorder=2)); ax.text(x, y, t, ha="center", va="center", fontsize=7.4, zorder=3)
    clu = [(5.5, 5.4, "A1"), (5.5, 4.2, "A2 (220)"), (5.5, 3.0, "..."), (5.5, 1.8, "Ak")]
    for x, y, t in clu:
        ax.add_patch(plt.Circle((x, y), 0.44, fc=C_PROC, ec=INK, lw=1.3, zorder=2)); ax.text(x, y, t, ha="center", va="center", fontsize=7.4, zorder=3)
    ax.add_patch(plt.Circle((9.6, 3.6), 0.7, fc=C_DEC, ec=INK, lw=1.6, zorder=2)); ax.text(9.6, 3.6, "Perf.\nP (230)", ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
    for (fx, fy, _), (cx, cy, _) in zip(feats, clu):
        ax.add_patch(FancyArrowPatch((fx + 0.42, fy), (cx - 0.44, cy), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#666", connectionstyle="arc3,rad=0.05", zorder=1))
    for cx, cy, _ in clu:
        ax.add_patch(FancyArrowPatch((cx + 0.44, cy), (9.6 - 0.7, 3.6), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#666", connectionstyle="arc3,rad=0.08", zorder=1))
    ax.text(1.2, 6.05, "External feature nodes", ha="center", fontsize=8, style="italic", color=ACCENT)
    ax.text(5.5, 6.05, "Internal activation-cluster nodes", ha="center", fontsize=8, style="italic", color=ACCENT)
    ax.text(9.6, 4.7, "Performance node", ha="center", fontsize=8, style="italic", color=ACCENT)
    ax.text(6.0, 0.7, "Edges (240): PC skeleton masked by NOTEARS-style continuous weights", ha="center", fontsize=7.6, color="#555")
    fig.savefig(OUT / "fig_cdag.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_mapping():
    """FIG. 6 — node -> retrainable subnetwork mapping."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(0.1, 6.1, "FIG. 6 — Attributed node → retrainable subnetwork mapping", fontsize=11.5, fontweight="bold")
    _refbox(ax, 0.4, 3.6, 2.6, 1.2, "220", "Activation cluster\nnode g_i")
    _refbox(ax, 0.4, 1.4, 2.6, 1.2, "210", "Feature node f_j", C_ENT)
    _refbox(ax, 4.3, 2.5, 2.7, 1.3, "310", "Mapping\nΦ: node → scope")
    # target model layers
    for i, (yy, lab, ref) in enumerate([(4.3, "Layer L1 (input-facing)", "320"), (2.9, "Layer L2", ""), (1.5, "Layer Lm (output)", "")]):
        _refbox(ax, 8.2, yy, 3.4, 1.0, ref, lab, C_STORE, fs=7.6)
    a = lambda p0, p1, rad=0: ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK, connectionstyle=f"arc3,rad={rad}", zorder=1))
    a((3.0, 4.2), (4.3, 3.4)); a((3.0, 2.0), (4.3, 2.8))
    a((7.0, 3.5), (8.2, 4.8), 0.1); a((7.0, 3.0), (8.2, 3.4)); a((7.0, 2.7), (8.2, 2.0), -0.1)
    ax.text(6.0, 0.6, "Cluster node → its layer; feature node → input-facing layer; multi-node → union of scopes; "
                       "no retrainable component → escalate to full retrain.", ha="center", fontsize=7.4, color="#555")
    fig.savefig(OUT / "fig_mapping.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_rso():
    """FIG. 7 — RSO constrained decision."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.1, 5.6, "FIG. 7 — Retraining-Scope Optimizer as a constrained decision (160)", fontsize=11.5, fontweight="bold")
    _refbox(ax, 0.3, 2.2, 3.0, 1.5, "410", "State s_t:\nPSI, F1, SLA margin,\nconcentration κ", C_STORE, fs=7.6)
    _refbox(ax, 4.2, 2.4, 3.0, 1.2, "420", "Policy π(a|s;λ)\nargmax J(a)", C_DEC)
    for i, (yy, lab, ref) in enumerate([(4.2, "no-op", "a0"), (2.8, "partial (EWC)", "a1"), (1.4, "full retrain", "a2")]):
        _refbox(ax, 8.4, yy, 3.2, 1.0, ref, lab, C_EXEC, fs=7.8)
    _refbox(ax, 4.2, 0.5, 3.0, 1.1, "430", "Dual update λ\n(SLA constraint)", C_STORE, fs=7.4)
    a = lambda p0, p1, rad=0: ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK, connectionstyle=f"arc3,rad={rad}", zorder=1))
    a((3.3, 2.95), (4.2, 2.95))
    a((7.2, 3.2), (8.4, 4.7), 0.1); a((7.2, 3.0), (8.4, 3.3)); a((7.2, 2.7), (8.4, 1.9), -0.1)
    a((5.7, 2.4), (5.7, 1.6)); a((5.7, 0.5), (4.4, 2.4), 0.3)
    ax.text(6.0, 0.05, "J(a)=E[ΔF1(a)] − w·C(a) − λ·max(0, SLA − F1(a));   λ ← clip(λ + η(v−δ),0,λmax)", ha="center", fontsize=7.2, color="#333")
    fig.savefig(OUT / "fig_rso.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_context(); fig_level1(); fig_pipeline()
    fig_arch_numbered(); fig_cdag(); fig_mapping(); fig_rso()
    print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
