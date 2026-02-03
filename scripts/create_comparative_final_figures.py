"""Create comparative, self-explanatory, colorful figures for final presentations.

Goal: four slide-ready figures summarizing the latest update in mldm-project.

Design rules:
- Each figure uses ONE chart type (annotations ok).
- Figures are self-explanatory (strong title + short subtitle + clear axes).
- Diverse aspect ratios (wide / tall / square).

Inputs:
- outputs/latest/slides/models_comparison.json (written by scripts/analyze_results.py)
- outputs/latest/slides/best_model*.json (approach variants / stabilization toggles)

Outputs (into outputs/latest/slides/):
- PRES1_MODELS_HEATMAP_WIDE.png
- PRES2_ROLLOUT_BARS_TALL.png
- PRES3_FIT_STABILITY_SQUARE.png
- PRES4_M3_VS_M4_DUMBBELL_WIDE.png
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = PROJECT_ROOT / "outputs" / "latest" / "slides"
PATCH_CSV = PROJECT_ROOT / "outputs" / "latest" / "patch_pde" / "PATCH_PDE_COEFFS.csv"


COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def _style() -> None:
    # Paper-ish defaults: clean, readable, minimal ink.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "legend.frameon": False,
        }
    )


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _short_model_name(name: str) -> str:
    # "Model 4: + Nonlinear (u^2)" -> "M4"
    if not name:
        return "?"
    if name.strip().startswith("Model"):
        parts = name.split(":", 1)[0].split()
        if len(parts) >= 2 and parts[1].isdigit():
            return f"M{parts[1]}"
    return name[:10]


def _model_id(name: str) -> str:
    # "Model 4: ..." -> "M4"
    s = str(name)
    if s.strip().startswith("Model"):
        parts = s.split(":", 1)[0].split()
        if len(parts) >= 2 and parts[1].isdigit():
            return f"M{parts[1]}"
    return "M?"


def _wrap_metric_label(s: str) -> str:
    # Keep x tick labels short to avoid overlap.
    return (
        s.replace("One-step fit:", "Fit")
        .replace("One-step error:", "Error")
        .replace("Stability:", "Stability")
        .replace("Complexity:", "Complexity")
        .replace(" (↑)", "\n(↑)")
        .replace(" (↓)", "\n(↓)")
        .replace("rollout k=10", "rollout\nk=10")
        .replace("#active terms", "#active\nterms")
    )


def _subtitle_from_models_json(d: dict) -> str:
    train_frac = d.get("train_frac", "?")
    space_train_frac = d.get("space_train_frac", "?")
    rollout_steps = d.get("rollout_steps", [])
    robust = d.get("use_robust_regression", False)
    return f"split: time={train_frac}/(1-time), space={space_train_frac}/(1-space); rollout k={rollout_steps}; robust={robust}"


def _tighten(ax: plt.Axes) -> None:
    # Reduce clutter from grids/spines.
    ax.grid(True, alpha=0.20)


def _shorten(s: str, max_len: int = 14) -> str:
    s = str(s)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _normalize_column(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    lo = float(np.nanmin(v[finite]))
    hi = float(np.nanmax(v[finite]))
    if abs(hi - lo) < 1e-12:
        out = np.zeros_like(v)
        out[finite] = 1.0
        return out
    out = (v - lo) / (hi - lo)
    out = _clamp01(out)
    return out if higher_is_better else (1.0 - out)


def _sig(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "∞"
    if abs(x) >= 1000:
        return f"{x:.2g}"
    # Use significant digits, but keep it presentation-friendly.
    return f"{x:.{digits}g}"


def _round2(x: float) -> str:
    if not np.isfinite(x):
        return "∞"
    return f"{x:.2f}"


def _extract_models_table(d: dict) -> dict:
    models = d.get("models", [])
    ids = [_model_id(m.get("name", "")) for m in models]
    names = [str(m.get("name", "")) for m in models]
    r2 = np.array([float(m.get("r2_test", np.nan)) for m in models])
    one_step = np.array([float(m.get("one_step_rmse", np.nan)) for m in models])
    rollout10 = np.array([float((m.get("rollout", {}).get("test", {}) or {}).get("nrmse", np.nan)) for m in models])
    active = np.array([float(m.get("n_active", np.nan)) for m in models])
    return {"models": models, "ids": ids, "names": names, "r2": r2, "one_step": one_step, "rollout10": rollout10, "active": active}


def fig_fig2_rollout_vs_horizon(models_json: Path, out_path: Path, *, model_ids: tuple[str, ...] = ("M3", "M4", "M5")) -> None:
    """Figure 2: rollout nRMSE vs horizon k (line plot)."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    models = t["models"]

    id_to_model = {t["ids"][i]: models[i] for i in range(len(models))}

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.6))

    plotted_any = False
    for i, mid in enumerate(model_ids):
        m = id_to_model.get(mid)
        if not m:
            continue
        curve = m.get("rollout_curve_test", {}) or {}
        steps = curve.get("steps", [])
        metrics = curve.get("metrics", {}) or {}
        if not steps:
            continue
        ks = np.array([int(k) for k in steps], dtype=int)
        ys = np.array([float((metrics.get(str(int(k)), {}) or {}).get("nrmse", np.nan)) for k in ks], dtype=float)
        color = COLORS[i % len(COLORS)]
        ax.plot(ks, ys, marker="o", linewidth=2.0, markersize=5.5, color=color, label=mid)
        plotted_any = True

    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Rollout error (nRMSE, lower is better)")
    ax.set_title("Rollout error grows with horizon")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, 11, 1))
    _tighten(ax)
    if plotted_any:
        ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig2_rollout_vs_horizon_all(models_json: Path, out_path: Path) -> None:
    """Figure 2 (alt): rollout nRMSE vs horizon for ALL models."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    models = t["models"]

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.0))

    all_y: list[float] = []
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for mid, m in zip(t["ids"], models):
        curve = m.get("rollout_curve_test", {}) or {}
        steps = curve.get("steps", [])
        metrics = curve.get("metrics", {}) or {}
        if not steps:
            continue
        ks = np.array([int(k) for k in steps], dtype=int)
        ys = np.array([float((metrics.get(str(int(k)), {}) or {}).get("nrmse", np.nan)) for k in ks], dtype=float)
        curves.append((mid, ks, ys))
        all_y.extend([float(v) for v in ys if np.isfinite(v)])

    cap = None
    if all_y:
        cap = float(np.percentile(np.asarray(all_y, dtype=float), 97))
        cap = max(cap, 1.0)

    for i, (mid, ks, ys) in enumerate(curves):
        color = COLORS[i % len(COLORS)]
        ys_plot = ys.copy()
        blow = ~np.isfinite(ys_plot)
        if cap is not None:
            ys_plot[blow] = cap
            ys_plot = np.clip(ys_plot, 0.0, cap)
        ax.plot(ks, ys_plot, marker="o", linewidth=1.8, markersize=4.6, color=color, alpha=0.92, label=mid)
        if cap is not None and blow.any():
            ax.plot(ks[blow], ys_plot[blow], linestyle="none", marker="x", markersize=6.5, color=color)

    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("Rollout error (nRMSE, log scale; lower is better)")
    ax.set_title("Rollout error vs horizon (all models)")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, 11, 1))
    _tighten(ax)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig1_alt_paired_twoaxis_bars(models_json: Path, out_path: Path) -> None:
    """Figure 1 alt: paired bars (R² vs rollout nRMSE) with two y-axes."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"].astype(float)
    rollout = t["rollout10"].astype(float)

    _style()
    fig, ax1 = plt.subplots(1, 1, figsize=(10.6, 4.8))
    ax2 = ax1.twinx()

    x = np.arange(len(ids))
    w = 0.36

    # Rollout handling: cap extreme/inf for readability, but mark blow-ups.
    finite_roll = rollout[np.isfinite(rollout)]
    cap = float(np.percentile(finite_roll, 95)) if finite_roll.size else 1.0
    cap = max(cap, 1.0)
    roll_plot = rollout.copy()
    blow = ~np.isfinite(roll_plot)
    roll_plot[blow] = cap
    roll_plot = np.clip(roll_plot, 1e-9, cap)

    b1 = ax1.bar(x - w / 2, r2, width=w, color="#4C78A8", label="Time-test R²")
    b2 = ax2.bar(x + w / 2, roll_plot, width=w, color="#F58518", label="Rollout nRMSE (k=10)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ids)
    ax1.set_ylabel("Time-test R² (higher is better)")
    ax2.set_ylabel("Rollout error nRMSE @ k=10 (log; lower is better)")
    ax2.set_yscale("log")

    ax1.set_title("High one-step fit ≠ stable rollout")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2.grid(False)

    # Mark blow-ups (if any)
    if blow.any():
        for xi in x[blow]:
            ax2.text(float(xi + w / 2), cap, "∞", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Combined legend
    handles = [b1, b2]
    labels = ["Time-test R²", "Rollout nRMSE (k=10)"]
    ax1.legend(handles, labels, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig1_alt_slopegraph(models_json: Path, out_path: Path) -> None:
    """Figure 1 alt: slope graph from fit score to rollout score (both normalized, higher=better)."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = np.array(t["ids"], dtype=object)
    r2 = t["r2"].astype(float)
    rollout = t["rollout10"].astype(float)

    fit_score = _normalize_column(r2, higher_is_better=True)
    roll_score = _normalize_column(rollout, higher_is_better=False)

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8))

    x0, x1 = 0.0, 1.0
    order = np.argsort(roll_score)
    for j, i in enumerate(order):
        c = COLORS[j % len(COLORS)]
        y0 = float(fit_score[i])
        y1 = float(roll_score[i])
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=2.0, alpha=0.92)
        ax.scatter([x0, x1], [y0, y1], color=c, s=70, zorder=3)
        # Put model id inside markers (no external labels -> no overlap)
        ax.text(x0, y0, str(ids[i]).replace("M", ""), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(x1, y1, str(ids[i]).replace("M", ""), ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Fit score (R²)", "Rollout score (k=10)"])
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_title("Model ranking changes across objectives")
    _tighten(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig1_alt_rank_heatmap(models_json: Path, out_path: Path) -> None:
    """Figure 1 alt: rank-based view (fit rank vs rollout rank)."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"].astype(float)
    rollout = t["rollout10"].astype(float)

    # Ranks: 1 is best
    fit_order = np.argsort(-r2)
    fit_rank = np.empty_like(fit_order)
    fit_rank[fit_order] = np.arange(1, len(ids) + 1)

    roll_vals = rollout.copy()
    roll_vals[~np.isfinite(roll_vals)] = np.inf
    roll_order = np.argsort(roll_vals)
    roll_rank = np.empty_like(roll_order)
    roll_rank[roll_order] = np.arange(1, len(ids) + 1)

    mat = np.column_stack([fit_rank.astype(float), roll_rank.astype(float)])
    _style()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels(ids)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fit rank", "Rollout rank"])
    ax.set_title("Rank mismatch shows the tradeoff")

    # annotate ranks
    for i in range(len(ids)):
        ax.text(0, i, f"{int(fit_rank[i])}", ha="center", va="center", color="white", fontweight="bold")
        if np.isfinite(rollout[i]):
            ax.text(1, i, f"{int(roll_rank[i])}", ha="center", va="center", color="white", fontweight="bold")
        else:
            ax.text(1, i, "∞", ha="center", va="center", color="white", fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Rank value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig1_alt_compact_heatmap(models_json: Path, out_path: Path) -> None:
    """Figure 1 alt: compact heatmap of (R², rollout nRMSE) only."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"].astype(float)
    rollout = t["rollout10"].astype(float)

    # normalize: higher better for both
    fit = _normalize_column(r2, higher_is_better=True)
    roll = _normalize_column(rollout, higher_is_better=False)
    z = np.column_stack([fit, roll])

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4))
    im = ax.imshow(z, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels(ids)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fit (R²)", "Rollout (k=10)"])
    ax.set_title("Tradeoff heatmap (good = brighter)")

    # minimal in-cell markers
    for i in range(len(ids)):
        ax.text(0, i, str(ids[i]).replace("M", ""), ha="center", va="center", color="white", fontweight="bold")
        if not np.isfinite(rollout[i]):
            ax.text(1, i, "∞", ha="center", va="center", color="white", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_rank_slope_by_metric(models_json: Path, out_path: Path) -> None:
    """Strong extra: rank-by-metric slope plot (fit rank -> rollout rank)."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = np.array(t["ids"], dtype=object)
    r2 = t["r2"].astype(float)
    rollout = t["rollout10"].astype(float)

    # Lower rank is better; plot as "goodness" by flipping axis
    fit_order = np.argsort(-r2)
    fit_rank = np.empty_like(fit_order)
    fit_rank[fit_order] = np.arange(1, len(ids) + 1)

    roll_vals = rollout.copy()
    roll_vals[~np.isfinite(roll_vals)] = np.inf
    roll_order = np.argsort(roll_vals)
    roll_rank = np.empty_like(roll_order)
    roll_rank[roll_order] = np.arange(1, len(ids) + 1)

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8))
    x0, x1 = 0.0, 1.0

    for i in range(len(ids)):
        c = COLORS[i % len(COLORS)]
        y0 = -float(fit_rank[i])
        y1 = -float(roll_rank[i])
        ax.plot([x0, x1], [y0, y1], color=c, linewidth=2.0, alpha=0.92)
        ax.scatter([x0, x1], [y0, y1], color=c, s=70, zorder=3)
        ax.text(x0, y0, str(ids[i]).replace("M", ""), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(x1, y1, str(ids[i]).replace("M", ""), ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Rank by fit (R²)", "Rank by rollout (k=10)"])
    ax.set_ylabel("Rank (top is best)")
    ax.set_yticks([-1, -2, -3, -4, -5, -6])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_title("Tradeoff between short-term fit and long-term stability")
    _tighten(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _rollout_nrmse_from_best_model_payload(p: dict) -> float:
    r = p.get("rollout_time_test", {}) or {}
    steps = r.get("steps", [])
    metrics = r.get("metrics", {}) or {}
    if not steps:
        return float("nan")
    k = int(steps[-1])
    return float((metrics.get(str(k), {}) or {}).get("nrmse", np.nan))


def _one_step_rmse_from_best_model_payload(p: dict) -> float:
    return float(p.get("one_step_rmse", np.nan))


def fig_fig3_stabilization_effect(slides_dir: Path, out_path: Path) -> None:
    """Figure 3: Model 4 stabilization variants (connected dots) vs rollout nRMSE at largest k."""
    baseline_p = slides_dir / "best_model_baseline.json"
    trans_p = slides_dir / "best_model_stabilized_translation.json"
    tofirst_p = slides_dir / "best_model_stab_to_first_sigma2.json"

    payloads: list[tuple[str, dict]] = []
    for label, path in [
        ("baseline", baseline_p),
        ("+translation", trans_p),
        ("to_first (σ=2)", tofirst_p),
    ]:
        if path.exists():
            payloads.append((label, _read_json(path)))

    labels = [p[0] for p in payloads]
    vals = np.array([_rollout_nrmse_from_best_model_payload(p[1]) for p in payloads], dtype=float)

    _style()
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.6))
    x = np.arange(len(labels), dtype=float)
    c = "#4C78A8"
    ax.plot(x, vals, marker="o", linewidth=2.4, markersize=7.0, color=c)
    for xi, yi in zip(x, vals):
        if np.isfinite(yi):
            ax.text(float(xi), float(yi), _sig(float(yi), 3), ha="center", va="bottom", fontsize=10, color=c)
        else:
            ax.text(float(xi), 1.0, "∞", ha="center", va="bottom", fontsize=11, fontweight="bold", color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rollout error (nRMSE @ k=10, lower is better)")
    ax.set_title("Stabilization reduces long-horizon error")
    ax.set_yscale("log")
    _tighten(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_fig3b_metric_disentanglement(slides_dir: Path, out_path: Path) -> None:
    """Figure 3B: two small panels showing one-step is flat while rollout improves."""
    baseline_p = slides_dir / "best_model_baseline.json"
    trans_p = slides_dir / "best_model_stabilized_translation.json"
    tofirst_p = slides_dir / "best_model_stab_to_first_sigma2.json"

    payloads: list[tuple[str, dict]] = []
    for label, path in [
        ("baseline", baseline_p),
        ("+translation", trans_p),
        ("to_first (σ=2)", tofirst_p),
    ]:
        if path.exists():
            payloads.append((label, _read_json(path)))

    labels = [p[0] for p in payloads]
    one_step = np.array([_one_step_rmse_from_best_model_payload(p[1]) for p in payloads], dtype=float)
    roll = np.array([_rollout_nrmse_from_best_model_payload(p[1]) for p in payloads], dtype=float)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    x = np.arange(len(labels), dtype=float)

    axes[0].plot(x, one_step, marker="o", linewidth=2.2, markersize=6.5, color="#4C78A8")
    axes[0].set_title("One-step error (nearly unchanged)")
    axes[0].set_ylabel("One-step RMSE")
    _tighten(axes[0])

    axes[1].plot(x, roll, marker="o", linewidth=2.2, markersize=6.5, color="#F58518")
    axes[1].set_title("Rollout error (improves)")
    axes[1].set_ylabel("Rollout nRMSE @ k=10")
    axes[1].set_yscale("log")
    _tighten(axes[1])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    fig.suptitle("Stabilization affects stability more than fit", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_pres1_models_heatmap(models_json: Path, out_path: Path) -> None:
    """Wide heatmap: models (rows) vs metrics (cols)."""
    d = _read_json(models_json)
    models = d.get("models", [])

    names = [m.get("name", "") for m in models]
    rows = [_short_model_name(n) for n in names]

    r2 = np.array([float(m.get("r2_test", np.nan)) for m in models])
    one_step = np.array([float(m.get("one_step_rmse", np.nan)) for m in models])
    rollout = np.array([float((m.get("rollout", {}).get("test", {}) or {}).get("nrmse", np.nan)) for m in models])
    active = np.array([float(m.get("n_active", np.nan)) for m in models])

    # Cap stability for color scaling (still annotate true values)
    rollout_plot = np.array(rollout, copy=True)
    rollout_plot[~np.isfinite(rollout_plot)] = np.nan
    if np.isfinite(rollout_plot).any():
        cap = float(np.nanpercentile(rollout_plot, 90))
        rollout_plot = np.clip(rollout_plot, 0.0, max(cap, 1.0))

    z = np.column_stack(
        [
            _normalize_column(r2, higher_is_better=True),
            _normalize_column(one_step, higher_is_better=False),
            _normalize_column(rollout_plot, higher_is_better=False),
            _normalize_column(active, higher_is_better=False),
        ]
    )

    cols = [
        _wrap_metric_label("Fit: R² (↑)"),
        _wrap_metric_label("Error: RMSE (↓)"),
        _wrap_metric_label("Stability: rollout k=10 nRMSE (↓)"),
        _wrap_metric_label("Complexity: #active terms (↓)"),
    ]

    # Order rows by stability-first composite (easy story for slides)
    composite = 0.45 * z[:, 2] + 0.35 * z[:, 0] + 0.20 * z[:, 3]
    order = np.argsort(-composite)
    rows = [rows[i] for i in order]
    r2 = r2[order]
    one_step = one_step[order]
    rollout = rollout[order]
    active = active[order]
    z = z[order, :]

    fig, ax = plt.subplots(figsize=(15.5, 5.6), constrained_layout=True)
    im = ax.imshow(z, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=0)
    ax.set_title("Latest model comparison (color = better)")

    # Highlight: best fit and best stability
    best_fit_i = int(np.nanargmax(r2)) if np.isfinite(r2).any() else 0
    rollout_finite = np.where(np.isfinite(rollout), rollout, np.nan)
    best_stab_i = int(np.nanargmin(rollout_finite)) if np.isfinite(rollout_finite).any() else 0

    ax.scatter([0], [best_fit_i], s=220, marker="o", color="white", edgecolor="black", linewidth=1.2, zorder=5)
    ax.scatter([2], [best_stab_i], s=220, marker="o", color="white", edgecolor="black", linewidth=1.2, zorder=5)
    ax.text(0.02, -0.08, "white circles: best R² and best rollout", transform=ax.transAxes, fontsize=11)

    # Annotate lightly: keep only stability + R² to avoid clutter.
    for i in range(len(rows)):
        ann = [
            f"{r2[i]:.2f}" if np.isfinite(r2[i]) else "?",
            "",
            ("∞" if not np.isfinite(rollout[i]) else f"{rollout[i]:.2f}"),
            "",
        ]
        for j, s in enumerate(ann):
            if not s:
                continue
            ax.text(j, i, s, ha="center", va="center", fontsize=11, color="white" if z[i, j] < 0.45 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Relative score (0=worst, 1=best)")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_pres2_rollout_bars_tall(models_json: Path, out_path: Path) -> None:
    """Tall horizontal bars: rollout k=10 nRMSE per model (log scale); color encodes R²."""
    d = _read_json(models_json)
    models = d.get("models", [])

    labels = [_short_model_name(str(m.get("name", ""))) for m in models]
    r2 = np.array([float(m.get("r2_test", np.nan)) for m in models])
    rollout = np.array([float((m.get("rollout", {}).get("test", {}) or {}).get("nrmse", np.nan)) for m in models])

    # Order by stability (best on top)
    rollout_for_sort = np.where(np.isfinite(rollout), rollout, np.nan)
    order = np.argsort(rollout_for_sort)
    labels = [labels[i] for i in order]
    r2 = r2[order]
    rollout = rollout[order]

    # Replace non-finite with a large number for plotting (still annotate as ∞)
    finite_vals = rollout[np.isfinite(rollout)]
    cap = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    rollout_plot = np.where(np.isfinite(rollout), rollout, cap * 1.6)
    rollout_plot = np.maximum(rollout_plot, 1e-3)

    fig, ax = plt.subplots(figsize=(7.6, 10.4), constrained_layout=True)

    # Color = normalized R²
    r2_norm = _normalize_column(r2, higher_is_better=True)
    cmap = plt.get_cmap("viridis")
    colors = cmap(r2_norm)

    y = np.arange(len(labels))
    bars = ax.barh(y, rollout_plot, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Rollout k=10 nRMSE (log scale, lower = more stable)")
    ax.set_title("Stability across models")
    ax.grid(True, axis="x", alpha=0.25)

    # Add margin so labels don't get cut off.
    xmax = float(np.nanmax(rollout_plot)) if np.isfinite(rollout_plot).any() else 1.0
    ax.set_xlim(left=max(1e-3, float(np.nanmin(rollout_plot)) * 0.8), right=xmax * 3.0)

    for i, b in enumerate(bars):
        val = rollout[i]
        txt = "∞" if not np.isfinite(val) else f"{val:.2f}"
        ax.text(b.get_width() * 1.10, b.get_y() + b.get_height() / 2, txt, va="center", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=float(np.nanmin(r2)), vmax=float(np.nanmax(r2))))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.06, pad=0.02)
    cbar.set_label("One-step fit: time-test R² (higher = better)")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


@dataclass(frozen=True)
class ApproachPoint:
    label: str
    r2: float
    rollout10: float


def _load_best_model_point(p: Path, label: str) -> ApproachPoint:
    d = _read_json(p)
    r2 = float(d.get("r2", np.nan))
    rollout10 = float((d.get("rollout_time_test", {}).get("metrics", {}) or {}).get("10", {}).get("nrmse", np.nan))
    return ApproachPoint(label=label, r2=r2, rollout10=rollout10)


def fig_approaches_scatter(out_path: Path) -> None:
    """Scatter: each point is an approach; x=fit, y=stability."""
    pts: list[ApproachPoint] = []

    # Model 4 under different stabilization toggles (approaches)
    base = SLIDES_DIR / "best_model_baseline.json"
    stab = SLIDES_DIR / "best_model_stabilized_translation.json"
    stab2 = SLIDES_DIR / "best_model_stab_to_first_sigma2.json"

    if base.exists():
        pts.append(_load_best_model_point(base, "M4: baseline"))
    if stab.exists():
        pts.append(_load_best_model_point(stab, "M4: +translation stab"))
    if stab2.exists():
        pts.append(_load_best_model_point(stab2, "M4: to_first, σ=2"))

    # Add the stable Model 3 reference
    m3 = SLIDES_DIR / "best_model.json"
    if m3.exists():
        pts.append(_load_best_model_point(m3, "M3: stable"))

    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)

    for i, p in enumerate(pts):
        ax.scatter(p.r2, p.rollout10, s=160, color=COLORS[i % len(COLORS)], edgecolor="black", linewidth=1.0)
        ax.text(p.r2, p.rollout10, f"  {p.label}", ha="left", va="center", fontsize=11)

    ax.set_title("Approach comparison: fit vs stability")
    ax.set_xlabel("One-step fit: time-test R² (higher is better)")
    ax.set_ylabel("Stability: rollout k=10 nRMSE (lower is better)")

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Visual cue arrows (self-explanatory)
    ax.annotate("better fit →", xy=(0.98, 0.08), xycoords="axes fraction", ha="right", va="center", fontsize=11)
    ax.annotate("more stable ↓", xy=(0.02, 0.92), xycoords="axes fraction", ha="left", va="center", fontsize=11)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def fig_pres3_fit_vs_stability_square(models_json: Path, out_path: Path) -> None:
    """Square scatter: all models, fit vs stability; marker size = complexity."""
    d = _read_json(models_json)
    models = d.get("models", [])

    names = [_short_model_name(str(m.get("name", ""))) for m in models]
    r2 = np.array([float(m.get("r2_test", np.nan)) for m in models])
    rollout = np.array([float((m.get("rollout", {}).get("test", {}) or {}).get("nrmse", np.nan)) for m in models])
    active = np.array([float(m.get("n_active", np.nan)) for m in models])

    # Replace inf for plotting y-scale; keep annotations
    finite = rollout[np.isfinite(rollout)]
    y_cap = float(np.nanpercentile(finite, 95)) if finite.size else 10.0
    y_plot = np.where(np.isfinite(rollout), rollout, y_cap * 1.6)

    sizes = 80 + 30 * np.nan_to_num(active, nan=0.0)

    fig, ax = plt.subplots(figsize=(7.8, 7.8), constrained_layout=True)

    # Use short IDs inside markers; avoid label collisions.
    ids = [_model_id(m.get("name", "")) for m in models]
    for i, mid in enumerate(ids):
        ax.scatter(
            r2[i],
            y_plot[i],
            s=float(sizes[i]),
            color=COLORS[i % len(COLORS)],
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            mid,
            (r2[i], y_plot[i]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_title("Fit vs stability (all models)")
    ax.set_xlabel("One-step fit: time-test R² (higher is better)")
    ax.set_ylabel("Stability: rollout k=10 nRMSE (log scale, lower is better)")
    ax.grid(True, alpha=0.25)
    ax.annotate("better fit →", xy=(0.98, 0.08), xycoords="axes fraction", ha="right", va="center", fontsize=11)
    ax.annotate("more stable ↓", xy=(0.02, 0.92), xycoords="axes fraction", ha="left", va="center", fontsize=11)
    ax.text(0.02, 0.02, "marker size = #active terms", transform=ax.transAxes, fontsize=11)
    ax.text(0.02, -0.10, "IDs: M1..M6 correspond to the term-library variants", transform=ax.transAxes, fontsize=11)

    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def fig_pres4_m3_vs_m4_dumbbell(models_json: Path, out_path: Path) -> None:
    """Wide dumbbell: compare coefficients of Model 3 vs Model 4 on shared terms."""
    d = _read_json(models_json)
    models = d.get("models", [])

    def find(prefix: str) -> dict | None:
        for m in models:
            if str(m.get("name", "")).startswith(prefix):
                return m
        return None

    m3 = find("Model 3")
    m4 = find("Model 4")
    if m3 is None or m4 is None:
        return

    def coeff_map(m: dict) -> dict[str, float]:
        names = list(m.get("terms", []))
        coeffs = list(m.get("coeffs", []))
        mp = {str(n): float(c) for n, c in zip(names, coeffs)}
        mp.pop("1", None)
        return mp

    c3 = coeff_map(m3)
    c4 = coeff_map(m4)
    terms = ["u", "u_x", "u_y", "lap(u)", "u^2"]

    x3 = np.array([c3.get(t, 0.0) for t in terms])
    x4 = np.array([c4.get(t, 0.0) for t in terms])

    y = np.arange(len(terms))

    fig, ax = plt.subplots(figsize=(14.5, 4.4), constrained_layout=True)
    for i in range(len(terms)):
        ax.plot([x3[i], x4[i]], [y[i], y[i]], color="gray", lw=2.0, alpha=0.7, zorder=1)
    ax.scatter(x3, y, s=120, color=COLORS[0], edgecolor="black", linewidth=1.0, label="M3 (stable)", zorder=3)
    ax.scatter(x4, y, s=120, color=COLORS[1], edgecolor="black", linewidth=1.0, label="M4 (best one-step fit)", zorder=3)

    ax.axvline(0.0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.set_xlabel("Coefficient value (symlog)")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_title("Coefficient shift: stable (M3) → best one-step fit (M4)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", ncol=2)

    for i in range(len(terms)):
        ax.text(x3[i], y[i] + 0.18, f"{x3[i]:.3g}", fontsize=10, color=COLORS[0], ha="center")
        ax.text(x4[i], y[i] - 0.22, f"{x4[i]:.3g}", fontsize=10, color=COLORS[1], ha="center")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_pick_terms_presence_heatmap(models_json: Path, out_path: Path) -> None:
    """Extra: heatmap showing which terms exist in each model library."""
    d = _read_json(models_json)
    models = d.get("models", [])

    model_ids = [_model_id(m.get("name", "")) for m in models]
    term_sets = [set(map(str, m.get("terms", []))) - {"1"} for m in models]

    # Union of all terms across models, sorted by rough complexity then alpha
    all_terms = sorted(set().union(*term_sets))
    def term_key(t: str) -> tuple[int, str]:
        if "^" in t or "*" in t:
            return (2, t)
        if "_" in t or "lap" in t:
            return (1, t)
        return (0, t)
    all_terms = sorted(all_terms, key=term_key)

    Z = np.zeros((len(all_terms), len(models)), dtype=float)
    for j, s in enumerate(term_sets):
        for i, t in enumerate(all_terms):
            Z[i, j] = 1.0 if t in s else 0.0

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Which terms are included in each model?")
    ax.set_xticks(np.arange(len(model_ids)))
    ax.set_xticklabels(model_ids)
    ax.set_yticks(np.arange(len(all_terms)))
    ax.set_yticklabels(all_terms)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_pick_approaches_bars(out_path: Path) -> None:
    """Extra: bar chart comparing approaches (best_model JSONs) on rollout stability; color = R²."""
    candidates = [
        ("M3 stable", SLIDES_DIR / "best_model.json"),
        ("M4 baseline", SLIDES_DIR / "best_model_baseline.json"),
        ("M4 +translation stab", SLIDES_DIR / "best_model_stabilized_translation.json"),
        ("M4 to_first, σ=2", SLIDES_DIR / "best_model_stab_to_first_sigma2.json"),
    ]
    labels: list[str] = []
    r2s: list[float] = []
    roll10: list[float] = []
    for label, path in candidates:
        if not path.exists():
            continue
        d = _read_json(path)
        labels.append(label)
        r2s.append(float(d.get("r2", np.nan)))
        roll10.append(float((d.get("rollout_time_test", {}).get("metrics", {}) or {}).get("10", {}).get("nrmse", np.nan)))

    if not labels:
        return

    r2 = np.array(r2s, dtype=float)
    rollout = np.array(roll10, dtype=float)

    order = np.argsort(np.where(np.isfinite(rollout), rollout, np.nan))
    labels = [labels[i] for i in order]
    r2 = r2[order]
    rollout = rollout[order]

    finite = rollout[np.isfinite(rollout)]
    cap = float(np.nanmax(finite)) if finite.size else 1.0
    rollout_plot = np.where(np.isfinite(rollout), rollout, cap * 1.6)
    rollout_plot = np.maximum(rollout_plot, 1e-3)

    fig, ax = plt.subplots(figsize=(12.8, 4.8), constrained_layout=True)
    r2_norm = _normalize_column(r2, higher_is_better=True)
    cmap = plt.get_cmap("viridis")
    colors = cmap(r2_norm)

    x = np.arange(len(labels))
    bars = ax.bar(x, rollout_plot, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rollout k=10 nRMSE (log scale, lower is better)")
    ax.set_title("Approach variants (same data, different stabilization / model family)")
    ax.grid(True, axis="y", alpha=0.25)

    for i, b in enumerate(bars):
        val = rollout[i]
        txt = "∞" if not np.isfinite(val) else f"{val:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, txt, ha="center", va="bottom", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=float(np.nanmin(r2)), vmax=float(np.nanmax(r2))))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("One-step fit: time-test R² (higher = better)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_paper1_score_stacked_bars(models_json: Path, out_path: Path) -> None:
    """Paper-style: stacked horizontal bars (normalized scores), one chart type."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"]
    one_step = t["one_step"]
    rollout10 = t["rollout10"]
    active = t["active"]

    # Cap extreme rollout for score only, so blow-ups don't dominate normalization.
    roll = np.array(rollout10, copy=True)
    roll[~np.isfinite(roll)] = np.nan
    if np.isfinite(roll).any():
        roll = np.clip(roll, 0.0, float(np.nanpercentile(roll, 90)))

    s_r2 = _normalize_column(r2, higher_is_better=True)
    s_step = _normalize_column(one_step, higher_is_better=False)
    s_roll = _normalize_column(roll, higher_is_better=False)
    s_comp = _normalize_column(active, higher_is_better=False)

    # Stability-first weights (simple, explainable)
    w = np.array([0.35, 0.25, 0.30, 0.10])
    S = np.column_stack([s_r2, s_step, s_roll, s_comp])
    total = S @ w

    order = np.argsort(-total)
    ids = [ids[i] for i in order]
    S = S[order, :]
    total = total[order]

    fig, ax = plt.subplots(figsize=(14.8, 5.2), constrained_layout=True)
    y = np.arange(len(ids))
    left = np.zeros_like(y, dtype=float)

    seg_labels = ["Fit (R²)", "One-step", "Rollout", "Simplicity"]
    seg_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for j in range(S.shape[1]):
        ax.barh(y, S[:, j], left=left, color=seg_colors[j], edgecolor="white", height=0.7, label=seg_labels[j])
        left = left + S[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Normalized score (0 worst → 1 best)")
    ax.set_title("Overall model ranking (normalized, stability-first)")
    ax.legend(ncol=4, loc="lower right")

    # No numeric annotations (prevents text overlaps on small screens)

    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def fig_paper2_pareto_scatter(models_json: Path, out_path: Path) -> None:
    """Paper-style: pareto scatter, minimal text; labels are inside markers."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"]
    rollout10 = t["rollout10"]
    active = t["active"]

    finite = rollout10[np.isfinite(rollout10)]
    y_cap = float(np.nanpercentile(finite, 95)) if finite.size else 10.0
    y_plot = np.where(np.isfinite(rollout10), rollout10, y_cap * 1.6)

    sizes = 90 + 30 * np.nan_to_num(active, nan=0.0)
    # Color by complexity (active terms), easy to read.
    c = np.nan_to_num(active, nan=0.0)

    fig, ax = plt.subplots(figsize=(7.8, 7.8), constrained_layout=True)
    sc = ax.scatter(r2, y_plot, s=sizes, c=c, cmap="viridis", edgecolor="black", linewidth=1.0, zorder=3)

    for i, mid in enumerate(ids):
        ax.annotate(mid, (r2[i], y_plot[i]), ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("One-step fit: time-test R² (higher is better)")
    ax.set_ylabel("Stability: rollout k=10 nRMSE (log, lower is better)")
    ax.set_title("Fit vs stability (all models)")
    ax.annotate("better fit →", xy=(0.98, 0.08), xycoords="axes fraction", ha="right", va="center", fontsize=11)
    ax.annotate("more stable ↓", xy=(0.02, 0.92), xycoords="axes fraction", ha="left", va="center", fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Model complexity (#active terms)")

    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def fig_paper3_coeff_heatmap(models_json: Path, out_path: Path) -> None:
    """Paper-style: coefficient heatmap across models and terms (signed log transform)."""
    d = _read_json(models_json)
    models = d.get("models", [])
    ids = [_model_id(m.get("name", "")) for m in models]

    # Use a small, readable key-term list (prevents y-label overlaps).
    key_terms = [
        "u",
        "u_x",
        "u_y",
        "lap(u)",
        "u_xx",
        "u_yy",
        "u^2",
        "u*u_x",
        "u*u_y",
        "u^3",
        "u_x^2",
        "u_y^2",
    ]
    term_sets = [set(map(str, m.get("terms", []))) - {"1"} for m in models]
    present = set().union(*term_sets)
    terms = [t for t in key_terms if t in present]

    def coeff_map(m: dict) -> dict[str, float]:
        names = list(m.get("terms", []))
        coeffs = list(m.get("coeffs", []))
        mp = {str(n): float(c) for n, c in zip(names, coeffs)}
        mp.pop("1", None)
        return mp

    C = np.zeros((len(terms), len(models)), dtype=float)
    for j, m in enumerate(models):
        mp = coeff_map(m)
        for i, t in enumerate(terms):
            C[i, j] = mp.get(t, 0.0)

    # Signed log compression: preserves sign, compresses magnitude.
    scale = np.nanmedian(np.abs(C[C != 0])) if np.any(C != 0) else 1.0
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    Z = np.sign(C) * np.log10(1.0 + np.abs(C) / scale)

    fig, ax = plt.subplots(figsize=(14.8, 5.2), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels(ids)
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_title("Discovered PDE coefficients (signed log-scaled)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("signed log10(1 + |c| / median|c|)")

    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def fig_paper4_approach_two_metric_bars(out_path: Path) -> None:
    """Paper-style: two aligned bar panels (same chart type) for approach variants."""
    candidates = [
        ("M3", "stable", SLIDES_DIR / "best_model.json"),
        ("M4", "baseline", SLIDES_DIR / "best_model_baseline.json"),
        ("M4", "+translation", SLIDES_DIR / "best_model_stabilized_translation.json"),
        ("M4", "to_first σ=2", SLIDES_DIR / "best_model_stab_to_first_sigma2.json"),
    ]

    labels: list[str] = []
    r2s: list[float] = []
    roll10: list[float] = []
    for mid, tag, path in candidates:
        if not path.exists():
            continue
        d = _read_json(path)
        labels.append(f"{mid} {tag}")
        r2s.append(float(d.get("r2", np.nan)))
        roll10.append(float((d.get("rollout_time_test", {}).get("metrics", {}) or {}).get("10", {}).get("nrmse", np.nan)))

    if not labels:
        return

    r2 = np.array(r2s, dtype=float)
    rollout = np.array(roll10, dtype=float)
    finite = rollout[np.isfinite(rollout)]
    cap = float(np.nanmax(finite)) if finite.size else 1.0
    rollout_plot = np.where(np.isfinite(rollout), rollout, cap * 1.6)
    rollout_plot = np.maximum(rollout_plot, 1e-3)

    order = np.argsort(rollout_plot)
    labels = [labels[i] for i in order]
    r2 = r2[order]
    rollout = rollout[order]
    rollout_plot = rollout_plot[order]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.6), constrained_layout=True, sharex=False)
    x = np.arange(len(labels))

    # Panel A: R²
    axes[0].bar(x, r2, color="#4C78A8", edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("time-test R² (higher is better)")
    axes[0].set_title("Fit")
    for i, v in enumerate(r2):
        axes[0].text(i, v + 0.02, _round2(float(v)), ha="center", va="bottom", fontsize=10)

    # Panel B: Rollout
    axes[1].bar(x, rollout_plot, color="#54A24B", edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rollout k=10 nRMSE (log, lower is better)")
    axes[1].set_title("Stability")
    for i, v in enumerate(rollout):
        axes[1].text(i, rollout_plot[i] * 1.12, "∞" if not np.isfinite(v) else _round2(float(v)), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Approach variants")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def fig_paper5_patch_stability_heatmap(coeff_csv: Path, out_path: Path) -> None:
    """Paper-style: patch term stability heatmap (no crowded numbers)."""
    if not coeff_csv.exists():
        return

    rows: list[dict] = []
    with Path(coeff_csv).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("term") in {"", "1"}:
                continue
            rows.append(r)

    terms = [r["term"] for r in rows]
    nonzero = np.array([float(r["nonzero_freq"]) for r in rows])
    sign = np.array([float(r["sign_stability"]) for r in rows])
    agg = np.array([abs(float(r["agg_coeff"])) for r in rows])
    q25 = np.array([float(r.get("q25", 0.0)) for r in rows])
    q75 = np.array([float(r.get("q75", 0.0)) for r in rows])
    iqr = np.abs(q75 - q25)

    order = np.argsort(-nonzero)
    terms = [terms[i] for i in order]
    nonzero = nonzero[order]
    sign = sign[order]
    agg = agg[order]
    iqr = iqr[order]

    agg_n = _normalize_column(agg, higher_is_better=True)
    iqr_n = _normalize_column(iqr, higher_is_better=False)
    Z = np.column_stack([nonzero, sign, agg_n, iqr_n])

    fig, ax = plt.subplots(figsize=(12.8, 4.8), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Presence", "Sign", "|coeff|", "Certainty"])
    ax.set_title("Patch-based stability of discovered terms")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Score")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def fig_mini1_rank_bars(models_json: Path, out_path: Path) -> None:
    """Ultra-minimal: one horizontal bar chart, total score only."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"]
    one_step = t["one_step"]
    rollout10 = t["rollout10"]
    active = t["active"]

    roll = np.array(rollout10, copy=True)
    roll[~np.isfinite(roll)] = np.nan
    if np.isfinite(roll).any():
        roll = np.clip(roll, 0.0, float(np.nanpercentile(roll, 90)))

    S = np.column_stack(
        [
            _normalize_column(r2, higher_is_better=True),
            _normalize_column(one_step, higher_is_better=False),
            _normalize_column(roll, higher_is_better=False),
            _normalize_column(active, higher_is_better=False),
        ]
    )
    w = np.array([0.35, 0.25, 0.30, 0.10])
    total = S @ w
    order = np.argsort(-total)
    ids = [ids[i] for i in order]
    total = total[order]

    fig, ax = plt.subplots(figsize=(12.8, 4.6), constrained_layout=True)
    y = np.arange(len(ids))
    ax.barh(y, total, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Overall score (0–1)")
    ax.set_title("Best model (stability-first)")
    _tighten(ax)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def fig_mini2_pareto(models_json: Path, out_path: Path) -> None:
    """Ultra-minimal: scatter with IDs only (no extra arrows/text)."""
    d = _read_json(models_json)
    t = _extract_models_table(d)
    ids = t["ids"]
    r2 = t["r2"]
    rollout10 = t["rollout10"]

    finite = rollout10[np.isfinite(rollout10)]
    y_cap = float(np.nanpercentile(finite, 95)) if finite.size else 10.0
    y_plot = np.where(np.isfinite(rollout10), rollout10, y_cap * 1.6)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    ax.scatter(r2, y_plot, s=220, color="#54A24B", edgecolor="black", linewidth=1.0)
    for i, mid in enumerate(ids):
        ax.annotate(mid, (r2[i], y_plot[i]), ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("R²")
    ax.set_ylabel("rollout nRMSE (k=10, log)")
    ax.set_title("Fit vs stability")
    _tighten(ax)
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def fig_mini3_coeff_keyterms(models_json: Path, out_path: Path) -> None:
    """Ultra-minimal: heatmap of key-term coefficients only, no extra text."""
    d = _read_json(models_json)
    models = d.get("models", [])
    ids = [_model_id(m.get("name", "")) for m in models]

    key_terms = ["u", "u_x", "u_y", "lap(u)", "u^2", "u*u_x", "u*u_y"]

    def coeff_map(m: dict) -> dict[str, float]:
        names = list(m.get("terms", []))
        coeffs = list(m.get("coeffs", []))
        mp = {str(n): float(c) for n, c in zip(names, coeffs)}
        mp.pop("1", None)
        return mp

    present = set()
    for m in models:
        present |= (set(map(str, m.get("terms", []))) - {"1"})
    terms = [t for t in key_terms if t in present]

    C = np.zeros((len(terms), len(models)), dtype=float)
    for j, m in enumerate(models):
        mp = coeff_map(m)
        for i, t in enumerate(terms):
            C[i, j] = mp.get(t, 0.0)

    scale = np.nanmedian(np.abs(C[C != 0])) if np.any(C != 0) else 1.0
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    Z = np.sign(C) * np.log10(1.0 + np.abs(C) / scale)
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0

    fig, ax = plt.subplots(figsize=(12.0, 4.4), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels(ids)
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_title("Coefficients (key terms)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def fig_mini4_approach_bars(out_path: Path) -> None:
    """Ultra-minimal: stability-only bars for approach variants."""
    candidates = [
        ("M3", SLIDES_DIR / "best_model.json"),
        ("M4 base", SLIDES_DIR / "best_model_baseline.json"),
        ("M4 +trans", SLIDES_DIR / "best_model_stabilized_translation.json"),
        ("M4 σ2", SLIDES_DIR / "best_model_stab_to_first_sigma2.json"),
    ]
    labels: list[str] = []
    vals: list[float] = []
    for lab, p in candidates:
        if not p.exists():
            continue
        d = _read_json(p)
        v = float((d.get("rollout_time_test", {}).get("metrics", {}) or {}).get("10", {}).get("nrmse", np.nan))
        labels.append(lab)
        vals.append(v)
    if not labels:
        return

    v = np.array(vals, dtype=float)
    finite = v[np.isfinite(v)]
    cap = float(np.nanmax(finite)) if finite.size else 1.0
    v_plot = np.where(np.isfinite(v), v, cap * 1.6)
    v_plot = np.maximum(v_plot, 1e-3)
    order = np.argsort(v_plot)
    labels = [labels[i] for i in order]
    v_plot = v_plot[order]

    fig, ax = plt.subplots(figsize=(12.0, 4.4), constrained_layout=True)
    ax.bar(labels, v_plot, color="#E45756", edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("rollout nRMSE (k=10, log)")
    ax.set_title("Approach stability")
    ax.tick_params(axis="x", rotation=10)
    _tighten(ax)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def fig_tradeoff_scatter_gold(models_json: Path, out_path: Path) -> None:
    """One-step fit vs rollout stability tradeoff (gold slide figure).

    X-axis: time-test R^2
    Y-axis: rollout k=10 nRMSE (log)
    One point per model (Models 1–6)
    """
    d = _read_json(models_json)
    t = _extract_models_table(d)
    models = t["models"]
    ids = t["ids"]
    r2 = t["r2"]
    rollout10 = t["rollout10"]

    # Plot-friendly y values (keep the story, avoid infinite axis).
    finite = rollout10[np.isfinite(rollout10)]
    if finite.size:
        y_cap = float(np.nanpercentile(finite, 95))
        y_plot = np.where(np.isfinite(rollout10), rollout10, y_cap * 2.0)
        y_plot = np.clip(y_plot, 1e-3, max(y_cap * 2.0, 1.0))
    else:
        y_plot = np.ones_like(rollout10)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)

    # Color by model index for clarity; use consistent styling.
    for i, mid in enumerate(ids):
        ax.scatter(r2[i], y_plot[i], s=260, color=COLORS[i % len(COLORS)], edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(mid, (r2[i], y_plot[i]), ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=4)

    ax.set_yscale("log")
    ax.set_xlabel("One-step fit: time-test R² (higher is better)")
    ax.set_ylabel("Rollout error: k=10 nRMSE (log scale, lower is better)")
    ax.set_title("One-step fit vs rollout stability tradeoff")
    _tighten(ax)

    # Explicit callouts (minimal, positioned with offsets to avoid overlap)
    id_to_idx = {ids[i]: i for i in range(len(ids))}

    def callout(mid: str, text: str, dx: int, dy: int) -> None:
        i = id_to_idx.get(mid)
        if i is None:
            return
        ax.annotate(
            text,
            xy=(r2[i], y_plot[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=12,
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "black", "lw": 0.8},
            zorder=5,
        )

    # We know the narrative models are M3 and M4.
    callout("M4", "M4: good fit\npoor stability", dx=25, dy=30)
    callout("M3", "M3: weaker fit\nmore stable", dx=-170, dy=-10)

    # Blow-up callout for M5/M6 (if present)
    for mid in ("M5", "M6"):
        i = id_to_idx.get(mid)
        if i is None:
            continue
        raw = float(rollout10[i])
        label = "unusable (blow-up)" if not np.isfinite(raw) else f"unusable (~{_sig(raw, 2)})"
        callout(mid, f"{mid}: {label}", dx=25, dy=-60)

    # Optional guide line: visually separate stable vs unstable.
    if finite.size:
        thresh = float(np.nanmedian(finite))
        ax.axhline(thresh, color="black", lw=1.0, alpha=0.25)

    # Presentation line on figure (small, bottom-right)
    ax.text(
        0.99,
        0.02,
        "Better one-step fit ≠ stable dynamics",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> int:
    _style()

    models_json = SLIDES_DIR / "models_comparison.json"
    if not models_json.exists():
        raise FileNotFoundError(models_json)

    # Presentation-ready set (tomorrow)
    fig_pres1_models_heatmap(models_json, SLIDES_DIR / "PRES1_MODELS_HEATMAP_WIDE.png")
    fig_pres2_rollout_bars_tall(models_json, SLIDES_DIR / "PRES2_ROLLOUT_BARS_TALL.png")
    fig_pres3_fit_vs_stability_square(models_json, SLIDES_DIR / "PRES3_FIT_STABILITY_SQUARE.png")
    fig_pres4_m3_vs_m4_dumbbell(models_json, SLIDES_DIR / "PRES4_M3_VS_M4_DUMBBELL_WIDE.png")

    # Extra comparative options (pick what you like)
    fig_pick_terms_presence_heatmap(models_json, SLIDES_DIR / "PICK_TERMS_PRESENCE_HEATMAP.png")
    fig_pick_approaches_bars(SLIDES_DIR / "PICK_APPROACHES_BARS.png")

    # Paper-grade options (minimal text, rounded labels, clean encodings)
    fig_paper1_score_stacked_bars(models_json, SLIDES_DIR / "PAPER1_SCORE_STACKED_BARS_WIDE.png")
    fig_paper2_pareto_scatter(models_json, SLIDES_DIR / "PAPER2_PARETO_SCATTER_SQUARE.png")
    fig_paper3_coeff_heatmap(models_json, SLIDES_DIR / "PAPER3_COEFF_HEATMAP_WIDE.png")
    fig_paper4_approach_two_metric_bars(SLIDES_DIR / "PAPER4_APPROACH_BARS_WIDE.png")
    fig_paper5_patch_stability_heatmap(PATCH_CSV, SLIDES_DIR / "PAPER5_PATCH_STABILITY_HEATMAP_WIDE.png")

    # Ultra-minimal set (no overlaps, easy to read from far)
    fig_mini1_rank_bars(models_json, SLIDES_DIR / "MINI1_RANK_BARS_WIDE.png")
    fig_mini2_pareto(models_json, SLIDES_DIR / "MINI2_PARETO_SQUARE.png")
    fig_mini3_coeff_keyterms(models_json, SLIDES_DIR / "MINI3_COEFF_KEYTERMS_WIDE.png")
    fig_mini4_approach_bars(SLIDES_DIR / "MINI4_APPROACH_STABILITY_WIDE.png")

    # Requested slide figure (tradeoff scatter)
    fig_tradeoff_scatter_gold(models_json, SLIDES_DIR / "TRADEOFF_FIT_VS_STABILITY_SCATTER.png")

    # Requested final-class figures (2/3)
    fig_fig2_rollout_vs_horizon(models_json, SLIDES_DIR / "FIG2_ROLLOUT_VS_HORIZON.png")
    fig_fig3_stabilization_effect(SLIDES_DIR, SLIDES_DIR / "FIG3_STABILIZATION_EFFECT.png")
    fig_fig3b_metric_disentanglement(SLIDES_DIR, SLIDES_DIR / "FIG3B_METRIC_DISENTANGLEMENT.png")

    # Figure 2 alternate: all models
    fig_fig2_rollout_vs_horizon_all(models_json, SLIDES_DIR / "FIG2_ROLLOUT_VS_HORIZON_ALL_MODELS.png")

    # Figure 1 alternatives (replace scatter when needed)
    fig_fig1_alt_paired_twoaxis_bars(models_json, SLIDES_DIR / "FIG1_ALT_PAIRED_BARS_R2_VS_ROLLOUT.png")
    fig_fig1_alt_slopegraph(models_json, SLIDES_DIR / "FIG1_ALT_SLOPEGRAPH_FIT_TO_ROLLOUT.png")
    fig_fig1_alt_rank_heatmap(models_json, SLIDES_DIR / "FIG1_ALT_RANK_VIEW.png")
    fig_fig1_alt_compact_heatmap(models_json, SLIDES_DIR / "FIG1_ALT_COMPACT_HEATMAP.png")
    fig_rank_slope_by_metric(models_json, SLIDES_DIR / "FIGX_RANK_SLOPE_FIT_VS_ROLLOUT.png")

    print("Wrote:")
    for p in [
        SLIDES_DIR / "PRES1_MODELS_HEATMAP_WIDE.png",
        SLIDES_DIR / "PRES2_ROLLOUT_BARS_TALL.png",
        SLIDES_DIR / "PRES3_FIT_STABILITY_SQUARE.png",
        SLIDES_DIR / "PRES4_M3_VS_M4_DUMBBELL_WIDE.png",
        SLIDES_DIR / "PICK_TERMS_PRESENCE_HEATMAP.png",
        SLIDES_DIR / "PICK_APPROACHES_BARS.png",
        SLIDES_DIR / "PAPER1_SCORE_STACKED_BARS_WIDE.png",
        SLIDES_DIR / "PAPER2_PARETO_SCATTER_SQUARE.png",
        SLIDES_DIR / "PAPER3_COEFF_HEATMAP_WIDE.png",
        SLIDES_DIR / "PAPER4_APPROACH_BARS_WIDE.png",
        SLIDES_DIR / "PAPER5_PATCH_STABILITY_HEATMAP_WIDE.png",
        SLIDES_DIR / "MINI1_RANK_BARS_WIDE.png",
        SLIDES_DIR / "MINI2_PARETO_SQUARE.png",
        SLIDES_DIR / "MINI3_COEFF_KEYTERMS_WIDE.png",
        SLIDES_DIR / "MINI4_APPROACH_STABILITY_WIDE.png",
        SLIDES_DIR / "TRADEOFF_FIT_VS_STABILITY_SCATTER.png",
        SLIDES_DIR / "FIG2_ROLLOUT_VS_HORIZON.png",
        SLIDES_DIR / "FIG3_STABILIZATION_EFFECT.png",
        SLIDES_DIR / "FIG3B_METRIC_DISENTANGLEMENT.png",
        SLIDES_DIR / "FIG2_ROLLOUT_VS_HORIZON_ALL_MODELS.png",
        SLIDES_DIR / "FIG1_ALT_PAIRED_BARS_R2_VS_ROLLOUT.png",
        SLIDES_DIR / "FIG1_ALT_SLOPEGRAPH_FIT_TO_ROLLOUT.png",
        SLIDES_DIR / "FIG1_ALT_RANK_VIEW.png",
        SLIDES_DIR / "FIG1_ALT_COMPACT_HEATMAP.png",
        SLIDES_DIR / "FIGX_RANK_SLOPE_FIT_VS_ROLLOUT.png",
    ]:
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
