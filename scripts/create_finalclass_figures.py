"""Create clean, chart-based figures for the 5-minute final-class presentation.

Outputs land in: outputs/latest/slides/

Figures generated (two styles):

Bar-heavy:
- FINAL1_PIPELINE_BARS.png
- FINAL2_MODEL3_VS_MODEL4.png
- FINAL3_COEFFS_COMPARISON.png
- FINAL4_PATCH_DIAGNOSTICS.png

Dashboard (2x2):
- FINAL1_PIPELINE_DASH.png
- FINAL2_MODEL3_VS_MODEL4_DASH.png
- FINAL3_COEFFS_DASH.png
- FINAL4_PATCH_DASH.png

The goal is "self-explanaining" visuals: big titles, short labels, no paragraphs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = PROJECT_ROOT / "outputs" / "latest" / "slides"
SLIDES_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "gray": "#5B5B5B",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
        }
    )


def _minimal_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


@dataclass(frozen=True)
class ModelSummary:
    name: str
    equation: str
    r2_test: float
    one_step_rmse: float
    k10_nrmse: float


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_summary(path: Path) -> ModelSummary:
    d = _read_json(path)
    k10 = float(d.get("rollout_time_test", {}).get("metrics", {}).get("10", {}).get("nrmse", float("nan")))
    return ModelSummary(
        name=str(d.get("name", path.stem)),
        equation=str(d.get("equation", "")),
        r2_test=float(d.get("r2", float("nan"))),
        one_step_rmse=float(d.get("one_step_rmse", float("nan"))),
        k10_nrmse=k10,
    )


def _read_rollout_nrmse(path: Path, k: int) -> float:
    d = _read_json(path)
    return float(d.get("rollout_time_test", {}).get("metrics", {}).get(str(int(k)), {}).get("nrmse", float("nan")))


def figure_pipeline_chart(out_path: Path) -> None:
    """Pipeline as a compact chart (connected markers) with minimal text."""
    steps = [
        "Frames",
        "Downsample",
        "Denoise",
        "Register",
        "Smooth",
        "Derivatives",
        "STRidge",
        "Validate",
    ]
    x = np.arange(1, len(steps) + 1)
    y = np.ones_like(x)

    fig, ax = plt.subplots(figsize=(14, 2.8), constrained_layout=True)
    ax.plot(x, y, color=COLORS["gray"], lw=2)
    ax.scatter(x, y, s=220, color=COLORS["blue"], zorder=3)

    for xi, label in zip(x, steps):
        ax.text(xi, 1.08, label, ha="center", va="bottom", fontsize=12)

    ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("Pipeline step")
    ax.set_ylim(0.75, 1.25)
    ax.set_title("Real-Image PDE Discovery Pipeline")
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_pipeline_bars(out_path: Path) -> None:
    """Bar-heavy pipeline: each step is a bar (same height), annotated."""
    steps = [
        "Frames",
        "Downsample",
        "Denoise",
        "Register",
        "Smooth",
        "Derivatives",
        "STRidge",
        "Validate",
    ]
    y = np.arange(len(steps))
    val = np.ones(len(steps))

    fig, ax = plt.subplots(figsize=(12.8, 4.2), constrained_layout=True)
    ax.barh(y, val, color=COLORS["blue"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(steps)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1.15)
    ax.set_title("Pipeline (Real Images)")
    for i in range(len(steps)):
        ax.text(1.02, i, f"Step {i+1}", va="center", ha="left", fontsize=11)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_pipeline_dashboard(out_path: Path) -> None:
    """2x2 dashboard: key dataset + split + processing knobs (charts only)."""
    frames = 51
    train_frac = 0.70
    space_frac = 0.70
    downsample = 2
    sigmas = [1.0, 1.5]
    rollout_steps = [5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    fig.suptitle("Pipeline Snapshot (Charts Only)")

    ax = axes[0, 0]
    ax.bar(["Frames"], [frames], color=COLORS["blue"])
    ax.set_title("Dataset size")
    ax.set_ylim(0, max(60, frames * 1.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0, frames, str(frames), ha="center", va="bottom", fontsize=12)

    ax = axes[0, 1]
    ax.bar(["Downsample"], [downsample], color=COLORS["orange"])
    ax.set_title("Spatial reduction")
    ax.set_ylim(0, 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0, downsample, f"×{downsample}", ha="center", va="bottom", fontsize=12)

    ax = axes[1, 0]
    ax.bar(["Time train", "Time test"], [train_frac, 1 - train_frac], color=[COLORS["green"], COLORS["gray"]])
    ax.set_title("Time split")
    ax.set_ylim(0, 1.0)
    _minimal_axes(ax)
    for i, v in enumerate([train_frac, 1 - train_frac]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    ax = axes[1, 1]
    ax.bar(["Space train", "Space test"], [space_frac, 1 - space_frac], color=[COLORS["green"], COLORS["gray"]])
    ax.set_title("Spatial split")
    ax.set_ylim(0, 1.0)
    _minimal_axes(ax)
    for i, v in enumerate([space_frac, 1 - space_frac]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    fig.text(0.01, 0.01, f"Denoise σ={sigmas[0]} → Smooth σ={sigmas[1]} · Rollout k={rollout_steps}", fontsize=11)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_model3_vs_model4(m3: ModelSummary, m4: ModelSummary, out_path: Path) -> None:
    # Two-metric comparison: pointwise fit vs rollout stability.
    labels = ["Model 3\n(stable)", "Model 4\n(best R²)"]
    r2 = [m3.r2_test, m4.r2_test]
    nrmse10 = [m3.k10_nrmse, m4.k10_nrmse]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), constrained_layout=True)

    # Panel A: R²
    ax = axes[0]
    ax.bar(labels, r2, color=[COLORS["blue"], COLORS["orange"]])
    ax.axhline(0.0, color="black", lw=1)
    ax.set_title("One-step Fit (time-test R²)", fontsize=14, fontweight="bold")
    ax.set_ylim(min(-2.5, min(r2) - 0.2), max(0.6, max(r2) + 0.1))
    for i, v in enumerate(r2):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=12)

    # Panel B: rollout
    ax = axes[1]
    ax.bar(labels, nrmse10, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_title("Dynamics (k=10 rollout nRMSE ↓)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(16.0, max(nrmse10) * 1.15))
    for i, v in enumerate(nrmse10):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12)

    fig.suptitle("Model 3 vs Model 4: Accuracy–Stability Tradeoff", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_model_dashboard(model3_json: Path, model4_json: Path, out_path: Path) -> None:
    """2x2 dashboard comparing key metrics (bar charts only)."""
    m3 = load_model_summary(model3_json)
    m4 = load_model_summary(model4_json)
    k5_3 = _read_rollout_nrmse(model3_json, 5)
    k5_4 = _read_rollout_nrmse(model4_json, 5)

    labels = ["Model 3", "Model 4"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    fig.suptitle("Model 3 vs Model 4 (Dashboard)")

    ax = axes[0, 0]
    vals = [m3.r2_test, m4.r2_test]
    ax.bar(labels, vals, color=[COLORS["blue"], COLORS["orange"]])
    ax.axhline(0.0, color="black", lw=1)
    ax.set_title("Time-test R² (↑)")
    _minimal_axes(ax)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")

    ax = axes[0, 1]
    vals = [m3.one_step_rmse, m4.one_step_rmse]
    ax.bar(labels, vals, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_title("One-step RMSE (↓)")
    _minimal_axes(ax)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    ax = axes[1, 0]
    vals = [k5_3, k5_4]
    ax.bar(labels, vals, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_title("Rollout k=5 nRMSE (↓)")
    _minimal_axes(ax)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    ax = axes[1, 1]
    vals = [m3.k10_nrmse, m4.k10_nrmse]
    ax.bar(labels, vals, color=[COLORS["blue"], COLORS["orange"]])
    ax.set_title("Rollout k=10 nRMSE (↓)")
    _minimal_axes(ax)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _terms_and_coeffs(path: Path) -> tuple[list[str], np.ndarray]:
    d = _read_json(path)
    terms = [t for t in d.get("terms", []) if t != "1"]
    coeffs = np.array(d.get("coeffs", []), dtype=float)
    if coeffs.size and len(d.get("terms", [])) == coeffs.size:
        # strip constant term
        coeffs = coeffs[1:]
    return terms, coeffs


def figure_coeffs_comparison(model3_json: Path, model4_json: Path, out_path: Path) -> None:
    """Grouped coefficient bars (symlog) to avoid equation-heavy slides."""
    t3, c3 = _terms_and_coeffs(model3_json)
    t4, c4 = _terms_and_coeffs(model4_json)

    all_terms = list(dict.fromkeys(t3 + t4))  # preserve order
    m3 = {t: float(v) for t, v in zip(t3, c3)}
    m4 = {t: float(v) for t, v in zip(t4, c4)}

    y = np.arange(len(all_terms))
    v3 = np.array([m3.get(t, 0.0) for t in all_terms])
    v4 = np.array([m4.get(t, 0.0) for t in all_terms])

    fig, ax = plt.subplots(figsize=(12.8, 4.8), constrained_layout=True)
    h = 0.38
    ax.barh(y - h / 2, v3, height=h, color=COLORS["blue"], label="Model 3")
    ax.barh(y + h / 2, v4, height=h, color=COLORS["orange"], label="Model 4")
    ax.axvline(0.0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(all_terms)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient value (symlog)")
    ax.set_title("Discovered Coefficients: Model 3 vs Model 4")
    ax.legend(loc="lower right", frameon=False)

    # Make tiny diffusion terms visible without hiding sign.
    ax.set_xscale("symlog", linthresh=1e-3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_coeffs_dashboard(model3_json: Path, model4_json: Path, out_path: Path) -> None:
    """2x2 dashboard: coeffs (symlog), abs magnitude, active terms, term overlap."""
    t3, c3 = _terms_and_coeffs(model3_json)
    t4, c4 = _terms_and_coeffs(model4_json)
    all_terms = list(dict.fromkeys(t3 + t4))
    m3 = {t: float(v) for t, v in zip(t3, c3)}
    m4 = {t: float(v) for t, v in zip(t4, c4)}
    v3 = np.array([m3.get(t, 0.0) for t in all_terms])
    v4 = np.array([m4.get(t, 0.0) for t in all_terms])
    y = np.arange(len(all_terms))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    fig.suptitle("Coefficient Comparison (Dashboard)")

    ax = axes[0, 0]
    h = 0.38
    ax.barh(y - h / 2, v3, height=h, color=COLORS["blue"], label="Model 3")
    ax.barh(y + h / 2, v4, height=h, color=COLORS["orange"], label="Model 4")
    ax.axvline(0.0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(all_terms)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_title("Coefficients (symlog)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.bar(["Model 3", "Model 4"], [int(np.count_nonzero(v3)), int(np.count_nonzero(v4))], color=[COLORS["blue"], COLORS["orange"]])
    ax.set_title("# active terms")
    _minimal_axes(ax)

    ax = axes[1, 0]
    abs3 = np.abs(v3) + 1e-12
    abs4 = np.abs(v4) + 1e-12
    ax.barh(y - h / 2, abs3, height=h, color=COLORS["blue"])
    ax.barh(y + h / 2, abs4, height=h, color=COLORS["orange"])
    ax.set_yticks(y)
    ax.set_yticklabels(all_terms)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_title("|coeff| magnitude (log)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    present3 = np.array([1 if t in t3 else 0 for t in all_terms])
    present4 = np.array([1 if t in t4 else 0 for t in all_terms])
    only3 = present3 * (1 - present4)
    only4 = present4 * (1 - present3)
    both = present3 * present4
    ax.barh(all_terms, both, color=COLORS["green"], label="both")
    ax.barh(all_terms, only3, left=both, color=COLORS["blue"], label="only M3")
    ax.barh(all_terms, only4, left=both + only3, color=COLORS["orange"], label="only M4")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_title("Term set overlap")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def read_patch_coeffs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def figure_patch_diagnostics(coeff_csv: Path, out_path: Path) -> None:
    rows = read_patch_coeffs(coeff_csv)
    # Filter out the constant term and zero rows
    cleaned = [r for r in rows if r.get("term") not in {"1", ""}]

    terms = [r["term"] for r in cleaned]
    nonzero = np.array([float(r["nonzero_freq"]) for r in cleaned], dtype=float)
    sign_stab = np.array([float(r["sign_stability"]) for r in cleaned], dtype=float)
    agg = np.array([float(r["agg_coeff"]) for r in cleaned], dtype=float)

    order = np.argsort(-nonzero)
    terms = [terms[i] for i in order]
    nonzero = nonzero[order]
    sign_stab = sign_stab[order]
    agg = agg[order]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.barh(terms, nonzero, color=COLORS["blue"])
    ax.set_xlim(0, 1.0)
    ax.set_title("Term presence\n(nonzero frequency)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0.6, color="black", lw=1, ls="--")
    ax.text(0.61, 0.5, "keep ≥ 0.6", transform=ax.get_xaxis_transform(), fontsize=10, va="center")

    ax = axes[1]
    ax.barh(terms, sign_stab, color=COLORS["orange"])
    ax.set_xlim(0, 1.0)
    ax.set_title("Sign consistency\n(sign stability)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    ax = axes[2]
    # show coefficient magnitude on symmetric scale
    m = float(np.nanmax(np.abs(agg))) if agg.size else 1.0
    m = max(m, 1e-6)
    ax.barh(terms, agg, color=COLORS["green"])
    ax.set_xlim(-1.1 * m, 1.1 * m)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_title("Aggregated coeff\n(median across patches)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    fig.suptitle("Patch-Based PDE: Stability Diagnostics", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_patch_dashboard(coeff_csv: Path, out_path: Path) -> None:
    """2x2 dashboard for patch stability diagnostics (charts only)."""
    rows = read_patch_coeffs(Path(coeff_csv))
    cleaned = [r for r in rows if r.get("term") not in {"1", ""}]

    terms = [r["term"] for r in cleaned]
    nonzero = np.array([float(r["nonzero_freq"]) for r in cleaned], dtype=float)
    sign_stab = np.array([float(r["sign_stability"]) for r in cleaned], dtype=float)
    agg = np.array([float(r["agg_coeff"]) for r in cleaned], dtype=float)
    q25 = np.array([float(r.get("q25", 0.0)) for r in cleaned], dtype=float)
    q75 = np.array([float(r.get("q75", 0.0)) for r in cleaned], dtype=float)
    iqr = q75 - q25

    order = np.argsort(-nonzero)
    terms = [terms[i] for i in order]
    nonzero = nonzero[order]
    sign_stab = sign_stab[order]
    agg = agg[order]
    q25 = q25[order]
    q75 = q75[order]
    iqr = iqr[order]
    y = np.arange(len(terms))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    fig.suptitle("Patch-Based PDE (Dashboard)")

    ax = axes[0, 0]
    ax.barh(y, nonzero, color=COLORS["blue"])
    ax.axvline(0.6, color="black", lw=1, ls="--")
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_title("Nonzero frequency")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.barh(y, sign_stab, color=COLORS["orange"])
    ax.set_xlim(0, 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_title("Sign stability")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    xerr = np.vstack((np.abs(agg - q25), np.abs(q75 - agg)))
    ax.errorbar(agg, y, xerr=xerr, fmt="o", color=COLORS["green"], ecolor=COLORS["gray"], elinewidth=2, capsize=3)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_title("Aggregated coeff (median ± IQR)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.barh(y, np.abs(iqr), color=COLORS["gray"])
    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_title("Uncertainty (IQR width)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # Use existing JSON summaries (already generated by analyze_results.py)
    model3_path = SLIDES_DIR / "best_model.json"  # Model 3 in this repo
    model4_path = SLIDES_DIR / "best_model_baseline.json"  # Model 4 best R²

    m3 = load_model_summary(model3_path)
    m4 = load_model_summary(model4_path)

    _apply_style()

    model3_path = SLIDES_DIR / "best_model.json"  # Model 3 in this repo
    model4_path = SLIDES_DIR / "best_model_baseline.json"  # Model 4 best R²

    # Bar-heavy versions
    figure_pipeline_bars(SLIDES_DIR / "FINAL1_PIPELINE_BARS.png")
    figure_model3_vs_model4(m3, m4, SLIDES_DIR / "FINAL2_MODEL3_VS_MODEL4.png")
    figure_coeffs_comparison(model3_path, model4_path, SLIDES_DIR / "FINAL3_COEFFS_COMPARISON.png")

    patch_csv = PROJECT_ROOT / "outputs" / "latest" / "patch_pde" / "PATCH_PDE_COEFFS.csv"
    figure_patch_diagnostics(patch_csv, SLIDES_DIR / "FINAL4_PATCH_DIAGNOSTICS.png")

    # Dashboard versions
    figure_pipeline_dashboard(SLIDES_DIR / "FINAL1_PIPELINE_DASH.png")
    figure_model_dashboard(model3_path, model4_path, SLIDES_DIR / "FINAL2_MODEL3_VS_MODEL4_DASH.png")
    figure_coeffs_dashboard(model3_path, model4_path, SLIDES_DIR / "FINAL3_COEFFS_DASH.png")
    figure_patch_dashboard(patch_csv, SLIDES_DIR / "FINAL4_PATCH_DASH.png")

    print("Wrote:")
    for p in [
        SLIDES_DIR / "FINAL1_PIPELINE_BARS.png",
        SLIDES_DIR / "FINAL2_MODEL3_VS_MODEL4.png",
        SLIDES_DIR / "FINAL3_COEFFS_COMPARISON.png",
        SLIDES_DIR / "FINAL4_PATCH_DIAGNOSTICS.png",
        SLIDES_DIR / "FINAL1_PIPELINE_DASH.png",
        SLIDES_DIR / "FINAL2_MODEL3_VS_MODEL4_DASH.png",
        SLIDES_DIR / "FINAL3_COEFFS_DASH.png",
        SLIDES_DIR / "FINAL4_PATCH_DASH.png",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
