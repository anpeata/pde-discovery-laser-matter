"""Create simplified Slide 3 variants (presentation-ready).

This script reads the best model from outputs/latest/slides/best_model.json
written by scripts/analyze_results.py. If that file is missing, it falls back
to a built-in default.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "latest" / "slides"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DEFAULT_BEST_MODEL = {
    "name": "Model 4: + Nonlinear (u^2)",
    "equation": "u_t = 0.35·u + 0.007·u_x + 0.005·u_y - 0.64·u^2",
    "r2": 0.431,
    "rmse": float("nan"),
    "nrmse": float("nan"),
    "corr": float("nan"),
    "one_step_rmse": float("nan"),
    "terms": ["u", "u_x", "u_y", "u^2"],
    "coeffs": [0.3511, 0.0070, 0.0052, -0.6413],
}


def load_best_model() -> dict:
    best_model_path = OUTPUT_FOLDER / "best_model.json"
    if not best_model_path.exists():
        return DEFAULT_BEST_MODEL
    try:
        with best_model_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure expected keys exist
        for k in ("name", "equation", "r2", "terms", "coeffs"):
            if k not in data:
                return DEFAULT_BEST_MODEL
        return data
    except Exception:
        return DEFAULT_BEST_MODEL


def fmt_coeff(c: float) -> str:
    if abs(c) >= 1:
        return f"{c:.3g}"
    if abs(c) >= 1e-2:
        return f"{c:.4f}"
    return f"{c:.2e}"


def equation_mathtext(terms: list[str], coeffs: list[float]) -> str:
    parts = []
    for name, c in zip(terms, coeffs):
        if abs(c) < 1e-8:
            continue
        sign = "+" if (c > 0 and parts) else ""
        parts.append(f"{sign}{fmt_coeff(c)}\\,{name}")
    rhs = " ".join(parts) if parts else "0"
    return rf"$u_t = {rhs}$"

best_model = load_best_model()

spatial = best_model.get("spatial_holdout") or {}
spatial_test = spatial.get("test") or {}
spatial_r2 = spatial_test.get("r2", float("nan"))
spatial_one_step = spatial_test.get("one_step_rmse", float("nan"))

spatial_tb = best_model.get("spatial_holdout_top_bottom") or {}
spatial_tb_test = spatial_tb.get("test") or {}
spatial_tb_r2 = spatial_tb_test.get("r2", float("nan"))
spatial_tb_one_step = spatial_tb_test.get("one_step_rmse", float("nan"))

rollout = best_model.get("rollout_time_test") or {}
rollout_metrics = rollout.get("metrics") or {}
rollout5 = rollout_metrics.get("5") or {}
rollout10 = rollout_metrics.get("10") or {}
rollout5_rmse = rollout5.get("rmse", float("nan"))
rollout10_rmse = rollout10.get("rmse", float("nan"))

# Clean term names for mathtext
term_map = {
    "1": "1",
    "u": "u",
    "u_x": "u_x",
    "u_y": "u_y",
    "lap(u)": "\\nabla^2 u",
    "u^2": "u^2",
    "u^3": "u^3",
    "u_xx": "u_{xx}",
    "u_yy": "u_{yy}",
    "u*u_x": "u\\,u_x",
    "u*u_y": "u\\,u_y",
    "u_x^2": "u_x^2",
    "u_y^2": "u_y^2",
}

terms = [term_map.get(t, t) for t in best_model.get("terms", DEFAULT_BEST_MODEL["terms"])]
coeffs = best_model.get("coeffs", DEFAULT_BEST_MODEL["coeffs"])

# ==================== SLIDE 3 (clean) ====================
plt.style.use("seaborn-v0_8-white")
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 0.85], hspace=0.25, wspace=0.18)

ax_eq = fig.add_subplot(gs[0, 0])
ax_eq.axis("off")

eq = equation_mathtext(terms, coeffs)
ax_eq.text(0.02, 0.70, "Discovered PDE (best model)", fontsize=24, fontweight="bold", transform=ax_eq.transAxes)
ax_eq.text(0.02, 0.35, eq, fontsize=30, transform=ax_eq.transAxes)

ax_meta = fig.add_subplot(gs[0, 1])
ax_meta.axis("off")
meta_lines = [
    f"Model: {best_model.get('name', 'best')} ",
    f"Active terms: {best_model.get('n_active', '—')}/{best_model.get('n_total', '—')}",
    "",
    f"R²: {best_model.get('r2', float('nan')):.3f}",
    f"R² (spatial): {spatial_r2:.3f}" if np.isfinite(spatial_r2) else "R² (spatial): —",
    f"R² (spatial TB): {spatial_tb_r2:.3f}" if np.isfinite(spatial_tb_r2) else "R² (spatial TB): —",
    f"RMSE: {best_model.get('rmse', float('nan')):.4f}",
    f"nRMSE: {best_model.get('nrmse', float('nan')):.2f}",
    f"corr: {best_model.get('corr', float('nan')):.2f}",
    f"one-step RMSE: {best_model.get('one_step_rmse', float('nan')):.4f}",
    f"one-step (spatial): {spatial_one_step:.4f}" if np.isfinite(spatial_one_step) else "one-step (spatial): —",
    f"one-step (spatial TB): {spatial_tb_one_step:.4f}" if np.isfinite(spatial_tb_one_step) else "one-step (spatial TB): —",
    f"rollout@5 RMSE: {rollout5_rmse:.4f}" if np.isfinite(rollout5_rmse) else "rollout@5 RMSE: —",
    f"rollout@10 RMSE: {rollout10_rmse:.4f}" if np.isfinite(rollout10_rmse) else "rollout@10 RMSE: —",
]
ax_meta.text(
    0.02,
    0.80,
    "\n".join(meta_lines),
    fontsize=16,
    transform=ax_meta.transAxes,
    va="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f2f2f2", edgecolor="#333333", linewidth=1.5),
)

ax_coeff = fig.add_subplot(gs[1, :])
colors = ["#2aa84a" if c > 0 else "#d64545" for c in coeffs]
bars = ax_coeff.bar(range(len(coeffs)), coeffs, color=colors, edgecolor="black", linewidth=1)
ax_coeff.axhline(0, color="black", linewidth=1)
ax_coeff.set_ylabel("Coefficient")
ax_coeff.set_xticks(range(len(terms)))
ax_coeff.set_xticklabels(terms, rotation=0, fontsize=14)
ax_coeff.set_title("Active coefficients", fontweight="bold")
ax_coeff.grid(True, alpha=0.25, axis="y")

for i, (bar, val) in enumerate(zip(bars, coeffs)):
    ax_coeff.text(i, val, fmt_coeff(val), ha="center", va="bottom" if val > 0 else "top", fontsize=12)

fig.suptitle("SLIDE 3: PDE Discovery Summary", fontsize=22, fontweight="bold", y=0.98)
plt.savefig(OUTPUT_FOLDER / "SLIDE3_SIMPLE_PDE_Results.png", dpi=300, bbox_inches="tight")
print("Created: SLIDE3_SIMPLE_PDE_Results.png")
plt.close(fig)

# ==================== SLIDE 3 (minimal) ====================
fig2, ax2 = plt.subplots(figsize=(16, 9))
ax2.axis("off")
ax2.text(0.5, 0.70, "Discovered PDE (best model)", ha="center", fontsize=28, fontweight="bold", transform=ax2.transAxes)
ax2.text(0.5, 0.48, eq, ha="center", fontsize=34, transform=ax2.transAxes)
ax2.text(
    0.5,
    0.18,
    f"time R²={best_model.get('r2', float('nan')):.3f}  |  space R² LR/TB={spatial_r2:.3f}/{spatial_tb_r2:.3f}  |  rollout@5={rollout5_rmse:.4f}",
    ha="center",
    fontsize=16,
    color="#333333",
    transform=ax2.transAxes,
)

plt.savefig(OUTPUT_FOLDER / "SLIDE3_MINIMAL_PDE_Results.png", dpi=300, bbox_inches="tight")
print("Created: SLIDE3_MINIMAL_PDE_Results.png")
plt.close(fig2)

print("\n" + "=" * 80)
print("CREATED 2 SIMPLIFIED SLIDES")
print("=" * 80)
