"""Create a single, presentation-ready method slide.

Outputs
- outputs/latest/slides/SLIDE_METHOD_PIPELINE.png

This slide summarizes:
- pipeline diagram (data -> preprocessing -> derivatives -> library -> STRidge)
- discovered PDE (from best_model.json)
- generalization metrics (time holdout + spatial holdout + rollout)

Reads
- outputs/latest/slides/best_model.json (written by scripts/analyze_results.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "slides"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_best_model() -> dict:
	p = OUT_DIR / "best_model.json"
	if not p.exists():
		raise FileNotFoundError(
			"best_model.json not found. Run scripts/analyze_results.py first to generate it."
		)
	return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: object, ndigits: int = 3) -> str:
	try:
		v = float(x)  # type: ignore[arg-type]
	except Exception:
		return "—"
	if not np.isfinite(v):
		return "—"
	return f"{v:.{ndigits}f}"


def _equation_mathtext(terms: list[str], coeffs: list[float]) -> str:
	term_map = {
		"1": "1",
		"u": "u",
		"u_x": "u_x",
		"u_y": "u_y",
		"lap(u)": "\\nabla^2 u",
		"u^2": "u^2",
		"u^3": "u^3",
		"u*u_x": "u\\,u_x",
		"u*u_y": "u\\,u_y",
		"u_xx": "u_{xx}",
		"u_yy": "u_{yy}",
		"u_x^2": "u_x^2",
		"u_y^2": "u_y^2",
	}

	parts: list[str] = []
	for t, c in zip(terms, coeffs):
		try:
			c = float(c)
		except Exception:
			continue
		if abs(c) < 1e-8:
			continue
		t2 = term_map.get(t, t)
		sign = "+" if (c > 0 and parts) else ""
		# Prefer fixed-point decimals for small coefficients (presentation-friendly).
		# Keep scientific only for extremely tiny magnitudes.
		if abs(c) >= 1:
			cstr = f"{c:.6f}"
		elif abs(c) >= 1e-6:
			cstr = f"{c:.6f}"
		else:
			cstr = f"{c:.2e}"
		parts.append(f"{sign}{cstr}\\,{t2}")

	rhs = " ".join(parts) if parts else "0"
	return rf"$u_t = {rhs}$"


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str) -> None:
	box = FancyBboxPatch(
		(x, y),
		w,
		h,
		boxstyle="round,pad=0.015,rounding_size=0.02",
		linewidth=1.6,
		edgecolor="#222222",
		facecolor="#f2f2f2",
	)
	ax.add_patch(box)
	ax.text(
		x + w / 2,
		y + h / 2,
		text,
		ha="center",
		va="center",
		fontsize=14,
		color="#111111",
		wrap=True,
	)


def main() -> int:
	best = _load_best_model()

	plt.style.use("seaborn-v0_8-white")
	fig = plt.figure(figsize=(16, 9))
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_axis_off()

	ax.text(
		0.5,
		0.95,
		"Method: PDE-FIND / SINDy (STRidge) on Real Images",
		ha="center",
		va="top",
		fontsize=24,
		fontweight="bold",
	)

	# Pipeline diagram
	ax.text(0.07, 0.86, "Pipeline", ha="left", va="center", fontsize=18, fontweight="bold")

	steps = [
		"Real TIFF frames\n(u(x,y,t))",
		"Denoise\n(Gaussian)",
		"Register\n(Farnebäck flow)",
		"Smooth +\ncrop/downsample",
		"Derivatives\n(finite diff.)",
		"Library Θ(u)\n+ STRidge",
		"Sparse PDE\n(model selection)",
	]

	x0, y1, y2 = 0.07, 0.73, 0.58
	w, h = 0.19, 0.10
	gapx = 0.02

	positions: list[tuple[float, float]] = []
	for i in range(4):
		positions.append((x0 + i * (w + gapx), y1))
	for i in range(3):
		positions.append((x0 + i * (w + gapx), y2))

	for (x, y), label in zip(positions, steps):
		_draw_box(ax, x, y, w, h, label)

	def arrow(xa: float, ya: float, xb: float, yb: float) -> None:
		ax.annotate(
			"",
			xy=(xb, yb),
			xytext=(xa, ya),
			arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
		)

	for i in range(3):
		xa, ya = positions[i]
		xb, yb = positions[i + 1]
		arrow(xa + w, ya + h / 2, xb, yb + h / 2)

	xa, ya = positions[3]
	xb, yb = positions[4]
	arrow(xa + w / 2, ya, xb + w / 2, yb + h)

	for i in range(4, 6):
		xa, ya = positions[i]
		xb, yb = positions[i + 1]
		arrow(xa + w, ya + h / 2, xb, yb + h / 2)

	# Equation
	ax.text(0.07, 0.48, "Discovered PDE", ha="left", va="center", fontsize=18, fontweight="bold")
	eq = _equation_mathtext(best.get("terms", []), best.get("coeffs", []))
	ax.text(0.07, 0.40, eq, ha="left", va="center", fontsize=26)

	# Generalization metrics
	ax.text(0.07, 0.30, "Generalization", ha="left", va="center", fontsize=18, fontweight="bold")
	r2_time = best.get("r2")
	one_step = best.get("one_step_rmse")

	lr_test = (best.get("spatial_holdout") or {}).get("test") or {}
	tb_test = (best.get("spatial_holdout_top_bottom") or {}).get("test") or {}

	rollout = (best.get("rollout_time_test") or {}).get("metrics") or {}
	r5 = rollout.get("5") or {}
	r10 = rollout.get("10") or {}

	lines = [
		f"Time holdout (future frames):  R^2={_fmt(r2_time, 6)}   |  one-step RMSE={_fmt(one_step, 6)}",
		f"Spatial holdout (Left/Right):   R^2={_fmt(lr_test.get('r2'), 6)}",
		f"Spatial holdout (Top/Bottom):   R^2={_fmt(tb_test.get('r2'), 6)}",
		f"Multi-step rollout (time test):  k=5 RMSE={_fmt(r5.get('rmse'), 6)}   |  k=10 RMSE={_fmt(r10.get('rmse'), 6)}",
	]

	ax.text(
		0.07,
		0.23,
		"\n".join(lines),
		ha="left",
		va="top",
		fontsize=14.5,
		bbox=dict(
			boxstyle="round,pad=0.5",
			facecolor="#f7f7f7",
			edgecolor="#333333",
			linewidth=1.4,
		),
	)

	# Note (intentionally omitted for a cleaner slide)
	# ax.text(
	#     0.07,
	#     0.06,
	#     "Note: metrics compare predicted u_t to data-derived u_t (after denoise+registration+smoothing).",
	#     ha="left",
	#     va="center",
	#     fontsize=11.5,
	#     color="#333333",
	# )

	out_path = OUT_DIR / "SLIDE_METHOD_PIPELINE.png"
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Created: {out_path.relative_to(PROJECT_ROOT)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
