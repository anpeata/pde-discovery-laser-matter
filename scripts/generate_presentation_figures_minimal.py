"""Generate minimal, presentation-ready figures.

Goal: produce clean, low-text visuals similar in style to the minimal slides.

Outputs
- outputs/latest/presentation_minimal/FIG_DATA_FRAMES.png
- outputs/latest/presentation_minimal/FIG_BEST_MODEL_COEFFS.png

Notes
- Reads best model from outputs/latest/slides/best_model.json (written by analyze_results.py).
- Does not depend on any of the heavy pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "presentation_minimal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_best_model() -> dict | None:
    p = PROJECT_ROOT / "outputs" / "latest" / "slides" / "best_model.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.5])
    out = (img - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)


def _fig_data_frames() -> Path:
    tif_files = sorted(DATA_DIR.glob("*.tif"))
    if len(tif_files) < 3:
        raise RuntimeError(f"Need at least 3 tif files in: {DATA_DIR}")

    idxs = [0, len(tif_files) // 2, len(tif_files) - 1]
    frames = [_normalize(_read_gray(tif_files[i])) for i in idxs]

    # Downsample for fast rendering
    h, w = frames[0].shape
    target_w = 900
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    frames = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames]

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, f, i in zip(axes, frames, idxs):
        ax.imshow(f, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Frame {i}", fontsize=16, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Observed field (u) over time", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUT_DIR / "FIG_DATA_FRAMES.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fig_best_model_coeffs() -> Path:
    best = _load_best_model()
    if best is None:
        raise RuntimeError(
            "best_model.json not found. Run scripts/analyze_results.py first to generate it."
        )

    terms = best.get("terms", [])
    coeffs = best.get("coeffs", [])
    if not terms or not coeffs:
        raise RuntimeError("best_model.json missing 'terms'/'coeffs'.")

    # Remove constant for readability
    pairs = [(t, float(c)) for t, c in zip(terms, coeffs) if t != "1"]
    terms2 = [p[0] for p in pairs]
    coeffs2 = np.array([p[1] for p in pairs], dtype=np.float64)

    # Order by magnitude
    order = np.argsort(np.abs(coeffs2))[::-1]
    terms2 = [terms2[i] for i in order]
    coeffs2 = coeffs2[order]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    colors = ["#2aa84a" if c > 0 else "#d64545" for c in coeffs2]
    y = np.arange(len(terms2))
    ax.barh(y, coeffs2, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(terms2, fontsize=13)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient")

    r2 = float(best.get("r2", float("nan")))
    one_step = float(best.get("one_step_rmse", float("nan")))
    spatial_lr = best.get("spatial_holdout") or {}
    spatial_lr_test = spatial_lr.get("test") or {}
    r2_space_lr = float(spatial_lr_test.get("r2", float("nan")))

    spatial_tb = best.get("spatial_holdout_top_bottom") or {}
    spatial_tb_test = spatial_tb.get("test") or {}
    r2_space_tb = float(spatial_tb_test.get("r2", float("nan")))

    rollout = best.get("rollout_time_test") or {}
    rollout_metrics = rollout.get("metrics") or {}
    rollout_5 = rollout_metrics.get("5") or {}
    rollout_5_rmse = rollout_5.get("rmse", float("nan"))

    title = (
        f"Best discovered PDE coefficients  (time R²={r2:.3f}, "
        f"space R² LR/TB={r2_space_lr:.3f}/{r2_space_tb:.3f})"
    )
    if np.isfinite(one_step):
        title += f"  |  one-step={one_step:.3f}"
    if np.isfinite(float(rollout_5_rmse)):
        title += f"  |  rollout@5 RMSE={float(rollout_5_rmse):.3f}"
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "FIG_BEST_MODEL_COEFFS.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    print("=" * 60)
    print("GENERATING MINIMAL PRESENTATION FIGURES")
    print("=" * 60)

    p1 = _fig_data_frames()
    p2 = _fig_best_model_coeffs()

    print("\nSaved:")
    print(f"- {p1.relative_to(PROJECT_ROOT)}")
    print(f"- {p2.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
