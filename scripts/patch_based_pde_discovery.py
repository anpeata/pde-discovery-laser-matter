"""Patch-based PDE discovery (real images) — robust derivatives + stability selection.

Why this version exists
- Finite differences on noisy real images can destroy PDE discovery.
- Patch-based fitting helps spatial heterogeneity, but MUST be validated out-of-sample.

Key upgrades vs the previous version
1) Derivatives via local 3D polynomial regression (Savitzky–Golay style) on a
    spatiotemporal neighborhood around each sample point.
2) Time-split evaluation (train early frames, test later frames) to reduce the
    “high R² but wrong PDE” failure mode.
3) Stability selection across patches (median + nonzero frequency + sign stability).
4) Metrics beyond R²: RMSE/MAE/nRMSE/corr + one-step prediction RMSE.

Outputs
- outputs/latest/patch_pde/PATCH_PDE_COEFFS.csv
- outputs/latest/patch_pde/PATCH_PDE_SUMMARY.png
- outputs/latest/patch_pde/PATCH_PDE_REPORT.txt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "patch_pde"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    y_std = float(np.std(y_true))
    nrmse = float(rmse / (y_std + 1e-12))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "corr": corr,
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "resid_med_abs": float(np.median(np.abs(resid))),
    }


def one_step_prediction_rmse(u_field: np.ndarray, ut_pred: np.ndarray, dt: float = 1.0) -> float:
    t_max = min(u_field.shape[0] - 1, ut_pred.shape[0])
    if t_max <= 0:
        return float("nan")
    u0 = u_field[:t_max]
    u1 = u_field[1 : t_max + 1]
    u1_pred = u0 + dt * ut_pred[:t_max]
    return float(np.sqrt(np.mean((u1 - u1_pred) ** 2)))


def stridge(X: np.ndarray, y: np.ndarray, alpha: float = 0.01, threshold: float = 1e-5, max_iter: int = 25) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    coeffs = model.coef_.copy()

    for _ in range(max_iter):
        small = np.abs(coeffs) < threshold
        coeffs[small] = 0
        big = ~small
        if big.sum() == 0:
            break
        model.fit(Xs[:, big], y)
        coeffs_big = model.coef_.copy()
        coeffs = np.zeros_like(coeffs)
        coeffs[big] = coeffs_big

    # Undo standardization
    return coeffs / (scaler.scale_ + 1e-12)


def load_images(folder: Path, max_images: int = 51) -> np.ndarray:
    files = sorted(folder.glob("*.tif"))[:max_images]
    if not files:
        raise FileNotFoundError(f"No .tif files found in: {folder}")

    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img.astype(np.float32))

    if not frames:
        raise RuntimeError("Failed to load any images.")

    U = np.stack(frames, axis=0)
    return U


def farneback_register(U: np.ndarray) -> np.ndarray:
    registered = [U[0].copy()]
    for i in range(1, U.shape[0]):
        ref = registered[-1]
        mov = U[i]
        ref_u8 = np.clip(ref * 255, 0, 255).astype(np.uint8)
        mov_u8 = np.clip(mov * 255, 0, 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            ref_u8,
            mov_u8,
            None,
            0.5,
            5,
            25,
            5,
            7,
            1.5,
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flow = cv2.GaussianBlur(flow, (11, 11), 2.0)
        h, w = mov.shape
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w) - flow[:, :, 0]
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis] - flow[:, :, 1]
        warped = cv2.remap(mov, flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        registered.append(warped)

    return np.stack(registered, axis=0)


@dataclass(frozen=True)
class Library:
    names: list[str]

    def feature_vector(self, u: float, ux: float, uy: float, uxx: float, uyy: float) -> np.ndarray:
        lap = uxx + uyy
        if self.names == ["1", "u", "u_x", "u_y", "lap(u)", "u^2"]:
            return np.array([1.0, u, ux, uy, lap, u**2], dtype=np.float64)
        return np.array([
            1.0,
            u,
            ux,
            uy,
            lap,
            u**2,
            u * ux,
            u * uy,
        ], dtype=np.float64)


def _poly3d_exponents(deg: int) -> list[tuple[int, int, int]]:
    exps: list[tuple[int, int, int]] = []
    for a in range(deg + 1):
        for b in range(deg + 1 - a):
            for c in range(deg + 1 - a - b):
                exps.append((a, b, c))
    return exps


def _poly3d_design(t: np.ndarray, x: np.ndarray, y: np.ndarray, exps: list[tuple[int, int, int]]) -> np.ndarray:
    # Each row is one sample point.
    cols = []
    for a, b, c in exps:
        cols.append((t**a) * (x**b) * (y**c))
    return np.column_stack(cols)


def local_poly_derivatives(
    U: np.ndarray,
    t0: int,
    y0: int,
    x0: int,
    rt: int,
    rs: int,
    deg: int,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[float, float, float, float, float, float]:
    """Estimate (u, u_t, u_x, u_y, u_xx, u_yy) at a point via 3D polynomial regression.

    U is indexed as U[t, y, x]. We fit p(t,x,y) on a neighborhood around (t0,y0,x0)
    in local coordinates centered at 0.
    """
    t_idx = np.arange(t0 - rt, t0 + rt + 1)
    y_idx = np.arange(y0 - rs, y0 + rs + 1)
    x_idx = np.arange(x0 - rs, x0 + rs + 1)

    # Local coordinate grids
    tt = (t_idx - t0) * dt
    yy = (y_idx - y0) * dy
    xx = (x_idx - x0) * dx
    Tt, Yy, Xx = np.meshgrid(tt, yy, xx, indexing="ij")

    vals = U[np.ix_(t_idx, y_idx, x_idx)].astype(np.float64)

    t_flat = Tt.ravel()
    x_flat = Xx.ravel()
    y_flat = Yy.ravel()
    v_flat = vals.ravel()

    exps = _poly3d_exponents(deg)
    A = _poly3d_design(t_flat, x_flat, y_flat, exps)

    # Least squares fit
    coef, *_ = np.linalg.lstsq(A, v_flat, rcond=None)

    def get_coef(a: int, b: int, c: int) -> float:
        try:
            idx = exps.index((a, b, c))
        except ValueError:
            return 0.0
        return float(coef[idx])

    u0 = get_coef(0, 0, 0)
    ut0 = get_coef(1, 0, 0)
    ux0 = get_coef(0, 1, 0)
    uy0 = get_coef(0, 0, 1)
    uxx0 = 2.0 * get_coef(0, 2, 0)
    uyy0 = 2.0 * get_coef(0, 0, 2)
    return u0, ut0, ux0, uy0, uxx0, uyy0


def safe_sample_points(
    rng: np.random.Generator,
    t_indices: np.ndarray,
    h: int,
    w: int,
    rs: int,
    n: int,
) -> list[tuple[int, int, int]]:
    ys = rng.integers(rs, h - rs, size=n)
    xs = rng.integers(rs, w - rs, size=n)
    ts = rng.choice(t_indices, size=n, replace=True)
    return list(zip(ts.tolist(), ys.tolist(), xs.tolist()))


def build_dataset(
    U: np.ndarray,
    points: list[tuple[int, int, int]],
    rt: int,
    rs: int,
    deg: int,
    dt: float,
    dx: float,
    dy: float,
    lib: Library,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    y = []
    for t0, y0, x0 in points:
        u0, ut0, ux0, uy0, uxx0, uyy0 = local_poly_derivatives(U, t0, y0, x0, rt, rs, deg, dt, dx, dy)
        rows.append(lib.feature_vector(u0, ux0, uy0, uxx0, uyy0))
        y.append(ut0)
    return np.vstack(rows), np.array(y)


def patch_grid(h: int, w: int, patch: int, overlap: int) -> list[tuple[int, int]]:
    stride = max(1, patch - overlap)
    coords = []
    for y0 in range(0, h - patch + 1, stride):
        for x0 in range(0, w - patch + 1, stride):
            coords.append((y0, x0))
    return coords


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch-based PDE discovery (robust derivatives + stability)")
    parser.add_argument("--max-images", type=int, default=51)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-registration", action="store_true")
    parser.add_argument("--rt", type=int, default=2, help="Half time window for local poly derivatives")
    parser.add_argument("--rs", type=int, default=3, help="Half spatial window for local poly derivatives")
    parser.add_argument("--deg", type=int, default=3, help="Polynomial degree for local derivatives")
    parser.add_argument("--patch", type=int, default=21, help="Patch size in downsampled grid (odd recommended)")
    parser.add_argument("--overlap", type=int, default=10, help="Patch overlap in downsampled grid")
    parser.add_argument("--samples-per-patch", type=int, default=120)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--stability-freq", type=float, default=0.6, help="Keep term if selected in >= this fraction of patches")
    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=["full", "model4"],
        help="Term set to use. 'model4' matches analyze_results.py Model 4: 1,u,u_x,u_y,lap(u),u^2",
    )
    args = parser.parse_args()

    if args.rt < 1:
        raise ValueError("--rt must be >= 1")
    if args.rs < 2:
        raise ValueError("--rs must be >= 2")
    if not (0.4 <= args.train_frac <= 0.9):
        raise ValueError("--train-frac should be in [0.4, 0.9]")

    print("=" * 80)
    print("PATCH-BASED PDE DISCOVERY (ROBUST) — REAL IMAGES")
    print("=" * 80)

    U_raw = load_images(DATA_DIR, max_images=args.max_images)
    T0, H0, W0 = U_raw.shape
    print(f"Loaded {T0} frames: {H0}x{W0}")

    # Downsample like analyze_results (keep it consistent)
    U_ds = np.array([cv2.resize(img, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA) for img in U_raw])

    # Denoise + normalize
    U_dn = np.array([gaussian_filter(img, sigma=1.0) for img in U_ds])
    U_norm = (U_dn - U_dn.min()) / (U_dn.max() - U_dn.min() + 1e-12)

    if not args.no_registration:
        print("Running light Farnebäck registration...")
        U_norm = farneback_register(U_norm)

    # Smooth
    U_sm = np.array([gaussian_filter(img, sigma=1.2) for img in U_norm])

    # Crop + subsample
    skip = 25
    subsample = 12
    U = U_sm[:, skip:-skip:subsample, skip:-skip:subsample]

    # Effective grid and physical steps (kept consistent with earlier scripts)
    dx, dy, dt = 0.1, 0.1, 1.0
    t_len, h, w = U.shape
    print(f"Working grid: T={t_len}, H={h}, W={w}")

    rt = args.rt
    rs = args.rs
    if (2 * rt + 1) * (2 * rs + 1) * (2 * rs + 1) < 30:
        print("Warning: very small neighborhood for derivatives; consider increasing --rt/--rs")

    # Time split for validation
    t_min = rt
    t_max = t_len - rt - 1
    if t_max <= t_min + 2:
        raise RuntimeError("Not enough frames after accounting for derivative window.")

    t_valid = np.arange(t_min, t_max + 1)
    split = int(math.floor(args.train_frac * len(t_valid)))
    t_train = t_valid[:split]
    t_test = t_valid[split:]
    if len(t_test) < 3:
        raise RuntimeError("Test set too small; reduce --train-frac or use more frames.")

    if args.model == "model4":
        lib = Library(names=["1", "u", "u_x", "u_y", "lap(u)", "u^2"])
    else:
        lib = Library(names=["1", "u", "u_x", "u_y", "lap(u)", "u^2", "u*u_x", "u*u_y"])

    # Patch grid
    patch = int(args.patch)
    overlap = int(args.overlap)
    if patch < 9:
        raise ValueError("--patch too small; use >= 9")
    if overlap >= patch:
        raise ValueError("--overlap must be < --patch")
    coords = patch_grid(h, w, patch=patch, overlap=overlap)
    if not coords:
        raise RuntimeError("Patch grid is empty; reduce patch size or check cropping.")

    rng = np.random.default_rng(args.seed)
    coeffs_list: list[np.ndarray] = []
    patch_train_metrics: list[dict] = []
    patch_test_metrics: list[dict] = []

    # Fit per patch
    for (y0, x0) in coords:
        y1 = y0 + patch
        x1 = x0 + patch

        # Sample points inside the patch, avoiding boundaries needed for derivative window
        ys_low = max(rs, y0 + rs)
        ys_high = min(h - rs, y1 - rs)
        xs_low = max(rs, x0 + rs)
        xs_high = min(w - rs, x1 - rs)
        if ys_high <= ys_low or xs_high <= xs_low:
            continue

        # Train samples
        n_s = int(args.samples_per_patch)
        ys = rng.integers(ys_low, ys_high, size=n_s)
        xs = rng.integers(xs_low, xs_high, size=n_s)
        ts = rng.choice(t_train, size=n_s, replace=True)
        train_pts = list(zip(ts.tolist(), ys.tolist(), xs.tolist()))

        # Test samples
        ys2 = rng.integers(ys_low, ys_high, size=max(30, n_s // 3))
        xs2 = rng.integers(xs_low, xs_high, size=max(30, n_s // 3))
        ts2 = rng.choice(t_test, size=max(30, n_s // 3), replace=True)
        test_pts = list(zip(ts2.tolist(), ys2.tolist(), xs2.tolist()))

        X_train, y_train = build_dataset(U, train_pts, rt=rt, rs=rs, deg=args.deg, dt=dt, dx=dx, dy=dy, lib=lib)
        X_test, y_test = build_dataset(U, test_pts, rt=rt, rs=rs, deg=args.deg, dt=dt, dx=dx, dy=dy, lib=lib)

        c = stridge(X_train, y_train, alpha=args.alpha, threshold=args.threshold)
        coeffs_list.append(c)

        y_pred_tr = X_train @ c
        y_pred_te = X_test @ c
        patch_train_metrics.append(regression_metrics(y_train, y_pred_tr))
        patch_test_metrics.append(regression_metrics(y_test, y_pred_te))

    if not coeffs_list:
        raise RuntimeError("No patches were fitted; check patch/overlap and rs/rt.")

    C = np.stack(coeffs_list, axis=0)
    nonzero = np.abs(C) > args.threshold
    freq = nonzero.mean(axis=0)
    median = np.median(C, axis=0)
    q25 = np.percentile(C, 25, axis=0)
    q75 = np.percentile(C, 75, axis=0)
    sign_stability = np.mean(np.sign(C) == np.sign(median + 1e-12), axis=0)

    keep = freq >= float(args.stability_freq)
    agg = np.where(keep, median, 0.0)

    # Global evaluation on held-out time points (sampled across space)
    test_points_global = safe_sample_points(rng, t_indices=t_test, h=h, w=w, rs=rs, n=800)
    Xg, yg = build_dataset(U, test_points_global, rt=rt, rs=rs, deg=args.deg, dt=dt, dx=dx, dy=dy, lib=lib)
    y_pred_g = Xg @ agg
    m_test = regression_metrics(yg, y_pred_g)

    # One-step prediction check on coarse grid using the aggregated model:
    # We estimate u_t_pred on a subset of grid points for each time.
    # (Keep it light: sample points rather than full field.)
    step_pts = safe_sample_points(rng, t_indices=t_valid[:-1], h=h, w=w, rs=rs, n=1200)
    Xs, ys_ut = build_dataset(U, step_pts, rt=rt, rs=rs, deg=args.deg, dt=dt, dx=dx, dy=dy, lib=lib)
    ut_pred = Xs @ agg
    # One-step RMSE in derivative-space proxy: compare u(t+1) - u(t) vs dt * ut_pred
    # Use sampled points only.
    errs = []
    for (t0, y0, x0), utp in zip(step_pts, ut_pred.tolist()):
        if t0 + 1 >= t_len:
            continue
        du = float(U[t0 + 1, y0, x0] - U[t0, y0, x0])
        errs.append((du - dt * utp) ** 2)
    one_step_rmse = float(np.sqrt(np.mean(errs))) if errs else float("nan")

    # Save CSV
    csv_path = OUT_DIR / "PATCH_PDE_COEFFS.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("term,median,q25,q75,nonzero_freq,sign_stability,agg_coeff\n")
        for name, med, a, b, fr, ss, ac in zip(lib.names, median, q25, q75, freq, sign_stability, agg):
            f.write(f"{name},{med:.8g},{a:.8g},{b:.8g},{fr:.3f},{ss:.3f},{ac:.8g}\n")

    # Build report
    report_path = OUT_DIR / "PATCH_PDE_REPORT.txt"
    parts = []
    for coef, name in zip(agg, lib.names):
        if abs(coef) > args.threshold:
            sign = "+" if coef > 0 and parts else ""
            parts.append(f"{sign}{coef:.4g}·{name}")
    eq = "u_t = " + (" ".join(parts) if parts else "0")

    def summarize_metrics(ms: list[dict]) -> dict:
        if not ms:
            return {"r2": float("nan"), "rmse": float("nan"), "nrmse": float("nan")}
        return {
            "r2_mean": float(np.mean([m["r2"] for m in ms])),
            "r2_median": float(np.median([m["r2"] for m in ms])),
            "rmse_mean": float(np.mean([m["rmse"] for m in ms])),
            "rmse_median": float(np.median([m["rmse"] for m in ms])),
            "nrmse_mean": float(np.mean([m["nrmse"] for m in ms])),
        }

    tr_sum = summarize_metrics(patch_train_metrics)
    te_sum = summarize_metrics(patch_test_metrics)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("PATCH-BASED PDE DISCOVERY REPORT (ROBUST)\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"Frames used: {args.max_images}\n")
        f.write(f"Grid after preprocessing: T={t_len}, H={h}, W={w}\n")
        f.write(f"Registration: {'OFF' if args.no_registration else 'ON'}\n")
        f.write(f"Local derivative neighborhood: rt={rt}, rs={rs}, degree={args.deg}\n")
        f.write(f"Patch grid: patch={patch}, overlap={overlap}, patches_fit={len(coeffs_list)}\n")
        f.write(f"Train/Test split (time): train_frac={args.train_frac:.2f}, train_T={len(t_train)}, test_T={len(t_test)}\n")
        f.write(f"Model term set: {args.model}\n")
        f.write(f"STRidge: alpha={args.alpha}, threshold={args.threshold}\n")
        f.write(f"Stability keep rule: nonzero_freq >= {args.stability_freq}\n\n")

        f.write("Per-patch metrics (train) summary:\n")
        f.write(f"  R² mean={tr_sum['r2_mean']:.4f}, median={tr_sum['r2_median']:.4f}\n")
        f.write(f"  RMSE mean={tr_sum['rmse_mean']:.6f}, median={tr_sum['rmse_median']:.6f}\n")
        f.write(f"  nRMSE mean={tr_sum['nrmse_mean']:.3f}\n\n")

        f.write("Per-patch metrics (test) summary:\n")
        f.write(f"  R² mean={te_sum['r2_mean']:.4f}, median={te_sum['r2_median']:.4f}\n")
        f.write(f"  RMSE mean={te_sum['rmse_mean']:.6f}, median={te_sum['rmse_median']:.6f}\n")
        f.write(f"  nRMSE mean={te_sum['nrmse_mean']:.3f}\n\n")

        f.write("Aggregated model (test samples) metrics:\n")
        f.write(f"  R²={m_test['r2']:.6f}\n")
        f.write(f"  RMSE={m_test['rmse']:.6f}\n")
        f.write(f"  MAE={m_test['mae']:.6f}\n")
        f.write(f"  nRMSE={m_test['nrmse']:.3f}\n")
        f.write(f"  corr={m_test['corr']:.3f}\n")
        f.write(f"  resid median abs={m_test['resid_med_abs']:.6f}\n")
        f.write(f"  one-step RMSE (sampled)={one_step_rmse:.6f}\n\n")

        f.write("Aggregated PDE:\n")
        f.write(f"  {eq}\n")

    # Plot (presentation-friendly, minimal text)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 6.5))
    x = np.arange(len(lib.names))

    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(x, median, color="#2a6fdb", alpha=0.85, edgecolor="black", linewidth=1)
    ax1.errorbar(x, median, yerr=[median - q25, q75 - median], fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lib.names, rotation=35, ha="right")
    ax1.set_title("Patch coefficients (median ± IQR)")
    ax1.set_ylabel("Coefficient")

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(x, freq, color="#2aa84a", alpha=0.85, edgecolor="black", linewidth=1)
    ax2.plot(x, sign_stability, color="#1f3d7a", marker="o", linewidth=1.5, label="sign stability")
    ax2.axhline(float(args.stability_freq), color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lib.names, rotation=35, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_title("Stability across patches")
    ax2.set_ylabel("frequency")
    ax2.legend(frameon=False, loc="lower right")

    fig.suptitle(
        f"Patch-based PDE (test)  R²={m_test['r2']:.2f}  nRMSE={m_test['nrmse']:.2f}  one-step RMSE={one_step_rmse:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = OUT_DIR / "PATCH_PDE_SUMMARY.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight")

    print("\nSaved outputs:")
    print(f"- {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"- {report_path.relative_to(PROJECT_ROOT)}")
    print(f"- {out_png.relative_to(PROJECT_ROOT)}")

    print("\nAggregated test metrics (sampled):")
    print(f"  R²={m_test['r2']:.6f}  RMSE={m_test['rmse']:.6f}  nRMSE={m_test['nrmse']:.3f}  corr={m_test['corr']:.3f}")
    print(f"  One-step RMSE={one_step_rmse:.6f}")

    print("\nAggregated PDE:")
    print(f"  {eq}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
