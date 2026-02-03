"""Analyze PDE discovery results and test simpler models.

This script is also used as a "source of truth" for presentation assets.

Important: selecting a "best" PDE purely by pointwise R² on u_t is brittle under
noise/registration errors. We therefore use a multi-objective selection that
prioritizes dynamical usefulness (rollout), then one-step error, then sparsity,
and only then R².
"""

import os

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "slides"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYZING PDE DISCOVERY RESULTS")
print("=" * 80)

# Train/test split (time axis) to reduce "high R² but wrong PDE" risk.
# Fit on early frames, evaluate on later frames.
TRAIN_FRAC = 0.7

# Spatial holdout split (image plane): fit on one region, evaluate on a disjoint region.
# This is a stronger generalization check when ground-truth PDE is unknown.
SPACE_TRAIN_FRAC = 0.7

# Multi-step rollout evaluation (explicit Euler) to test dynamical usefulness.
# Used for Figure 2 (rollout error vs horizon). Override via env var PDE_ROLLOUT_STEPS.
def _parse_rollout_steps(env_value: str | None, default_steps: tuple[int, ...]) -> tuple[int, ...]:
    if env_value is None:
        return default_steps
    s = str(env_value).strip()
    if not s:
        return default_steps
    # Accept "1,2,3" or "1-10"
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        lo = int(a.strip())
        hi = int(b.strip())
        if hi < lo:
            lo, hi = hi, lo
        return tuple(range(max(1, lo), max(1, hi) + 1))
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            k = int(p)
        except Exception:
            continue
        if k > 0:
            out.append(k)
    out = sorted(set(out))
    return tuple(out) if out else default_steps


ROLLOUT_STEPS = _parse_rollout_steps(os.getenv("PDE_ROLLOUT_STEPS"), tuple(range(1, 11)))


# Optional global translation stabilization (helps against camera jitter).
# Defaults are OFF to preserve current slide-generation behavior.
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


STABILIZE_TRANSLATION = _env_bool("PDE_STABILIZE_TRANSLATION", False)
STABILIZE_MODE = os.getenv("PDE_STABILIZE_MODE", "to_prev")  # "to_prev" or "to_first"
STABILIZE_EST_SIGMA = float(os.getenv("PDE_STABILIZE_EST_SIGMA", "2.0"))


def _shift_frame_reflect(frame: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Subpixel shift (y,x) using reflect border (good default for real images)."""
    M = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
    out = cv2.warpAffine(
        frame.astype(np.float32, copy=False),
        M,
        dsize=(frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out.astype(np.float32, copy=False)


def _estimate_shift_phasecorr(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    """Return (shift_y, shift_x) to apply to mov to align it to ref."""
    ref32 = ref.astype(np.float32, copy=False)
    mov32 = mov.astype(np.float32, copy=False)
    (dx, dy), _ = cv2.phaseCorrelate(ref32, mov32)
    return -float(dy), -float(dx)


def stabilize_translation_sequence(
    U: np.ndarray,
    *,
    mode: str = "to_prev",
    estimate_sigma: float = 2.0,
) -> np.ndarray:
    """Align frames by undoing global translations (phase correlation)."""
    mode = str(mode)
    if mode not in {"to_prev", "to_first"}:
        raise ValueError("mode must be 'to_prev' or 'to_first'")
    U = U.astype(np.float32, copy=False)
    out = np.empty_like(U)
    out[0] = U[0]
    ref = out[0]
    ref_est = gaussian_filter(ref, sigma=float(estimate_sigma)) if estimate_sigma > 0 else ref
    for t in range(1, U.shape[0]):
        mov = U[t]
        mov_est = gaussian_filter(mov, sigma=float(estimate_sigma)) if estimate_sigma > 0 else mov
        sy, sx = _estimate_shift_phasecorr(ref_est, mov_est)
        out[t] = _shift_frame_reflect(mov, shift_y=sy, shift_x=sx)
        if mode == "to_prev":
            ref = out[t]
            ref_est = gaussian_filter(ref, sigma=float(estimate_sigma)) if estimate_sigma > 0 else ref
    return out


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


def one_step_prediction_rmse(
    u_field: np.ndarray,
    ut_pred: np.ndarray,
    dt: float = 1.0,
    spatial_mask: np.ndarray | None = None,
) -> float:
    """One-step check: u(t+1) \approx u(t) + dt * u_t_pred(t).

    This is not a full PDE rollout, but it is a stronger sanity check than R² alone.
    """
    # Use forward step where possible
    t_max = min(u_field.shape[0] - 1, ut_pred.shape[0])
    if t_max <= 0:
        return float("nan")
    u0 = u_field[:t_max]
    u1 = u_field[1 : t_max + 1]
    u1_pred = u0 + dt * ut_pred[:t_max]

    err = (u1 - u1_pred) ** 2
    if spatial_mask is not None:
        spatial_mask = np.asarray(spatial_mask, dtype=bool)
        if spatial_mask.ndim != 2:
            raise ValueError("spatial_mask must be 2D (HxW)")
        if spatial_mask.shape != err.shape[1:]:
            raise ValueError(
                f"spatial_mask shape {spatial_mask.shape} does not match field shape {err.shape[1:]}"
            )
        m3 = np.broadcast_to(spatial_mask, err.shape)
        err = err[m3]
    return float(np.sqrt(np.mean(err)))


def split_time(t_len: int, train_frac: float) -> tuple[slice, slice]:
    if not (0.4 <= train_frac <= 0.9):
        raise ValueError("TRAIN_FRAC should be in [0.4, 0.9]")
    split = int(np.floor(train_frac * t_len))
    split = max(1, min(t_len - 1, split))
    return slice(0, split), slice(split, t_len)

# Load images and reproduce the analysis
IMAGE_FOLDER = DATA_DIR
OUTPUT_FOLDER = OUTPUT_DIR

# Load and process images (simplified - just use best registration)
image_files = sorted(IMAGE_FOLDER.glob("*.tif"))[:51]
U_raw = []
for img_file in image_files:
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        U_raw.append(img)

U_raw = np.array(U_raw, dtype=np.float32)
T, H, W = U_raw.shape
print(f"\n1. Loaded {T} images: {H}x{W} pixels")

# Downsample
U_raw = np.array([cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_AREA) for img in U_raw])
T, H, W = U_raw.shape

# Denoise and normalize
U_denoised = np.array([gaussian_filter(img, sigma=1.0) for img in U_raw])
U_norm = (U_denoised - U_denoised.min()) / (U_denoised.max() - U_denoised.min())

if STABILIZE_TRANSLATION:
    print("\n2.5. Translation stabilization (phase correlation)...")
    U_norm = stabilize_translation_sequence(
        U_norm,
        mode=str(STABILIZE_MODE),
        estimate_sigma=float(STABILIZE_EST_SIGMA),
    )

# Quick Farneback registration
print("\n2. Performing registration...")
registered = [U_norm[0].copy()]
for i in range(1, len(U_norm)):
    ref = (registered[-1] * 255).astype(np.uint8)
    mov = (U_norm[i] * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(ref, mov, None, 0.5, 5, 25, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = cv2.GaussianBlur(flow, (11, 11), 2.0)
    h, w = U_norm[i].shape
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w) - flow[:, :, 0]
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis] - flow[:, :, 1]
    warped = cv2.remap(U_norm[i], flow_map, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    registered.append(warped)
U_registered = np.array(registered)

# Smooth
print("\n3. Smoothing (skipping Savitzky-Golay for speed)...")
U_smooth = np.array([gaussian_filter(img, sigma=1.5) for img in U_registered])

# Crop and subsample
skip = 25
subsample = 12
U_crop = U_smooth[:, skip:-skip:subsample, skip:-skip:subsample]

# Derivatives
print("\n4. Computing derivatives...")
dx, dy, dt = 0.1, 0.1, 1.0

u_x = (U_crop[:, :, 2:] - U_crop[:, :, :-2]) / (2*dx)
u_y = (U_crop[:, 2:, :] - U_crop[:, :-2, :]) / (2*dy)
u_xx = (U_crop[:, :, 2:] - 2*U_crop[:, :, 1:-1] + U_crop[:, :, :-2]) / (dx**2)
u_yy = (U_crop[:, 2:, :] - 2*U_crop[:, 1:-1, :] + U_crop[:, :-2, :]) / (dy**2)
u_t = (U_crop[2:, :, :] - U_crop[:-2, :, :]) / (2*dt)

# Align
min_t = min(u_x.shape[0], u_y.shape[0], u_xx.shape[0], u_yy.shape[0], u_t.shape[0])
min_h = min(u_x.shape[1], u_y.shape[1], u_xx.shape[1], u_yy.shape[1], u_t.shape[1])
min_w = min(u_x.shape[2], u_y.shape[2], u_xx.shape[2], u_yy.shape[2], u_t.shape[2])

u = U_crop[:min_t, :min_h, :min_w]
u_x = u_x[:min_t, :min_h, :min_w]
u_y = u_y[:min_t, :min_h, :min_w]
u_xx = u_xx[:min_t, :min_h, :min_w]
u_yy = u_yy[:min_t, :min_h, :min_w]
u_t = u_t[:min_t, :min_h, :min_w]
laplacian = u_xx + u_yy

print(f"   Aligned shape: {min_t}x{min_h}x{min_w}")

train_sl, test_sl = split_time(min_t, TRAIN_FRAC)
print(f"   Train/Test split (time): train_t={train_sl.stop}, test_t={min_t - train_sl.stop} (TRAIN_FRAC={TRAIN_FRAC:.2f})")


def split_space_left_right(width: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.4 <= train_frac <= 0.9):
        raise ValueError("SPACE_TRAIN_FRAC should be in [0.4, 0.9]")
    split = int(np.floor(train_frac * width))
    split = max(1, min(width - 1, split))
    train_mask = np.zeros((min_h, min_w), dtype=bool)
    train_mask[:, :split] = True
    return train_mask, ~train_mask


def split_space_top_bottom(height: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.4 <= train_frac <= 0.9):
        raise ValueError("SPACE_TRAIN_FRAC should be in [0.4, 0.9]")
    split = int(np.floor(train_frac * height))
    split = max(1, min(height - 1, split))
    train_mask = np.zeros((min_h, min_w), dtype=bool)
    train_mask[:split, :] = True
    return train_mask, ~train_mask


def derivs_2d(
    field_2d: np.ndarray, dx_: float, dy_: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute u_x, u_y, u_xx, u_yy, lap(u) on the SAME grid via reflect padding."""
    f = np.asarray(field_2d, dtype=np.float64)
    fpad = np.pad(f, ((1, 1), (1, 1)), mode="reflect")

    u_x_ = (fpad[1:-1, 2:] - fpad[1:-1, :-2]) / (2.0 * dx_)
    u_y_ = (fpad[2:, 1:-1] - fpad[:-2, 1:-1]) / (2.0 * dy_)

    u_xx_ = (fpad[1:-1, 2:] - 2.0 * fpad[1:-1, 1:-1] + fpad[1:-1, :-2]) / (dx_**2)
    u_yy_ = (fpad[2:, 1:-1] - 2.0 * fpad[1:-1, 1:-1] + fpad[:-2, 1:-1]) / (dy_**2)
    lap_ = u_xx_ + u_yy_
    return u_x_, u_y_, u_xx_, u_yy_, lap_


def ut_from_pde(u2d: np.ndarray, terms: list[str], coeffs: np.ndarray) -> np.ndarray:
    """Compute u_t from current u using the model's term list."""
    u_x2, u_y2, u_xx2, u_yy2, lap2 = derivs_2d(u2d, dx, dy)

    term_map = {
        "1": np.ones_like(u2d, dtype=np.float64),
        "u": u2d.astype(np.float64),
        "u_x": u_x2,
        "u_y": u_y2,
        "u_xx": u_xx2,
        "u_yy": u_yy2,
        "lap(u)": lap2,
        "u^2": (u2d.astype(np.float64) ** 2),
        "u^3": (u2d.astype(np.float64) ** 3),
        "u*u_x": (u2d.astype(np.float64) * u_x2),
        "u*u_y": (u2d.astype(np.float64) * u_y2),
        "u_x^2": (u_x2**2),
        "u_y^2": (u_y2**2),
    }

    out = np.zeros_like(u2d, dtype=np.float64)
    for name, c in zip(terms, coeffs):
        if abs(float(c)) < 1e-12:
            continue
        if name not in term_map:
            raise KeyError(f"Rollout: unsupported term '{name}'")
        out += float(c) * term_map[name]
    return out


def rollout_k_rmse(
    u_true: np.ndarray,
    model_terms: list[str],
    model_coeffs: np.ndarray,
    k: int,
    time_slice: slice,
    spatial_mask: np.ndarray | None = None,
) -> dict:
    """k-step explicit Euler rollout RMSE over multiple start times.

    Predict u(t+k) from u(t) by iterating u_{n+1} = u_n + dt * f(u_n).
    Evaluate error on selected times and spatial region.
    """
    if k <= 0:
        return {"rmse": float("nan"), "nrmse": float("nan")}

    t0 = time_slice.start or 0
    t1 = time_slice.stop or u_true.shape[0]
    t1 = min(t1, u_true.shape[0])
    if t1 - t0 <= k:
        return {"rmse": float("nan"), "nrmse": float("nan")}

    errs = []
    trues = []
    for t in range(t0, t1 - k):
        u_pred = u_true[t].astype(np.float64)
        for _ in range(k):
            utp = ut_from_pde(u_pred, model_terms, model_coeffs)
            u_pred = u_pred + dt * utp
        u_target = u_true[t + k].astype(np.float64)
        diff = u_target - u_pred

        if spatial_mask is not None:
            m = np.asarray(spatial_mask, dtype=bool)
            diff = diff[m]
            u_target = u_target[m]
        errs.append(diff.ravel())
        trues.append(u_target.ravel())

    if not errs:
        return {"rmse": float("nan"), "nrmse": float("nan")}
    e = np.concatenate(errs)
    y = np.concatenate(trues)
    rmse = float(np.sqrt(np.mean(e**2)))
    nrmse = float(rmse / (float(np.std(y)) + 1e-12))
    return {"rmse": rmse, "nrmse": nrmse}


def rollout_predict_frame(
    u0: np.ndarray,
    model_terms: list[str],
    model_coeffs: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return u(t+k) predicted from u(t)=u0 via k explicit Euler steps."""
    u_pred = np.asarray(u0, dtype=np.float64)
    for _ in range(int(k)):
        utp = ut_from_pde(u_pred, model_terms, model_coeffs)
        u_pred = u_pred + dt * utp
    return u_pred


# =============================================================================
# ROBUST REGRESSION METHODS
# =============================================================================


def huber_weight(r: np.ndarray, delta: float = 1.35) -> np.ndarray:
    """Huber weight function: w = min(1, delta / |r|)."""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-12))


def irls_huber_fit(
    X: np.ndarray, y: np.ndarray, *, alpha: float = 0.01, delta: float = 1.35, max_iter: int = 50, tol: float = 1e-6
) -> np.ndarray:
    """Iteratively Reweighted Least Squares with Huber loss."""
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX + alpha * np.eye(p), Xty)

    for _ in range(max_iter):
        r = y - X @ beta
        sigma = float(np.median(np.abs(r)) * 1.4826 + 1e-12)
        r_scaled = r / sigma
        w = huber_weight(r_scaled, delta=delta)

        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ (w * y)
        beta_new = np.linalg.solve(XtWX + alpha * np.eye(p), XtWy)

        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new

    return beta


def robust_stridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.01,
    threshold: float = 1e-5,
    max_iter: int = 20,
    use_huber: bool = True,
    huber_delta: float = 1.35,
    trim_frac: float = 0.05,
    n_bootstrap: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full robust STRidge for real-image PDE discovery.

    Combines trimmed outlier rejection, Huber loss, and bootstrap ensemble.

    Returns:
        coeffs: Robust coefficient estimates (already unscaled)
        scale: Feature scaling factors
        info: Dict with uncertainty estimates
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, p = X_scaled.shape

    # Step 1: Initial fit to identify outliers
    c_init = np.linalg.lstsq(X_scaled, y, rcond=None)[0]
    resid = np.abs(y - X_scaled @ c_init)

    # Step 2: Trim worst outliers
    n_trim = int(n * trim_frac)
    if n_trim > 0:
        keep_idx = np.argsort(resid)[: n - n_trim]
        X_clean = X_scaled[keep_idx]
        y_clean = y[keep_idx]
    else:
        X_clean = X_scaled
        y_clean = y

    # Step 3: Bootstrap ensemble with Huber
    rng = np.random.default_rng(42)
    n_clean = len(y_clean)
    all_coeffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_clean, size=int(n_clean * 0.8), replace=True)
        X_sub = X_clean[idx]
        y_sub = y_clean[idx]

        if use_huber:
            c = irls_huber_fit(X_sub, y_sub, alpha=alpha, delta=huber_delta)
        else:
            XtX = X_sub.T @ X_sub
            Xty = X_sub.T @ y_sub
            c = np.linalg.solve(XtX + alpha * np.eye(p), Xty)

        # STRidge thresholding
        for _ in range(max_iter):
            small = np.abs(c) < threshold
            if small.all():
                c = np.zeros(p)
                break
            c[small] = 0.0
            big = ~small
            if use_huber:
                c_big = irls_huber_fit(X_sub[:, big], y_sub, alpha=alpha, delta=huber_delta)
            else:
                XtX = X_sub[:, big].T @ X_sub[:, big]
                Xty = X_sub[:, big].T @ y_sub
                c_big = np.linalg.solve(XtX + alpha * np.eye(big.sum()), Xty)
            c = np.zeros(p)
            c[big] = c_big

        all_coeffs.append(c)

    all_coeffs = np.stack(all_coeffs, axis=0)

    # Robust aggregation
    coeffs = np.median(all_coeffs, axis=0) / scaler.scale_
    std_coeffs = np.std(all_coeffs, axis=0) / scaler.scale_
    ci_low = np.percentile(all_coeffs, 2.5, axis=0) / scaler.scale_
    ci_high = np.percentile(all_coeffs, 97.5, axis=0) / scaler.scale_

    info = {
        "std": std_coeffs,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_trimmed": n_trim,
        "n_bootstrap": n_bootstrap,
    }

    return coeffs, scaler.scale_, info


# Environment variable to enable robust regression
USE_ROBUST_REGRESSION = _env_bool("PDE_ROBUST_REGRESSION", False)


# STRidge function
def stridge(X, y, alpha=0.01, threshold=1e-5, max_iter=20):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    coeffs = model.coef_
    
    for iteration in range(max_iter):
        small_indices = np.abs(coeffs) < threshold
        coeffs[small_indices] = 0
        big_indices = ~small_indices
        if big_indices.sum() == 0:
            break
        model.fit(X_scaled[:, big_indices], y)
        coeffs_big = model.coef_
        coeffs = np.zeros_like(coeffs)
        coeffs[big_indices] = coeffs_big
    
    return coeffs / scaler.scale_, scaler


def fit_pde_model(X_train, y_train, alpha=0.01, threshold=1e-5):
    """Wrapper to select standard or robust STRidge based on environment."""
    if USE_ROBUST_REGRESSION:
        coeffs, scale, info = robust_stridge(
            X_train, y_train,
            alpha=alpha,
            threshold=threshold,
            use_huber=True,
            huber_delta=1.35,
            trim_frac=0.05,
            n_bootstrap=30,
        )
        # Return in same format as stridge (coeffs, scaler-like object)
        class FakeScaler:
            def __init__(self, s):
                self.scale_ = s
        return coeffs, FakeScaler(scale), info
    else:
        coeffs, scaler = stridge(X_train, y_train, alpha=alpha, threshold=threshold)
        return coeffs, scaler, None


# Test different model complexities
print("\n5. Testing different model complexities...")
print("="*80)
if USE_ROBUST_REGRESSION:
    print("Using ROBUST regression (Huber + trimmed + bootstrap)")
else:
    print("Using STANDARD regression (Ridge)")

models = {
    "Model 1: Diffusion only": {
        "terms": [np.ones_like(u), u, laplacian],
        "names": ['1', 'u', 'lap(u)']
    },
    "Model 2: Diffusion + Linear Growth": {
        "terms": [np.ones_like(u), u, laplacian],
        "names": ['1', 'u', 'lap(u)']
    },
    "Model 3: + First order spatial": {
        "terms": [np.ones_like(u), u, u_x, u_y, laplacian],
        "names": ['1', 'u', 'u_x', 'u_y', 'lap(u)']
    },
    "Model 4: + Nonlinear (u^2)": {
        "terms": [np.ones_like(u), u, u_x, u_y, laplacian, u**2],
        "names": ['1', 'u', 'u_x', 'u_y', 'lap(u)', 'u^2']
    },
    "Model 5: + Advection (u*grad(u))": {
        "terms": [np.ones_like(u), u, u_x, u_y, laplacian, u**2, u*u_x, u*u_y],
        "names": ['1', 'u', 'u_x', 'u_y', 'lap(u)', 'u^2', 'u*u_x', 'u*u_y']
    },
    "Model 6: Full (original)": {
        "terms": [np.ones_like(u), u, u_x, u_y, u_xx, u_yy, laplacian, 
                  u**2, u*u_x, u*u_y, u**3, u_x**2, u_y**2],
        "names": ['1', 'u', 'u_x', 'u_y', 'u_xx', 'u_yy', 'lap(u)', 'u^2', 'u*u_x', 'u*u_y', 'u^3', 'u_x^2', 'u_y^2']
    }
}

results = []

for model_name, model_spec in models.items():
    X_train = np.column_stack([term[train_sl].ravel() for term in model_spec['terms']])
    y_train = u_t[train_sl].ravel()
    X_test = np.column_stack([term[test_sl].ravel() for term in model_spec['terms']])
    y_test = u_t[test_sl].ravel()

    coeffs, scaler, robust_info = fit_pde_model(X_train, y_train, alpha=0.01, threshold=1e-5)
    y_pred_train = X_train @ coeffs
    y_pred_test = X_test @ coeffs

    m_train = regression_metrics(y_train, y_pred_train)
    m_test = regression_metrics(y_test, y_pred_test)

    # One-step RMSE computed on train/test time segments separately
    ut_pred_full = np.zeros_like(u_t)
    ut_pred_full[train_sl] = y_pred_train.reshape(u_t[train_sl].shape)
    ut_pred_full[test_sl] = y_pred_test.reshape(u_t[test_sl].shape)
    one_step_rmse_train = one_step_prediction_rmse(u[train_sl], ut_pred_full[train_sl], dt=dt)
    one_step_rmse_test = one_step_prediction_rmse(u[test_sl], ut_pred_full[test_sl], dt=dt)

    # Rollout metrics (dynamical usefulness)
    rollout = {}
    for k in ROLLOUT_STEPS:
        rollout[f"k{k}_train"] = rollout_k_rmse(
            u,
            model_spec["names"],
            coeffs,
            int(k),
            train_sl,
        )
        rollout[f"k{k}_test"] = rollout_k_rmse(
            u,
            model_spec["names"],
            coeffs,
            int(k),
            test_sl,
        )
    
    # Count active terms
    n_active = np.sum(np.abs(coeffs) > 1e-5)
    
    # Build equation
    eq_parts = []
    for coeff, name in zip(coeffs, model_spec['names']):
        if np.abs(coeff) > 1e-5:
            sign = '+' if coeff > 0 and len(eq_parts) > 0 else ''
            eq_parts.append(f"{sign}{coeff:.4f}*{name}")
    
    equation = "u_t = " + " ".join(eq_parts) if eq_parts else "u_t = 0"
    
    results.append({
        'name': model_name,
        'r2': m_test['r2'],
        'rmse': m_test['rmse'],
        'mae': m_test['mae'],
        'nrmse': m_test['nrmse'],
        'corr': m_test['corr'],
        'resid_med_abs': m_test['resid_med_abs'],
        'one_step_rmse': one_step_rmse_test,
        'train_r2': m_train['r2'],
        'train_rmse': m_train['rmse'],
        'train_nrmse': m_train['nrmse'],
        'train_corr': m_train['corr'],
        'train_one_step_rmse': one_step_rmse_train,
        'rollout': rollout,
        'n_active': n_active,
        'n_total': len(coeffs),
        'equation': equation,
        'coeffs': coeffs,
        'names': model_spec['names']
    })
    
    print(f"\n{model_name}")
    print(f"  Train: R2={m_train['r2']:.6f} | RMSE={m_train['rmse']:.6f} | nRMSE={m_train['nrmse']:.3f} | corr={m_train['corr']:.3f} | one-step={one_step_rmse_train:.6f}")
    print(f"  Test:  R2={m_test['r2']:.6f} | RMSE={m_test['rmse']:.6f} | nRMSE={m_test['nrmse']:.3f} | corr={m_test['corr']:.3f} | one-step={one_step_rmse_test:.6f}")
    if ROLLOUT_STEPS:
        k_show = int(ROLLOUT_STEPS[-1])
        rk_tr = rollout.get(f"k{k_show}_train", {})
        rk_te = rollout.get(f"k{k_show}_test", {})
        if rk_tr and rk_te:
            print(
                f"  Rollout k={k_show}: train_nRMSE={rk_tr.get('nrmse', float('nan')):.3f} | test_nRMSE={rk_te.get('nrmse', float('nan')):.3f}"
            )
    print(f"  Active terms: {n_active}/{len(coeffs)}")
    print(f"  {equation}")

# Create comparison figure
print("\n6. Creating comparison figure...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for idx, result in enumerate(results):
    ax = axes[idx]
    
    # Bar chart of coefficients
    colors = ['red' if abs(c) > 1e-5 else 'lightgray' for c in result['coeffs']]
    ax.bar(range(len(result['coeffs'])), result['coeffs'], color=colors, edgecolor='black')
    ax.set_xticks(range(len(result['names'])))
    ax.set_xticklabels(result['names'], fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_title(f"{result['name']}\nR2 = {result['r2']:.4f}, Active: {result['n_active']}/{result['n_total']}", 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / 'MODEL_COMPARISON.png', dpi=300, bbox_inches='tight')
print(f"   Saved: MODEL_COMPARISON.png")

# Summary table
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)
print("\n{:<40} {:>10} {:>10} {:>10} {:>15}".format("Model", "R2(test)", "RMSE", "nRMSE", "Active Terms"))
print("-" * 80)
for result in results:
    print("{:<40} {:>10.6f} {:>10.6f} {:>10.3f} {:>15}".format(
        result['name'], 
        result['r2'], 
        result['rmse'],
        result['nrmse'],
        f"{result['n_active']}/{result['n_total']}"
    ))

print("\n" + "="*80)
print("OBSERVATIONS:")
print("="*80)

def _rank(values: list[float], *, reverse: bool) -> list[int]:
    # Returns rank indices: 0 is best.
    order = np.argsort(values)
    if reverse:
        order = order[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return [int(r) for r in ranks]


def select_best(results_list: list[dict], *, max_active: int | None = None) -> dict:
    """Multi-objective selection.

    Priority (best first):
      1) rollout test nRMSE at largest k
      2) one-step RMSE (test)
      3) sparsity (n_active)
      4) test R²
    Implemented via weighted rank-sum to avoid unit sensitivity.
    """
    cand = [r for r in results_list if (max_active is None or int(r.get("n_active", 0)) <= int(max_active))]
    if not cand:
        return results_list[0]

    k_eval = int(ROLLOUT_STEPS[-1]) if ROLLOUT_STEPS else 0
    rollout_nrmse = []
    for r in cand:
        v = float("inf")
        if k_eval > 0:
            v = float(r.get("rollout", {}).get(f"k{k_eval}_test", {}).get("nrmse", float("inf")))
        rollout_nrmse.append(v)

    one_step = [float(r.get("one_step_rmse", float("inf"))) for r in cand]
    n_active = [float(r.get("n_active", float("inf"))) for r in cand]
    r2 = [float(r.get("r2", float("-inf"))) for r in cand]

    rank_roll = _rank(rollout_nrmse, reverse=False)
    rank_step = _rank(one_step, reverse=False)
    rank_sparse = _rank(n_active, reverse=False)
    rank_r2 = _rank(r2, reverse=True)

    # Heavily prefer dynamical stability, then one-step, then sparsity, then R².
    scores = [
        5.0 * rr + 3.0 * rs + 1.0 * r0 + 1.0 * r2r
        for rr, rs, r0, r2r in zip(rank_roll, rank_step, rank_sparse, rank_r2)
    ]
    best_idx = int(np.argmin(scores))
    out = dict(cand[best_idx])
    out["selection"] = {
        "k_eval": k_eval,
        "rank_rollout": rank_roll[best_idx],
        "rank_one_step": rank_step[best_idx],
        "rank_sparsity": rank_sparse[best_idx],
        "rank_r2": rank_r2[best_idx],
        "score": float(scores[best_idx]),
    }
    return out


# Find best models with multi-objective selection
best_simple = select_best(results, max_active=5)
best_overall = select_best(results, max_active=None)


def spatial_holdout_eval(model_key: str) -> dict:
    """Fit on train spatial region, evaluate on disjoint region (all times)."""
    if model_key not in models:
        return {}
    spec = models[model_key]
    train_mask2, test_mask2 = split_space_left_right(min_w, SPACE_TRAIN_FRAC)
    train_mask3 = np.broadcast_to(train_mask2, u_t.shape)
    test_mask3 = np.broadcast_to(test_mask2, u_t.shape)

    X_train = np.column_stack([term.ravel()[train_mask3.ravel()] for term in spec['terms']])
    y_train = u_t.ravel()[train_mask3.ravel()]
    X_test = np.column_stack([term.ravel()[test_mask3.ravel()] for term in spec['terms']])
    y_test = u_t.ravel()[test_mask3.ravel()]

    coeffs, _, _ = fit_pde_model(X_train, y_train, alpha=0.01, threshold=1e-5)
    y_pred_train = X_train @ coeffs
    y_pred_test = X_test @ coeffs

    m_train = regression_metrics(y_train, y_pred_train)
    m_test = regression_metrics(y_test, y_pred_test)

    # One-step check on each spatial region, using the same coefficients.
    # Compute u_t prediction everywhere for the chosen model.
    Theta_all = np.column_stack([term.ravel() for term in spec['terms']])
    ut_pred_all = (Theta_all @ coeffs).reshape(u_t.shape)
    one_step_train = one_step_prediction_rmse(u, ut_pred_all, dt=dt, spatial_mask=train_mask2)
    one_step_test = one_step_prediction_rmse(u, ut_pred_all, dt=dt, spatial_mask=test_mask2)

    return {
        "space_train_frac": SPACE_TRAIN_FRAC,
        "space_split": "left_right",
        "train": {
            **m_train,
            "one_step_rmse": one_step_train,
        },
        "test": {
            **m_test,
            "one_step_rmse": one_step_test,
        },
        "coeffs": [float(c) for c in coeffs],
        "terms": spec["names"],
    }


def spatial_holdout_eval_top_bottom(model_key: str) -> dict:
    if model_key not in models:
        return {}
    spec = models[model_key]
    train_mask2, test_mask2 = split_space_top_bottom(min_h, SPACE_TRAIN_FRAC)
    train_mask3 = np.broadcast_to(train_mask2, u_t.shape)
    test_mask3 = np.broadcast_to(test_mask2, u_t.shape)

    X_train = np.column_stack([term.ravel()[train_mask3.ravel()] for term in spec['terms']])
    y_train = u_t.ravel()[train_mask3.ravel()]
    X_test = np.column_stack([term.ravel()[test_mask3.ravel()] for term in spec['terms']])
    y_test = u_t.ravel()[test_mask3.ravel()]

    coeffs, _, _ = fit_pde_model(X_train, y_train, alpha=0.01, threshold=1e-5)
    y_pred_train = X_train @ coeffs
    y_pred_test = X_test @ coeffs

    m_train = regression_metrics(y_train, y_pred_train)
    m_test = regression_metrics(y_test, y_pred_test)

    Theta_all = np.column_stack([term.ravel() for term in spec['terms']])
    ut_pred_all = (Theta_all @ coeffs).reshape(u_t.shape)
    one_step_train = one_step_prediction_rmse(u, ut_pred_all, dt=dt, spatial_mask=train_mask2)
    one_step_test = one_step_prediction_rmse(u, ut_pred_all, dt=dt, spatial_mask=test_mask2)

    return {
        "space_train_frac": SPACE_TRAIN_FRAC,
        "space_split": "top_bottom",
        "train": {
            **m_train,
            "one_step_rmse": one_step_train,
        },
        "test": {
            **m_test,
            "one_step_rmse": one_step_test,
        },
        "coeffs": [float(c) for c in coeffs],
        "terms": spec["names"],
    }

print(f"\n1. Best SIMPLE model (<=5 terms):")
print(f"   {best_simple['name']}")
print(f"   Test R2 = {best_simple['r2']:.6f}")
print(f"   {best_simple['equation']}")

print(f"\n2. Best OVERALL model:")
print(f"   {best_overall['name']}")
print(f"   Test R2 = {best_overall['r2']:.6f}")
print(f"   {best_overall['equation']}")

print("\n2b. Spatial holdout (best overall):")
spatial_eval = spatial_holdout_eval(best_overall["name"])
spatial_eval_tb = spatial_holdout_eval_top_bottom(best_overall["name"])
if spatial_eval:
    s_tr = spatial_eval["train"]
    s_te = spatial_eval["test"]
    print(
        f"   Train(space): R2={s_tr['r2']:.6f} | RMSE={s_tr['rmse']:.6f} | nRMSE={s_tr['nrmse']:.3f} | corr={s_tr['corr']:.3f} | one-step={s_tr['one_step_rmse']:.6f}"
    )
    print(
        f"   Test(space):  R2={s_te['r2']:.6f} | RMSE={s_te['rmse']:.6f} | nRMSE={s_te['nrmse']:.3f} | corr={s_te['corr']:.3f} | one-step={s_te['one_step_rmse']:.6f}"
    )
if spatial_eval_tb:
    s_tr = spatial_eval_tb["train"]
    s_te = spatial_eval_tb["test"]
    print(
        f"   Train(space TB): R2={s_tr['r2']:.6f} | RMSE={s_tr['rmse']:.6f} | nRMSE={s_tr['nrmse']:.3f} | corr={s_tr['corr']:.3f} | one-step={s_tr['one_step_rmse']:.6f}"
    )
    print(
        f"   Test(space TB):  R2={s_te['r2']:.6f} | RMSE={s_te['rmse']:.6f} | nRMSE={s_te['nrmse']:.3f} | corr={s_te['corr']:.3f} | one-step={s_te['one_step_rmse']:.6f}"
    )

print("\n2c. Multi-step rollout (best overall, time-test frames):")
rollout = {}
for k in ROLLOUT_STEPS:
    rollout[k] = rollout_k_rmse(u, best_overall["names"], best_overall["coeffs"], k=k, time_slice=test_sl)
    print(f"   k={k}: RMSE={rollout[k]['rmse']:.6f} | nRMSE={rollout[k]['nrmse']:.3f}")

print("\n2d. Multi-step rollout on spatial-test regions (using time-fit coeffs):")
rollout_space = {}
rollout_space_tb = {}
try:
    lr_train, lr_test = split_space_left_right(min_w, SPACE_TRAIN_FRAC)
    tb_train, tb_test = split_space_top_bottom(min_h, SPACE_TRAIN_FRAC)
    for k in ROLLOUT_STEPS:
        rollout_space[k] = rollout_k_rmse(u, best_overall["names"], best_overall["coeffs"], k=k, time_slice=slice(0, min_t), spatial_mask=lr_test)
        rollout_space_tb[k] = rollout_k_rmse(u, best_overall["names"], best_overall["coeffs"], k=k, time_slice=slice(0, min_t), spatial_mask=tb_test)
        print(f"   LR test, k={k}: RMSE={rollout_space[k]['rmse']:.6f} | nRMSE={rollout_space[k]['nrmse']:.3f}")
        print(f"   TB test, k={k}: RMSE={rollout_space_tb[k]['rmse']:.6f} | nRMSE={rollout_space_tb[k]['nrmse']:.3f}")
except Exception as e:
    print(f"   (skipped spatial rollout: {e})")

print(f"\n3. R2 improvement from simplest to most complex:")
r2_improvement = (best_overall['r2'] - results[0]['r2']) / abs(results[0]['r2']) * 100
print(f"   {results[0]['r2']:.6f} -> {best_overall['r2']:.6f} ({r2_improvement:+.1f}%)")

print("\n4. PHYSICAL INTERPRETATION:")
if best_simple['n_active'] == 3 and 'u' in best_simple['equation'] and 'lap(u)' in best_simple['equation']:
    print("   -> Reaction-Diffusion PDE: u_t = a*u + b*lap(u)")
    print("   -> Linear growth + spatial diffusion")
    print("   -> Similar to Fisher-KPP or heat equation with source")
elif 'u^2' in best_simple['equation']:
    print("   -> Nonlinear dynamics present (u^2 term)")
    print("   -> Amplitude-dependent behavior")
elif 'u*u_x' in best_simple['equation'] or 'u*u_y' in best_simple['equation']:
    print("   -> Advection present (u*grad(u) terms)")
    print("   -> Wave propagation dynamics")

print("\n5. RECOMMENDATION:")
if best_overall['r2'] - best_simple['r2'] < 0.01:
    print(f"   Use SIMPLE model: {best_simple['name']}")
    print(f"   Reason: Only {abs(best_overall['r2'] - best_simple['r2']):.6f} R2 loss, much simpler")
else:
    print(f"   Complex model needed for good fit")
    print(f"   R2 gain from complexity: {best_overall['r2'] - best_simple['r2']:.6f}")

print("="*80)

# Save best model (for slides/figures)
best_model_path = OUTPUT_FOLDER / "best_model.json"
best_payload = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "selection": "best_test_by_r2",
    "train_frac": TRAIN_FRAC,
    "name": best_overall["name"],
    "r2": best_overall["r2"],
    "rmse": best_overall["rmse"],
    "mae": best_overall["mae"],
    "nrmse": best_overall["nrmse"],
    "corr": best_overall["corr"],
    "resid_med_abs": best_overall["resid_med_abs"],
    "one_step_rmse": best_overall["one_step_rmse"],
    "train_r2": best_overall.get("train_r2"),
    "train_rmse": best_overall.get("train_rmse"),
    "train_nrmse": best_overall.get("train_nrmse"),
    "train_corr": best_overall.get("train_corr"),
    "train_one_step_rmse": best_overall.get("train_one_step_rmse"),
    "n_active": int(best_overall["n_active"]),
    "n_total": int(best_overall["n_total"]),
    "equation": best_overall["equation"],
    "terms": best_overall["names"],
    "coeffs": [float(c) for c in best_overall["coeffs"]],
    "spatial_holdout": spatial_eval,
    "spatial_holdout_top_bottom": spatial_eval_tb,
    "rollout_time_test": {
        "steps": list(ROLLOUT_STEPS),
        "metrics": {str(k): rollout[k] for k in ROLLOUT_STEPS},
    },
    "rollout_space_test_left_right": {
        "steps": list(ROLLOUT_STEPS),
        "metrics": {str(k): rollout_space.get(k, {}) for k in ROLLOUT_STEPS},
    },
    "rollout_space_test_top_bottom": {
        "steps": list(ROLLOUT_STEPS),
        "metrics": {str(k): rollout_space_tb.get(k, {}) for k in ROLLOUT_STEPS},
    },
}
with best_model_path.open("w", encoding="utf-8") as f:
    json.dump(best_payload, f, indent=2)
print(f"\n   Saved: best_model.json")


# Save full model comparison (for downstream comparative figures)
def _to_builtin(x):
    try:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.ravel().tolist()]
    return x


models_table_path = OUTPUT_FOLDER / "models_comparison.json"
models_table = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "train_frac": TRAIN_FRAC,
    "space_train_frac": SPACE_TRAIN_FRAC,
    "rollout_steps": list(ROLLOUT_STEPS),
    "use_robust_regression": bool(USE_ROBUST_REGRESSION),
    "best_simple": {
        "name": best_simple.get("name"),
        "r2_test": _to_builtin(best_simple.get("r2")),
        "one_step_rmse": _to_builtin(best_simple.get("one_step_rmse")),
        "n_active": int(best_simple.get("n_active", 0)),
    },
    "best_overall": {
        "name": best_overall.get("name"),
        "r2_test": _to_builtin(best_overall.get("r2")),
        "one_step_rmse": _to_builtin(best_overall.get("one_step_rmse")),
        "n_active": int(best_overall.get("n_active", 0)),
    },
    "models": [],
}

for r in results:
    k_eval = int(ROLLOUT_STEPS[-1]) if ROLLOUT_STEPS else 0
    rollout_k_test = {}
    if k_eval > 0:
        rollout_k_test = r.get("rollout", {}).get(f"k{k_eval}_test", {}) or {}

    rollout_curve_test = {
        str(int(k)): {
            kk: _to_builtin(vv)
            for kk, vv in (r.get("rollout", {}).get(f"k{int(k)}_test", {}) or {}).items()
        }
        for k in ROLLOUT_STEPS
    }

    models_table["models"].append(
        {
            "name": r.get("name"),
            "r2_test": _to_builtin(r.get("r2")),
            "rmse_test": _to_builtin(r.get("rmse")),
            "nrmse_test": _to_builtin(r.get("nrmse")),
            "corr_test": _to_builtin(r.get("corr")),
            "one_step_rmse": _to_builtin(r.get("one_step_rmse")),
            "n_active": int(r.get("n_active", 0)),
            "n_total": int(r.get("n_total", 0)),
            "equation": r.get("equation"),
            "terms": list(r.get("names", [])),
            "coeffs": [float(c) for c in np.asarray(r.get("coeffs"), dtype=float).ravel().tolist()],
            "rollout": {
                "k_eval": k_eval,
                "test": {k: _to_builtin(v) for k, v in rollout_k_test.items()},
            },
            "rollout_curve_test": {
                "steps": list(ROLLOUT_STEPS),
                "metrics": rollout_curve_test,
            },
        }
    )

with models_table_path.open("w", encoding="utf-8") as f:
    json.dump(models_table, f, indent=2)
print(f"\n   Saved: models_comparison.json")


# Qualitative rollout snapshot (Figure 4 helper asset)
try:
    if ROLLOUT_STEPS:
        # Export a small sweep of qualitative snapshots for the pitch deck.
        # Keep the existing default (last rollout step) but also generate k=5..9.
        k_default = int(ROLLOUT_STEPS[-1])
        k_snaps = [k for k in range(5, 10)]
        if k_default not in k_snaps:
            k_snaps.append(k_default)
        r3 = next((r for r in results if str(r.get("name", "")).strip().startswith("Model 3")), None)
        r4 = next((r for r in results if str(r.get("name", "")).strip().startswith("Model 4")), None)
        if r3 is not None and r4 is not None:
            base_t_start = int(test_sl.start or 0)
            err_maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            # Compute error maps for all requested horizons first so we can
            # use a shared color scale (more comparable across k).
            for k_snap in sorted(set(int(k) for k in k_snaps)):
                if k_snap <= 0:
                    continue
                if int(u.shape[0]) <= k_snap:
                    continue

                t_start = max(0, min(base_t_start, int(u.shape[0]) - k_snap - 1))
                if t_start + k_snap >= int(u.shape[0]):
                    continue

                u0 = u[t_start]
                gt = u[t_start + k_snap]
                pred4 = rollout_predict_frame(
                    u0,
                    list(r4.get("names", [])),
                    np.asarray(r4.get("coeffs"), dtype=float),
                    k_snap,
                )
                pred3 = rollout_predict_frame(
                    u0,
                    list(r3.get("names", [])),
                    np.asarray(r3.get("coeffs"), dtype=float),
                    k_snap,
                )

                err4 = np.abs(gt.astype(np.float64) - pred4.astype(np.float64))
                err3 = np.abs(gt.astype(np.float64) - pred3.astype(np.float64))
                err_maps[k_snap] = (err4, err3)

            if not err_maps:
                raise RuntimeError("No valid k values for qualitative rollout snapshot")

            stack_err = np.stack([v for pair in err_maps.values() for v in pair], axis=0)
            finite = np.isfinite(stack_err)
            if finite.any():
                vals = stack_err[finite]
                vmin, vmax = np.percentile(vals, [1, 99])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
                    vmin, vmax = float(np.min(vals)), float(np.max(vals))
            else:
                vmin, vmax = None, None

            # Precompute rollout curves up to the largest requested horizon.
            k_max = max(err_maps)
            ks_full = list(range(1, int(k_max) + 1))
            curve4_full = [
                rollout_k_rmse(
                    u,
                    list(r4.get("names", [])),
                    np.asarray(r4.get("coeffs"), dtype=float),
                    k=kk,
                    time_slice=test_sl,
                )["nrmse"]
                for kk in ks_full
            ]
            curve3_full = [
                rollout_k_rmse(
                    u,
                    list(r3.get("names", [])),
                    np.asarray(r3.get("coeffs"), dtype=float),
                    k=kk,
                    time_slice=test_sl,
                )["nrmse"]
                for kk in ks_full
            ]

            for k_snap in sorted(err_maps):
                err4, err3 = err_maps[k_snap]

                fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                im0 = axes[0].imshow(err4, cmap="magma", vmin=vmin, vmax=vmax)
                axes[0].set_title(f"M4 |error| (k={k_snap})")
                axes[1].imshow(err3, cmap="magma", vmin=vmin, vmax=vmax)
                axes[1].set_title(f"M3 |error| (k={k_snap})")
                fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)
                plt.tight_layout()
                out_img = OUTPUT_FOLDER / f"FIG4_QUAL_ROLLOUT_K{k_snap}.png"
                plt.savefig(out_img, dpi=240, bbox_inches="tight")
                plt.close(fig)
                print(f"\n   Saved: {out_img.name}")

                # Figure 4B: error growth curve (k=1..k_snap) + one snapshot error map (M4).
                ks = ks_full[: int(k_snap)]
                curve4 = curve4_full[: int(k_snap)]
                curve3 = curve3_full[: int(k_snap)]

                fig2, axes2 = plt.subplots(1, 2, figsize=(11.0, 4.0))
                axes2[0].plot(ks, curve4, marker="o", linewidth=2.0, label="M4")
                axes2[0].plot(ks, curve3, marker="o", linewidth=2.0, label="M3")
                axes2[0].set_yscale("log")
                axes2[0].set_xlabel("Horizon k")
                axes2[0].set_ylabel("Rollout nRMSE (lower is better)")
                axes2[0].set_title("Error compounds over rollout")
                axes2[0].grid(True, alpha=0.25)
                axes2[0].legend(frameon=False, ncol=2)

                im = axes2[1].imshow(err4, cmap="magma", vmin=vmin, vmax=vmax)
                axes2[1].set_xticks([])
                axes2[1].set_yticks([])
                axes2[1].set_title(f"M4 |error| at k={k_snap}")
                fig2.colorbar(im, ax=axes2[1], fraction=0.046, pad=0.04)
                plt.tight_layout()
                out_img2 = OUTPUT_FOLDER / f"FIG4B_ERROR_GROWTH_PLUS_MAP_K{k_snap}.png"
                plt.savefig(out_img2, dpi=240, bbox_inches="tight")
                plt.close(fig2)
                print(f"\n   Saved: {out_img2.name}")
except Exception as e:
    print(f"\n   (skipped qualitative rollout snapshot: {e})")
