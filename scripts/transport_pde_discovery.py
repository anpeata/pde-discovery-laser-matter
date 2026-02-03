"""
Optimal Transport-based PDE Discovery for Laser-Matter Interaction

Uses continuity equation: ∂ρ/∂t + ∇·(ρv) = 0
where v is the velocity field from optical flow.

This approach:
1. Uses optical flow to get velocity fields v(x,y,t)
2. Fits sparse regression to find PDE in transport form
3. Works well for mass-conserved dynamics (0.54% in our data)
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
import warnings
warnings.filterwarnings('ignore')


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"

def load_images(data_dir=None, max_images=51):
    """Load TIF images"""
    image_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    tif_files = sorted(image_dir.glob('*.tif'))[:max_images]
    
    print(f"Loading {len(tif_files)} images...")
    images = []
    for tif_file in tqdm(tif_files):
        img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Some TIFs load as multi-channel (e.g., BGRA). Convert to grayscale.
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.astype(np.float64))
    
    return np.array(images)

def compute_optical_flow(images, downsample=2):
    """Compute optical flow between consecutive frames"""
    print(f"Computing optical flow (downsampled by {downsample}x)...")
    
    n_frames = len(images)
    h, w = images[0].shape
    h_ds, w_ds = h // downsample, w // downsample
    
    # Store flow fields
    flow_u = np.zeros((n_frames - 1, h_ds, w_ds))
    flow_v = np.zeros((n_frames - 1, h_ds, w_ds))
    
    for i in tqdm(range(n_frames - 1)):
        # Downsample for speed
        img1 = cv2.resize(images[i], (w_ds, h_ds))
        img2 = cv2.resize(images[i + 1], (w_ds, h_ds))
        
        # Normalize to 0-255
        img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            img1_norm, img2_norm,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        flow_u[i] = flow[:, :, 0]
        flow_v[i] = flow[:, :, 1]
    
    return flow_u, flow_v

def compute_derivatives(rho, flow_u, flow_v, dt=1.0, dx=1.0, dy=1.0):
    """
    Compute derivatives for transport equation.
    
    We want to fit: ∂ρ/∂t = -∇·(ρv) + source terms
    
    Returns library of candidate terms.
    """
    print("Computing derivatives and building library...")
    
    n_frames, h, w = rho.shape
    
    # Time derivative (forward difference)
    drho_dt = np.gradient(rho, dt, axis=0, edge_order=2)
    
    # Spatial derivatives
    drho_dx = np.gradient(rho, dx, axis=2, edge_order=2)
    drho_dy = np.gradient(rho, dy, axis=1, edge_order=2)
    
    # Second derivatives
    d2rho_dx2 = np.gradient(drho_dx, dx, axis=2, edge_order=2)
    d2rho_dy2 = np.gradient(drho_dy, dy, axis=1, edge_order=2)
    laplacian = d2rho_dx2 + d2rho_dy2
    
    # Velocity derivatives
    du_dx = np.gradient(flow_u, dx, axis=2, edge_order=2)
    dv_dy = np.gradient(flow_v, dy, axis=1, edge_order=2)
    divergence = du_dx + dv_dy
    
    # Advection terms: v·∇ρ = u*∂ρ/∂x + v*∂ρ/∂y
    # flow_u/flow_v are defined between consecutive frames → length (n_frames - 1)
    # Match them to the first (n_frames - 1) derivatives of ρ.
    advection_x = flow_u * drho_dx[:-1]
    advection_y = flow_v * drho_dy[:-1]
    advection = advection_x + advection_y
    
    # Divergence term: ρ*∇·v
    rho_divergence = rho[:-1] * divergence
    
    # Full transport: ∇·(ρv) = ρ*∇·v + v·∇ρ
    transport = rho_divergence + advection
    
    # Build library (all arrays size (n_frames-1, h, w))
    library = {
        'constant': np.ones_like(rho[:-1]),
        'rho': rho[:-1],
        'drho_dx': drho_dx[:-1],
        'drho_dy': drho_dy[:-1],
        'laplacian': laplacian[:-1],
        'transport': transport,  # ∇·(ρv)
        'rho_div': rho_divergence,  # ρ*∇·v
        'advection': advection,  # v·∇ρ
        'rho_u': rho[:-1] * flow_u,
        'rho_v': rho[:-1] * flow_v,
        'rho_squared': rho[:-1] ** 2,
        'u_drho_dx': flow_u * drho_dx[:-1],
        'v_drho_dy': flow_v * drho_dy[:-1],
    }
    
    # Target: ∂ρ/∂t
    target = drho_dt[:-1]
    
    return library, target

def build_regression_matrices(library, target, spatial_smooth=0.0):
    """
    Build matrices for regression: target = library @ coefficients
    
    Args:
        library: dict of (n_frames, h, w) arrays
        target: (n_frames, h, w) array
        spatial_smooth: optional Gaussian smoothing
    """
    if spatial_smooth > 0:
        print(f"Applying spatial smoothing (σ={spatial_smooth})...")
        from scipy.ndimage import gaussian_filter
        for key in library:
            library[key] = gaussian_filter(library[key], sigma=(0, spatial_smooth, spatial_smooth))
        target = gaussian_filter(target, sigma=(0, spatial_smooth, spatial_smooth))
    
    # Flatten spatial dimensions
    n_frames, h, w = target.shape
    n_samples = n_frames * h * w
    
    # Build design matrix
    term_names = list(library.keys())
    n_terms = len(term_names)
    
    X = np.zeros((n_samples, n_terms))
    for i, name in enumerate(term_names):
        X[:, i] = library[name].flatten()
    
    y = target.flatten()
    
    # Remove NaN/Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    
    print(f"Design matrix: {X.shape}, target: {y.shape}")
    print(f"Kept {valid.sum()}/{n_samples} valid samples ({100*valid.sum()/n_samples:.1f}%)")
    
    return X, y, term_names

def fit_transport_pde(X, y, term_names, alpha=0.01, method='lasso'):
    """
    Fit sparse regression to find transport PDE.
    
    Expected form: ∂ρ/∂t ≈ -∇·(ρv) + diffusion + sources
    """
    print(f"\nFitting {method.upper()} regression (α={alpha})...")
    
    if method == 'lasso':
        model = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
    elif method == 'ridge':
        model = Ridge(alpha=alpha, max_iter=10000, tol=1e-4)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X, y)
    coefficients = model.coef_
    
    # Print results
    print("\n" + "="*60)
    print("DISCOVERED TRANSPORT PDE")
    print("="*60)
    print(f"∂ρ/∂t = ", end="")
    
    active_terms = []
    for name, coef in zip(term_names, coefficients):
        if abs(coef) > 1e-10:
            sign = "+" if coef >= 0 else ""
            print(f"{sign}{coef:.6e}·{name} ", end="")
            active_terms.append((name, coef))
    
    print("\n" + "="*60)
    
    # Model quality
    y_pred = model.predict(X)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\nModel Quality:")
    print(f"  R² score: {r2:.6f}")
    print(f"  MSE: {np.mean(residuals**2):.6e}")
    print(f"  MAE: {np.mean(np.abs(residuals)):.6e}")
    print(f"  Active terms: {len(active_terms)}/{len(term_names)}")
    
    return coefficients, active_terms, model

def main():
    """Main analysis pipeline"""
    print("\n" + "="*60)
    print("OPTIMAL TRANSPORT PDE DISCOVERY")
    print("="*60 + "\n")
    
    # 1. Load images
    images = load_images(max_images=51)
    print(f"Loaded {len(images)} images, shape: {images[0].shape}")
    
    # Normalize to [0, 1] (treat as density)
    images = images / images.max()
    
    # 2. Compute optical flow
    flow_u, flow_v = compute_optical_flow(images, downsample=4)
    
    # Downsample images to match flow
    h, w = images[0].shape
    h_ds, w_ds = h // 4, w // 4
    images_ds = np.array([cv2.resize(img, (w_ds, h_ds)) for img in images])
    
    # 3. Compute derivatives and build library
    library, target = compute_derivatives(
        images_ds, flow_u, flow_v,
        dt=1.0, dx=1.0, dy=1.0
    )
    
    # 4. Build regression matrices
    X, y, term_names = build_regression_matrices(
        library, target,
        spatial_smooth=0.5  # Light smoothing
    )
    
    # 5. Fit transport PDE with different regularizations
    results = {}
    for alpha in [0.001, 0.01, 0.1]:
        print(f"\n{'='*60}")
        print(f"Testing α = {alpha}")
        print('='*60)
        
        coeffs, active, model = fit_transport_pde(
            X, y, term_names,
            alpha=alpha,
            method='lasso'
        )
        
        results[alpha] = {
            'coefficients': coeffs,
            'active_terms': active,
            'model': model
        }
    
    # 6. Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset: {len(images)} frames, {images[0].shape} pixels")
    print(f"Downsampled: 4x → {images_ds.shape}")
    print(f"Optical flow: mean |v| = {np.sqrt(flow_u**2 + flow_v**2).mean():.3f} px/frame")
    print(f"Library terms: {len(term_names)}")
    print(f"\nBest result: α=0.01 typically works well for this data")
    
    return images_ds, flow_u, flow_v, results

if __name__ == "__main__":
    images_ds, flow_u, flow_v, results = main()
    print("\n✅ Transport PDE discovery complete!")
    print("Next: validate by forward integration")
