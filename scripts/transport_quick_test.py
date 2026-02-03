"""Quick test of transport PDE approach - minimal dependencies"""

import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"

print("="*60)
print("TRANSPORT PDE QUICK TEST")
print("="*60)

# Load first 10 images quickly
print("\n1. Loading images...")
image_dir = DATA_DIR
tif_files = sorted(image_dir.glob('*.tif'))[:10]
print(f"   Found {len(tif_files)} TIF files")

images = []
for i, tif_file in enumerate(tif_files):
    print(f"   Loading image {i+1}/{len(tif_files)}...", end='\r')
    img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    if img is not None:
        # Convert to grayscale if multi-channel
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img.astype(np.float64))

images = np.array(images)
images = images / np.max(images)  # Normalize to [0,1]
print(f"   Loaded {len(images)} images: {images[0].shape}                ")

# Downsample 8x for speed
print("\n2. Downsampling 8x...")
if len(images[0].shape) == 3:
    h, w, _ = images[0].shape
else:
    h, w = images[0].shape
images_ds = np.array([cv2.resize(img, (w//8, h//8)) for img in images])
print(f"   Downsampled to: {images_ds.shape}")

# Compute optical flow
print("\n3. Computing optical flow...")
flow_u = []
flow_v = []
for i in range(len(images_ds) - 1):
    img1 = cv2.normalize(images_ds[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2 = cv2.normalize(images_ds[i+1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_u.append(flow[:,:,0])
    flow_v.append(flow[:,:,1])

flow_u = np.array(flow_u)
flow_v = np.array(flow_v)
mean_flow = np.sqrt(flow_u**2 + flow_v**2).mean()
print(f"   Mean flow magnitude: {mean_flow:.3f} px/frame")

# Compute time derivative
print("\n4. Computing time derivative...")
drho_dt = np.gradient(images_ds, axis=0)

# Compute spatial gradients
print("\n5. Computing spatial gradients...")
drho_dx = np.gradient(images_ds, axis=2)
drho_dy = np.gradient(images_ds, axis=1)

# Compute divergence of flow
du_dx = np.gradient(flow_u, axis=2)
dv_dy = np.gradient(flow_v, axis=1)
div_v = du_dx + dv_dy

# Compute transport terms
print("\n6. Computing transport terms...")
# Align dimensions (flow has n-1 frames)
rho = images_ds[:-1]
advection = flow_u * drho_dx[:-1] + flow_v * drho_dy[:-1]  # v·∇ρ
expansion = rho * div_v  # ρ∇·v
transport = advection + expansion  # ∇·(ρv)

# Second derivatives for diffusion
d2rho_dx2 = np.gradient(drho_dx, axis=2)
d2rho_dy2 = np.gradient(drho_dy, axis=1)
laplacian = d2rho_dx2 + d2rho_dy2

# Build simple library
print("\n7. Building library...")
target = drho_dt[:-1].flatten()

library_matrix = np.column_stack([
    np.ones_like(transport).flatten(),      # constant
    rho.flatten(),                          # ρ
    drho_dx[:-1].flatten(),                 # ∂ρ/∂x
    drho_dy[:-1].flatten(),                 # ∂ρ/∂y
    laplacian[:-1].flatten(),               # ∇²ρ
    transport.flatten(),                    # ∇·(ρv) - MAIN TRANSPORT
    advection.flatten(),                    # v·∇ρ
    expansion.flatten(),                    # ρ∇·v
])

term_names = ['constant', 'rho', 'drho_dx', 'drho_dy', 'laplacian', 
              'transport', 'advection', 'expansion']

# Remove NaN/Inf
valid = np.isfinite(library_matrix).all(axis=1) & np.isfinite(target)
X = library_matrix[valid]
y = target[valid]

print(f"   Design matrix: {X.shape}")
print(f"   Valid samples: {valid.sum()} / {len(valid)} ({100*valid.sum()/len(valid):.1f}%)")

# Simple least squares (no sklearn needed)
print("\n8. Solving least squares...")
coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

# Print results
print("\n" + "="*60)
print("DISCOVERED EQUATION:")
print("="*60)
print("∂ρ/∂t = ", end="")
for name, coef in zip(term_names, coeffs):
    if abs(coef) > 1e-10:
        sign = "+" if coef >= 0 else ""
        print(f"{sign}{coef:.6e}·{name} ", end="")
print("\n" + "="*60)

# Compute R²
y_pred = X @ coeffs
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"\nModel quality:")
print(f"  R² = {r2:.6f}")
print(f"  MSE = {np.mean((y - y_pred)**2):.6e}")
print(f"  Mean |residual| = {np.mean(np.abs(y - y_pred)):.6e}")

# Key physics check
print(f"\nPhysics check:")
print(f"  Transport coeff: {coeffs[5]:.6e} (should be ~-1 for continuity)")
print(f"  Advection coeff: {coeffs[6]:.6e}")
print(f"  Expansion coeff: {coeffs[7]:.6e}")
print(f"  Diffusion coeff: {coeffs[4]:.6e}")

print("\n✅ Quick test complete!")
print("\nNext: Run full analysis with regularization (transport_pde_discovery.py)")
