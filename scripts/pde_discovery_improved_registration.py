"""
PDE Discovery with Improved Multi-Method Registration
Compares Farnebäck, TV-L1, and generates presentation slides
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "slides"

# ==================== CONFIGURATION ====================
IMAGE_FOLDER = DATA_DIR
OUTPUT_FOLDER = OUTPUT_DIR
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

N_IMAGES = 51  # ALL IMAGES
dx, dy, dt = 0.1, 0.1, 1.0

print("="*80)
print("PDE DISCOVERY WITH ALL 51 IMAGES - 2ND ORDER DERIVATIVES")
print("="*80)

# ==================== 1. LOAD IMAGES ====================
print("\n1. Loading images...")
image_files = sorted(IMAGE_FOLDER.glob("*.tif"))[:N_IMAGES]
U_raw = []

for img_file in image_files:
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        U_raw.append(img)

U_raw = np.array(U_raw, dtype=np.float32)
T, H, W = U_raw.shape
print(f"   Loaded {T} images: {H}×{W} pixels")

# Downsample for memory efficiency (optical flow on large images is memory-intensive)
print(f"   Downsampling by factor 2 for memory efficiency...")
U_raw = np.array([cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_AREA) for img in U_raw])
T, H, W = U_raw.shape
print(f"   Working resolution: {H}×{W} pixels")

# ==================== 2. NORMALIZE ====================
print("\n2. Denoising and normalizing...")
U_denoised = np.array([gaussian_filter(img, sigma=1.0) for img in U_raw])
U_norm = (U_denoised - U_denoised.min()) / (U_denoised.max() - U_denoised.min())

# ==================== 3. REGISTRATION METHODS ====================

def register_farneback_improved(images):
    """
    Improved Farnebäck optical flow with recommended parameters for SEM images.
    """
    print("\n3A. Farnebäck Registration (Improved Parameters)...")
    registered = [images[0].copy()]
    flows = []
    
    # Recommended parameters for textured SEM patterns
    params = dict(
        pyr_scale=0.5,
        levels=5,           # deeper pyramid for large motions
        winsize=25,         # larger window for SEM texture
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    
    for i in range(1, len(images)):
        ref = (registered[-1] * 255).astype(np.uint8)
        mov = (images[i] * 255).astype(np.uint8)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(ref, mov, None, **params)
        
        # CRITICAL: Smooth the flow field for stability
        flow_smoothed = cv2.GaussianBlur(flow, (11, 11), 2.0)
        flows.append(flow_smoothed)
        
        # Warp with subpixel interpolation
        h, w = images[i].shape
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w) - flow_smoothed[:, :, 0]
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis] - flow_smoothed[:, :, 1]
        
        warped = cv2.remap(images[i], flow_map, None, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        registered.append(warped)
        
        if (i+1) % 10 == 0:
            print(f"   Registered frame {i+1}/{len(images)}")
    
    return np.array(registered), flows


def register_dis_improved(images):
    """
    DIS (Dense Inverse Search) optical flow - robust alternative to TV-L1.
    Available in standard OpenCV, excellent for noisy SEM images.
    """
    print("\n3B. DIS Optical Flow Registration (Robust for SEM)...")
    registered = [images[0].copy()]
    flows = []
    
    # Create DIS optical flow object (similar quality to TV-L1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # PRESET_MEDIUM balances speed and accuracy
    
    for i in range(1, len(images)):
        ref = (registered[-1] * 255).astype(np.uint8)
        mov = (images[i] * 255).astype(np.uint8)
        
        # Compute DIS flow
        flow = dis.calc(ref, mov, None)
        
        # Smooth flow field for stability
        flow_smoothed = cv2.GaussianBlur(flow, (11, 11), 2.0)
        flows.append(flow_smoothed)
        
        # Warp
        h, w = images[i].shape
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        flow_map[:, :, 0] = np.arange(w) - flow_smoothed[:, :, 0]
        flow_map[:, :, 1] = np.arange(h)[:, np.newaxis] - flow_smoothed[:, :, 1]
        
        warped = cv2.remap(images[i], flow_map, None,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        registered.append(warped)
        
        if (i+1) % 10 == 0:
            print(f"   Registered frame {i+1}/{len(images)}")
    
    return np.array(registered), flows


def compute_registration_metrics(images_before, images_after, flows):
    """Compute quantitative registration quality metrics."""
    # Average flow magnitude
    flow_mags_before = []
    flow_mags_after = []
    
    for i in range(len(images_after)-1):
        # Before registration
        ref_b = (images_before[i] * 255).astype(np.uint8)
        mov_b = (images_before[i+1] * 255).astype(np.uint8)
        flow_b = cv2.calcOpticalFlowFarneback(ref_b, mov_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mags_before.append(np.sqrt(flow_b[:,:,0]**2 + flow_b[:,:,1]**2).mean())
        
        # After registration
        ref_a = (images_after[i] * 255).astype(np.uint8)
        mov_a = (images_after[i+1] * 255).astype(np.uint8)
        flow_a = cv2.calcOpticalFlowFarneback(ref_a, mov_a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mags_after.append(np.sqrt(flow_a[:,:,0]**2 + flow_a[:,:,1]**2).mean())
    
    avg_before = np.mean(flow_mags_before)
    avg_after = np.mean(flow_mags_after)
    improvement = (avg_before - avg_after) / avg_before * 100
    
    return avg_before, avg_after, improvement


# ==================== 4. PERFORM REGISTRATIONS ====================
U_farneback, flows_farneback = register_farneback_improved(U_norm)
U_dis, flows_dis = register_dis_improved(U_norm)

# Compute metrics
print("\n4. Computing registration quality metrics...")
fb_before, fb_after, fb_improve = compute_registration_metrics(U_norm, U_farneback, flows_farneback)
dis_before, dis_after, dis_improve = compute_registration_metrics(U_norm, U_dis, flows_dis)

print(f"\n   Farnebäck: {fb_before:.2f} → {fb_after:.2f} px ({fb_improve:.1f}% improvement)")
print(f"   DIS Flow:  {dis_before:.2f} → {dis_after:.2f} px ({dis_improve:.1f}% improvement)")

# ==================== 5. CREATE COMPARISON SLIDE 1: REGISTRATION QUALITY ====================
print("\n5. Creating SLIDE 1: Registration Quality Comparison...")

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)

# Sample frames
frame_idx = 15
next_idx = 16

# Row 1: Farnebäck
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(U_norm[frame_idx], cmap='gray')
ax1.set_title(f'Unregistered\nFrame {frame_idx}', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(U_farneback[frame_idx], cmap='gray')
ax2.set_title(f'Farnebäck Registered\nFrame {frame_idx}', fontsize=14, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
diff_before_fb = np.abs(U_norm[next_idx] - U_norm[frame_idx])
im3 = ax3.imshow(diff_before_fb, cmap='hot', vmin=0, vmax=0.3)
ax3.set_title(f'Difference (Unregistered)\nFrames {frame_idx}→{next_idx}', fontsize=14, fontweight='bold')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, fraction=0.046)

ax4 = fig.add_subplot(gs[0, 3])
diff_after_fb = np.abs(U_farneback[next_idx] - U_farneback[frame_idx])
im4 = ax4.imshow(diff_after_fb, cmap='hot', vmin=0, vmax=0.3)
ax4.set_title(f'Difference (Farnebäck)\nFrames {frame_idx}→{next_idx}', fontsize=14, fontweight='bold')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, fraction=0.046)

# Row 2: TV-L1
ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(U_norm[frame_idx], cmap='gray')
ax5.set_title(f'Unregistered\nFrame {frame_idx}', fontsize=14, fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(U_dis[frame_idx], cmap='gray')
ax6.set_title(f'DIS Registered\nFrame {frame_idx}', fontsize=14, fontweight='bold')
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
diff_before_tv = np.abs(U_norm[next_idx] - U_norm[frame_idx])
im7 = ax7.imshow(diff_before_tv, cmap='hot', vmin=0, vmax=0.3)
ax7.set_title(f'Difference (Unregistered)\nFrames {frame_idx}→{next_idx}', fontsize=14, fontweight='bold')
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046)

ax8 = fig.add_subplot(gs[1, 3])
diff_after_dis = np.abs(U_dis[next_idx] - U_dis[frame_idx])
im8 = ax8.imshow(diff_after_dis, cmap='hot', vmin=0, vmax=0.3)
ax8.set_title(f'Difference (DIS)\nFrames {frame_idx}→{next_idx}', fontsize=14, fontweight='bold')
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046)

# Add metrics text
fig.text(0.5, 0.95, 'SLIDE 1: Registration Quality Comparison', 
         ha='center', fontsize=18, fontweight='bold')
fig.text(0.5, 0.52, f'Farnebäck: {fb_before:.2f}→{fb_after:.2f} px ({fb_improve:.1f}% improvement)', 
         ha='center', fontsize=13, fontweight='bold', color='darkblue')
fig.text(0.5, 0.02, f'DIS Flow: {dis_before:.2f}→{dis_after:.2f} px ({dis_improve:.1f}% improvement)', 
         ha='center', fontsize=13, fontweight='bold', color='darkgreen')

plt.savefig(OUTPUT_FOLDER / 'SLIDE1_Registration_Quality_51images.png', dpi=300, bbox_inches='tight')
print("   ✅ SLIDE 1 saved")
plt.close()

# ==================== 6. CREATE COMPARISON SLIDE 2: FLOW FIELDS & METRICS ====================
print("\n6. Creating SLIDE 2: Flow Fields and Quantitative Metrics...")

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Row 1: Farnebäck flow analysis
ax1 = fig.add_subplot(gs[0, 0])
flow_fb = flows_farneback[frame_idx]
mag_fb = np.sqrt(flow_fb[:,:,0]**2 + flow_fb[:,:,1]**2)
im1 = ax1.imshow(mag_fb, cmap='jet')
ax1.set_title('Farnebäck Flow Magnitude\n(Smoothed, σ=2.0)', fontsize=13, fontweight='bold')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='Pixels', fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
# Quiver plot (subsample for visibility)
step = 50
Y, X = np.mgrid[0:H:step, 0:W:step]
U_flow = flow_fb[::step, ::step, 0]
V_flow = flow_fb[::step, ::step, 1]
ax2.imshow(U_norm[frame_idx], cmap='gray', alpha=0.6)
ax2.quiver(X, Y, U_flow, V_flow, color='cyan', scale=200, width=0.003)
ax2.set_title('Farnebäck Flow Vectors\n(winsize=25, levels=5)', fontsize=13, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
# Flow magnitude histogram
ax3.hist(mag_fb.ravel(), bins=100, alpha=0.7, color='blue', edgecolor='black')
ax3.axvline(mag_fb.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {mag_fb.mean():.2f} px')
ax3.set_xlabel('Flow Magnitude (pixels)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Farnebäck Flow Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Row 2: DIS flow analysis
ax4 = fig.add_subplot(gs[1, 0])
flow_dis = flows_dis[frame_idx]
mag_dis = np.sqrt(flow_dis[:,:,0]**2 + flow_dis[:,:,1]**2)
im4 = ax4.imshow(mag_dis, cmap='jet')
ax4.set_title('DIS Flow Magnitude\n(PRESET_MEDIUM)', fontsize=13, fontweight='bold')
ax4.axis('off')
plt.colorbar(im4, ax=ax4, label='Pixels', fraction=0.046)

ax5 = fig.add_subplot(gs[1, 1])
U_flow_dis = flow_dis[::step, ::step, 0]
V_flow_dis = flow_dis[::step, ::step, 1]
ax5.imshow(U_norm[frame_idx], cmap='gray', alpha=0.6)
ax5.quiver(X, Y, U_flow_dis, V_flow_dis, color='lime', scale=200, width=0.003)
ax5.set_title('DIS Flow Vectors\n(Dense Inverse Search)', fontsize=13, fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(mag_dis.ravel(), bins=100, alpha=0.7, color='green', edgecolor='black')
ax6.axvline(mag_dis.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {mag_dis.mean():.2f} px')
ax6.set_xlabel('Flow Magnitude (pixels)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('DIS Flow Distribution', fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

fig.text(0.5, 0.96, 'SLIDE 2: Flow Field Analysis and Parameters', 
         ha='center', fontsize=18, fontweight='bold')

# Add parameter boxes
param_fb = """Farnebäck Parameters:
• pyr_scale = 0.5
• levels = 5
• winsize = 25
• poly_n = 7
• poly_sigma = 1.5
• Flow smoothing: σ=2.0"""

param_dis = """DIS Parameters:
• Preset: MEDIUM
• Patch size: 8
• Patch stride: 4
• Gradient descent: 25 iters
• Variational refinement
• Flow smoothing: σ=2.0"""

fig.text(0.25, 0.02, param_fb, ha='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
fig.text(0.75, 0.02, param_dis, ha='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.savefig(OUTPUT_FOLDER / 'SLIDE2_Flow_Fields_51images.png', dpi=300, bbox_inches='tight')
print("   ✅ SLIDE 2 saved")
plt.close()

# ==================== 7. SELECT BEST METHOD AND PROCEED WITH SINDY ====================
print("\n7. Selecting best registration method...")

if dis_improve > fb_improve:
    U_registered = U_dis
    method_name = "DIS"
    improvement = dis_improve
else:
    U_registered = U_farneback
    method_name = "Farnebäck"
    improvement = fb_improve

print(f"   Selected: {method_name} ({improvement:.1f}% improvement)")

# ==================== 8. SINDY PDE DISCOVERY ====================
print("\n8. Performing SINDy PDE Discovery...")

# Temporal smoothing
print("   Applying Savitzky-Golay smoothing...")
U_smooth = np.array([savgol_filter(U_registered[:, i, j], window_length=7, polyorder=3, axis=0)
                     for i in range(H) for j in range(W)]).reshape(T, H, W)

# Compute derivatives
print("   Computing 2nd order derivatives...")
skip = 25
subsample = 12
U_crop = U_smooth[:, skip:-skip:subsample, skip:-skip:subsample]

def compute_2nd_order_derivatives(U, dx, dy, dt):
    """2nd order central differences - faster than 4th order."""
    # Spatial
    u_x = (U[:, :, 2:] - U[:, :, :-2]) / (2*dx)
    u_y = (U[:, 2:, :] - U[:, :-2, :]) / (2*dy)
    
    u_xx = (U[:, :, 2:] - 2*U[:, :, 1:-1] + U[:, :, :-2]) / (dx**2)
    u_yy = (U[:, 2:, :] - 2*U[:, 1:-1, :] + U[:, :-2, :]) / (dy**2)
    
    # Temporal (central difference)
    u_t = (U[2:, :, :] - U[:-2, :, :]) / (2*dt)
    
    return u_x, u_y, u_xx, u_yy, u_t

u_x, u_y, u_xx, u_yy, u_t = compute_2nd_order_derivatives(U_crop, dx, dy, dt)

# Align arrays
min_t = min(u_x.shape[0], u_y.shape[0], u_xx.shape[0], u_yy.shape[0], u_t.shape[0])
min_h = min(u_x.shape[1], u_y.shape[1], u_xx.shape[1], u_yy.shape[1], u_t.shape[1])
min_w = min(u_x.shape[2], u_y.shape[2], u_xx.shape[2], u_yy.shape[2], u_t.shape[2])

u_crop_aligned = U_crop[:min_t, :min_h, :min_w]
u_x = u_x[:min_t, :min_h, :min_w]
u_y = u_y[:min_t, :min_h, :min_w]
u_xx = u_xx[:min_t, :min_h, :min_w]
u_yy = u_yy[:min_t, :min_h, :min_w]
u_t = u_t[:min_t, :min_h, :min_w]

laplacian = u_xx + u_yy

# Build library
print("   Building SINDy library...")
library_terms = [
    np.ones_like(u_crop_aligned),
    u_crop_aligned,
    u_x, u_y,
    u_xx, u_yy, laplacian,
    u_crop_aligned**2,
    u_crop_aligned * u_x,
    u_crop_aligned * u_y,
    u_crop_aligned**3,
    u_x**2,
    u_y**2
]

term_names = ['1', 'u', 'u_x', 'u_y', 'u_xx', 'u_yy', '∇²u', 'u²', 'u·u_x', 'u·u_y', 'u³', 'u_x²', 'u_y²']

X = np.column_stack([term.ravel() for term in library_terms])
y = u_t.ravel()

# STRidge
print("   Performing STRidge regression...")
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
    
    coeffs_unscaled = coeffs / scaler.scale_
    return coeffs_unscaled, scaler

coeffs, scaler = stridge(X, y, alpha=0.01, threshold=1e-5)

# Evaluate
y_pred = X @ coeffs
r2 = r2_score(y, y_pred)

print(f"\n   Model R² = {r2:.6f}")
print("\n   Discovered PDE:")
equation_parts = []
for coeff, name in zip(coeffs, term_names):
    if np.abs(coeff) > 1e-5:
        sign = '+' if coeff > 0 and len(equation_parts) > 0 else ''
        equation_parts.append(f"{sign}{coeff:.6f}·{name}")
        print(f"      {name}: {coeff:.6f}")

equation = "u_t = " + " ".join(equation_parts) if equation_parts else "u_t = 0"
print(f"\n   {equation}")

# ==================== 9. CREATE SLIDE 3: PDE RESULTS ====================
print("\n9. Creating SLIDE 3: PDE Discovery Results...")

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

# Coefficient bar chart
ax1 = fig.add_subplot(gs[0, :])
colors = ['red' if abs(c) > 1e-5 else 'lightgray' for c in coeffs]
bars = ax1.bar(range(len(coeffs)), coeffs, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(term_names)))
ax1.set_xticklabels(term_names, fontsize=12, fontweight='bold')
ax1.set_ylabel('Coefficient Value', fontsize=13, fontweight='bold')
ax1.set_title(f'Discovered PDE Coefficients (R² = {r2:.6f})', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Prediction vs actual
ax2 = fig.add_subplot(gs[1, 0])
sample_indices = np.random.choice(len(y), size=min(10000, len(y)), replace=False)
ax2.scatter(y[sample_indices], y_pred[sample_indices], alpha=0.3, s=1)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect fit')
ax2.set_xlabel('Actual u_t', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted u_t', fontsize=12, fontweight='bold')
ax2.set_title('Model Predictions', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Residuals
ax3 = fig.add_subplot(gs[1, 1])
residuals = y - y_pred
ax3.hist(residuals, bins=100, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Mean = {residuals.mean():.2e}')
ax3.set_xlabel('Residual (u_t actual - predicted)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Equation text
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')
equation_text = f"Discovered PDE:\n\n{equation}\n\n"
equation_text += f"Model Performance:\n"
equation_text += f"• R² Score: {r2:.6f}\n"
equation_text += f"• Active Terms: {np.sum(np.abs(coeffs) > 1e-5)}/{len(coeffs)}\n"
equation_text += f"• Registration: {method_name}\n"
equation_text += f"• Improvement: {improvement:.1f}%\n"
equation_text += f"• Total Frames: {T}\n"
equation_text += f"• Spatial Points: {min_h}×{min_w}"

ax4.text(0.5, 0.5, equation_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='center', horizontalalignment='center',
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.text(0.5, 0.96, 'SLIDE 3: PDE Discovery Results (51 Images, 2nd Order)', 
         ha='center', fontsize=18, fontweight='bold')

plt.savefig(OUTPUT_FOLDER / 'SLIDE3_PDE_Results_51images.png', dpi=300, bbox_inches='tight')
print("   ✅ SLIDE 3 saved")
plt.close()

# ==================== 10. CREATE SLIDE 4: SPATIOTEMPORAL ====================
print("\n10. Creating SLIDE 4: Spatiotemporal Evolution...")

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

# Show temporal evolution at different frames
frame_indices = [0, 10, 20, 30, 40, 49]
for idx, frame_i in enumerate(frame_indices[:4]):
    ax = fig.add_subplot(gs[0, idx])
    ax.imshow(U_registered[frame_i], cmap='viridis')
    ax.set_title(f'Frame {frame_i}', fontsize=12, fontweight='bold')
    ax.axis('off')

# Show derivatives
ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(u_t[20], cmap='RdBu_r', vmin=-0.1, vmax=0.1)
ax5.set_title('Temporal Derivative (u_t)\nFrame 20', fontsize=12, fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(laplacian[20], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax6.set_title('Laplacian (∇²u)\nFrame 20', fontsize=12, fontweight='bold')
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
ax7.imshow(u_x[20], cmap='RdBu_r')
ax7.set_title('Spatial Gradient (u_x)\nFrame 20', fontsize=12, fontweight='bold')
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
ax8.imshow(u_y[20], cmap='RdBu_r')
ax8.set_title('Spatial Gradient (u_y)\nFrame 20', fontsize=12, fontweight='bold')
ax8.axis('off')

# Time series of mean intensity
ax9 = fig.add_subplot(gs[2, :2])
mean_intensity = [U_registered[i].mean() for i in range(T)]
ax9.plot(mean_intensity, linewidth=2, color='blue')
ax9.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
ax9.set_ylabel('Mean Intensity', fontsize=12, fontweight='bold')
ax9.set_title('Mean Intensity Evolution (51 Frames)', fontsize=13, fontweight='bold')
ax9.grid(True, alpha=0.3)

# Variance over time
ax10 = fig.add_subplot(gs[2, 2:])
var_intensity = [U_registered[i].var() for i in range(T)]
ax10.plot(var_intensity, linewidth=2, color='red')
ax10.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
ax10.set_ylabel('Intensity Variance', fontsize=12, fontweight='bold')
ax10.set_title('Variance Evolution (51 Frames)', fontsize=13, fontweight='bold')
ax10.grid(True, alpha=0.3)

fig.text(0.5, 0.96, 'SLIDE 4: Spatiotemporal Evolution and Derivatives', 
         ha='center', fontsize=18, fontweight='bold')

plt.savefig(OUTPUT_FOLDER / 'SLIDE4_Spatiotemporal_51images.png', dpi=300, bbox_inches='tight')
print("   ✅ SLIDE 4 saved")
plt.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE - ALL 51 IMAGES!")
print("="*80)
print(f"\nGenerated files in '{OUTPUT_FOLDER}':")
print(f"  1. SLIDE1_Registration_Quality_51images.png")
print(f"     - Registration comparison (Farnebäck vs DIS)")
print(f"  2. SLIDE2_Flow_Fields_51images.png")
print(f"     - Flow field analysis and parameters")
print(f"  3. SLIDE3_PDE_Results_51images.png")
print(f"     - PDE coefficients and model performance")
print(f"  4. SLIDE4_Spatiotemporal_51images.png")
print(f"     - Temporal evolution and derivatives")
print(f"\nFinal Results:")
print(f"  • Registration Method: {method_name}")
print(f"  • Registration Improvement: {improvement:.1f}%")
print(f"  • Model R²: {r2:.6f}")
print(f"  • Discovered PDE: {equation}")
print(f"  • Active Terms: {np.sum(np.abs(coeffs) > 1e-5)}/{len(coeffs)}")
print("="*80)
