"""
Generate visualization figures for presentation slides
Creates 4-5 key plots showing main findings
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "presentation_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style for presentation (clean, minimal text)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def _load_best_model() -> dict | None:
    p = PROJECT_ROOT / "outputs" / "latest" / "slides" / "best_model.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_patch_pde_metrics() -> dict | None:
    p = PROJECT_ROOT / "outputs" / "latest" / "patch_pde" / "PATCH_PDE_REPORT.txt"
    if not p.exists():
        return None
    metrics = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("R²="):
                metrics["r2"] = float(line.split("=")[1])
            if line.startswith("RMSE="):
                metrics["rmse"] = float(line.split("=")[1])
            if line.startswith("nRMSE="):
                metrics["nrmse"] = float(line.split("=")[1])
            if line.startswith("corr="):
                metrics["corr"] = float(line.split("=")[1])
            if "one-step RMSE" in line and "=" in line:
                metrics["one_step_rmse"] = float(line.split("=")[1])
        return metrics if metrics else None
    except Exception:
        return None

def load_sample_images(n=5):
    """Load sample images for visualization"""
    image_dir = DATA_DIR
    tif_files = sorted(image_dir.glob('*.tif'))[:n]
    
    images = []
    for tif_file in tif_files:
        img = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.astype(np.float64))
    
    return np.array(images)

def main():
    """Generate all presentation figures"""
    print("\n" + "="*60)
    print("GENERATING PRESENTATION FIGURES")
    print("="*60 + "\n")
    
    # Load images once
    print("Loading images...")
    images_all = load_sample_images(n=5)
    print(f"Loaded {len(images_all)} images\n")
    
    # Create figures (pass images to avoid reloading)
    figure1_data_overview_cached(images_all)
    figure2_motion_comparison_cached(images_all)
    figure3_velocity_field_cached(images_all)
    figure4_method_comparison()
    figure5_pde_coefficients()
    figure6_physics_schematic()
    
    print("\n" + "="*60)
    print("✅ ALL FIGURES GENERATED")
    print("="*60)
    print("\nFiles created:")
    print(f"  1. {OUTPUT_DIR / 'fig1_data_overview.png'}       - Sample frames")
    print(f"  2. {OUTPUT_DIR / 'fig2_motion_comparison.png'}   - Camera vs material motion")
    print(f"  3. {OUTPUT_DIR / 'fig3_velocity_field.png'}      - Velocity field quiver")
    print(f"  4. {OUTPUT_DIR / 'fig4_method_comparison.png'}   - SINDy vs Transport")
    print(f"  5. {OUTPUT_DIR / 'fig5_pde_coefficients.png'}    - Discovered coefficients")
    print(f"  6. {OUTPUT_DIR / 'fig6_physics_schematic.png'}   - Physics interpretation")
    print("\nUse these in your presentation slides!")

def figure1_data_overview_cached(images):
    """Figure 1: Sample images showing temporal evolution"""
    print("Creating Figure 1: Data Overview...")
    
    images_norm = images / images.max()
    
    fig = plt.figure(figsize=(16, 4))
    frames = [0, 1, 2, 3, 4]
    
    for i, frame_idx in enumerate(frames):
        if frame_idx < len(images):
            ax = fig.add_subplot(1, 5, i+1)
            im = ax.imshow(images_norm[i], cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Frame {frame_idx*10}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.04, label='Normalized Intensity')
    out_path = OUTPUT_DIR / 'fig1_data_overview.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()

def figure2_motion_comparison_cached(images):
    """Figure 2: Camera motion vs Material motion"""
    print("Creating Figure 2: Motion Analysis...")
    
    h, w = images[0].shape
    
    # Downsample for optical flow
    img1 = cv2.resize(images[0], (w//4, h//4))
    img2 = cv2.resize(images[1], (w//4, h//4))
    
    img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        img1_norm, img2_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    mean_flow = np.array([flow[:, :, 0].mean(), flow[:, :, 1].mean()])
    cam_mag = float(np.sqrt((mean_flow**2).sum()))
    resid = flow - mean_flow.reshape(1, 1, 2)
    resid_mag = float(np.sqrt(resid[:, :, 0] ** 2 + resid[:, :, 1] ** 2).mean())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Frame 1
    axes[0].imshow(img1, cmap='hot')
    axes[0].set_title('Frame t')
    axes[0].axis('off')
    
    # Frame 2
    axes[1].imshow(img2, cmap='hot')
    axes[1].set_title('Frame t+1')
    axes[1].axis('off')
    
    # Optical flow magnitude
    im = axes[2].imshow(magnitude, cmap='viridis')
    axes[2].set_title('Material Motion (Optical Flow)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='|v| (pixels/frame)')
    
    # Small annotation (computed, not hard-coded)
    fig.text(
        0.5,
        0.02,
        f"Mean flow (camera-like): {cam_mag:.3f} px/frame   |   Residual (material-like): {resid_mag:.3f} px/frame",
        ha="center",
        fontsize=12,
        color="#333333",
    )
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'fig2_motion_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()

def figure3_velocity_field_cached(images):
    """Figure 3: Velocity field quiver plot"""
    print("Creating Figure 3: Velocity Field...")
    
    h, w = images[0].shape
    
    # Downsample for optical flow
    img1 = cv2.resize(images[0], (w//8, h//8))
    img2 = cv2.resize(images[1], (w//8, h//8))
    
    img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        img1_norm, img2_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Subsample for quiver plot
    step = 15
    y, x = np.mgrid[0:flow.shape[0]:step, 0:flow.shape[1]:step]
    u = flow[::step, ::step, 0]
    v = flow[::step, ::step, 1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img1, cmap='gray', alpha=0.7)
    
    # Quiver plot
    Q = ax.quiver(x, y, u, v, np.sqrt(u**2 + v**2),
                  cmap='jet', alpha=0.8, scale=50, width=0.003)
    
    ax.set_title('Velocity Field v(x,y,t)', fontsize=20, weight='bold')
    ax.axis('off')
    
    plt.colorbar(Q, ax=ax, fraction=0.046, pad=0.04, label='Speed (pixels/frame)')
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'fig3_velocity_field.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()

def figure1_data_overview():
    """Figure 1: Sample images showing temporal evolution"""
    images = load_sample_images(n=5)
    figure1_data_overview_cached(images)

def figure2_motion_comparison():
    """Figure 2: Camera motion vs Material motion"""
    images = load_sample_images(n=3)
    figure2_motion_comparison_cached(images)

def figure3_velocity_field():
    """Figure 3: Velocity field quiver plot"""
    images = load_sample_images(n=2)
    figure3_velocity_field_cached(images)

def figure4_method_comparison():
    """Figure 4: Comparison of methods (metrics)"""
    print("Creating Figure 4: Method Comparison...")
    best = _load_best_model()
    patch = _load_patch_pde_metrics()

    labels = []
    r2_vals = []
    step_vals = []

    if best is not None:
        labels.append("Global best\n(STRidge)")
        r2_vals.append(float(best.get("r2", np.nan)))
        step_vals.append(float(best.get("one_step_rmse", np.nan)))
    if patch is not None:
        labels.append("Patch robust\n(agg)")
        r2_vals.append(float(patch.get("r2", np.nan)))
        step_vals.append(float(patch.get("one_step_rmse", np.nan)))

    if not labels:
        labels = ["(run analyze_results.py)", "(run patch_based_pde_discovery.py)"]
        r2_vals = [0.0, 0.0]
        step_vals = [0.0, 0.0]

    colors = ["#2a6fdb", "#2aa84a"][: len(labels)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars1 = ax1.bar(labels, r2_vals, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("R²")
    ax1.set_title("Model fit (R²)", fontweight="bold")
    ax1.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars1, r2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=12)

    bars2 = ax2.bar(labels, step_vals, color=colors, edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("One-step RMSE")
    ax2.set_title("Predictive sanity check", fontweight="bold")
    ax2.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars2, step_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=12)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'fig4_method_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()

def figure5_pde_coefficients():
    """Figure 5: Discovered PDE coefficients (best model)"""
    print("Creating Figure 5: PDE Coefficients...")
    best = _load_best_model()
    if best is None:
        print("   ⚠️  best_model.json not found; run analyze_results.py first")
        return

    terms = best.get("terms", [])
    coeffs = best.get("coeffs", [])
    if not terms or not coeffs:
        print("   ⚠️  best_model.json missing coefficients")
        return

    # Drop constant term for readability if present
    pairs = [(t, float(c)) for t, c in zip(terms, coeffs) if t != "1"]
    terms2 = [p[0] for p in pairs]
    coeffs2 = [p[1] for p in pairs]
    colors = ["#2aa84a" if c > 0 else "#d64545" for c in coeffs2]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(terms2))
    bars = ax.barh(y, coeffs2, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(terms2)
    ax.set_xlabel("Coefficient")
    ax.set_title("Discovered PDE coefficients (best model)", fontweight="bold")
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.25, axis="x")

    for bar, val in zip(bars, coeffs2):
        x = bar.get_width()
        ax.text(x, bar.get_y() + bar.get_height() / 2.0, f"{val:.3g}", va="center", ha="left" if x >= 0 else "right", fontsize=12)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'fig5_pde_coefficients.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()

def figure6_physics_schematic():
    """Figure 6: Physics interpretation schematic"""
    print("Creating Figure 6: Physics Schematic...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'PDE terms (schematic)', fontsize=22, weight='bold', ha='center')
    
    # Central field
    circle = plt.Circle((5, 5), 1.5, color='#f0ad4e', alpha=0.7, linewidth=2, edgecolor='#8a5a00')
    ax.add_patch(circle)
    ax.text(5, 5, 'Field\n(u)', fontsize=18, ha='center', va='center', weight='bold')
    
    # Drift/advection arrows
    arrow_props = dict(arrowstyle='->', lw=3, color='#d64545')
    ax.annotate('', xy=(7.5, 6), xytext=(6.5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 4), xytext=(6.5, 4.5), arrowprops=arrow_props)
    ax.text(8.4, 5, 'Drift', fontsize=14, ha='center', weight='bold', color='#d64545')
    
    # Diffusion (spreading)
    arrow_props2 = dict(arrowstyle='<->', lw=2, color='#2a6fdb', linestyle='--')
    ax.annotate('', xy=(5, 7), xytext=(5, 6.5), arrowprops=arrow_props2)
    ax.annotate('', xy=(5, 3), xytext=(5, 3.5), arrowprops=arrow_props2)
    ax.annotate('', xy=(3, 5), xytext=(3.5, 5), arrowprops=arrow_props2)
    ax.text(2, 5, 'Diffusion', fontsize=14, ha='center', weight='bold', color='#2a6fdb')
    
    # Growth / reaction (upward)
    arrow_props3 = dict(arrowstyle='->', lw=4, color='#2aa84a')
    ax.annotate('', xy=(5, 8.5), xytext=(5, 6.5), arrowprops=arrow_props3)
    ax.text(5, 8.8, 'Reaction', fontsize=14, ha='center', weight='bold', color='#2aa84a')
    
    # Saturation / nonlinearity (downward)
    arrow_props4 = dict(arrowstyle='->', lw=3, color='#555555')
    ax.annotate('', xy=(5, 1.5), xytext=(5, 3.5), arrowprops=arrow_props4)
    ax.text(5, 1.0, 'Nonlinearity', fontsize=14, ha='center', weight='bold', color='#555555')
    
    # Equation at bottom (from best model if available)
    best = _load_best_model()
    eq_text = best.get("equation") if best is not None else "u_t = ..."
    ax.text(5, 0.3, eq_text, fontsize=14, ha='center', bbox=dict(boxstyle='round', facecolor='#e8f1ff', alpha=0.9))
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'fig6_physics_schematic.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {out_path}")
    plt.close()



if __name__ == "__main__":
    main()
