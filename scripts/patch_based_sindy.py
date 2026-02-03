"""
Patch-Based SINDy with Local Registration
==========================================
This approach:
1. Divides images into overlapping patches
2. Registers each patch independently (more robust to local deformations)
3. Performs PDE discovery on each patch separately
4. Aggregates results across patches for robust coefficient estimation

This handles non-rigid deformations and local misalignments better than global registration.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import glob
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import defaultdict
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Real-Images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latest" / "patch_sindy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PatchBasedSINDy:
    """
    SINDy with patch-level registration for handling local misalignments
    """
    
    def __init__(self, dt=1.0, dx=1.0, dy=1.0, patch_size=256, overlap=64):
        """
        Parameters:
        -----------
        dt : float
            Time step
        dx, dy : float
            Spatial resolution
        patch_size : int
            Size of each patch (pixels)
        overlap : int
            Overlap between adjacent patches (pixels)
        """
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        self.images = []
        self.patches_per_frame = []
        
    def load_images(self, folder, pattern="*.tif", max_images=None):
        """Load images"""
        print("="*70)
        print("PATCH-BASED SINDy - LOADING IMAGES")
        print("="*70)
        
        files = sorted(glob.glob(str(Path(folder) / pattern)))
        if max_images:
            files = files[:max_images]
        
        print(f"Loading {len(files)} images...")
        
        for i, f in enumerate(files):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load {f}")
            # Normalize to [0, 1]
            img_norm = img.astype(np.float64) / 255.0
            self.images.append(img_norm)
            
            if (i+1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(files)}")
        
        print(f"✓ Loaded {len(self.images)} images of shape {self.images[0].shape}")
        return self.images
    
    def preprocess_images(self, spatial_sigma=0.5, temporal_window=3):
        """
        Pre-denoise images before patch extraction
        Minimal smoothing to preserve 90-95% of signal
        """
        print("\n" + "="*70)
        print("PRE-DENOISING (before patch extraction)")
        print("="*70)
        print(f"Spatial smoothing: Gaussian (sigma={spatial_sigma})")
        print(f"Temporal smoothing: Savitzky-Golay (window={temporal_window})")
        
        # Spatial smoothing
        print("Applying spatial smoothing...")
        images_spatial = [gaussian_filter(img, sigma=spatial_sigma) for img in self.images]
        
        # Temporal smoothing
        if len(images_spatial) >= temporal_window:
            from scipy.signal import savgol_filter
            print("Applying temporal smoothing...")
            image_stack = np.array(images_spatial)
            image_stack = savgol_filter(image_stack, temporal_window, 2, axis=0)
            self.images = [image_stack[i] for i in range(len(image_stack))]
        else:
            self.images = images_spatial
        
        # Show denoising effect
        orig_diff = np.mean([np.abs(self.images[i+1] - self.images[i]).mean() 
                            for i in range(len(self.images)-1)])
        print(f"Frame-to-frame difference after denoising: {orig_diff:.4f}")
        print(f"✓ Pre-denoising complete\n")
        
        return self.images
    
    def extract_patches(self, image):
        """
        Extract overlapping patches from an image
        
        Returns:
        --------
        patches : list of (patch, (y, x)) tuples
            Each patch with its top-left coordinate
        """
        h, w = image.shape
        patches = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append((patch.copy(), (y, x)))
        
        return patches
    
    def register_patch_sequence(self, patch_sequence, method='ecc'):
        """
        Register a sequence of patches (same spatial location across time)
        
        Parameters:
        -----------
        patch_sequence : list of patches
            Patches from consecutive frames at same location
        method : str
            'ecc' (Enhanced Correlation Coefficient) or 'optical_flow'
            
        Returns:
        --------
        registered_patches : list
            Aligned patches
        registration_quality : float
            Quality metric (higher = better alignment)
        """
        if len(patch_sequence) < 2:
            return patch_sequence, 1.0
        
        # Convert to uint8 for OpenCV
        patches_uint8 = [(p * 255).astype(np.uint8) for p in patch_sequence]
        
        registered = [patch_sequence[0]]  # First patch is reference
        quality_scores = []
        
        for i in range(1, len(patch_sequence)):
            prev_uint8 = (registered[-1] * 255).astype(np.uint8)
            curr_uint8 = patches_uint8[i]
            
            if method == 'ecc':
                # ECC registration (more robust for patches)
                warp_mode = cv2.MOTION_EUCLIDEAN
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
                
                try:
                    cc, warp_matrix = cv2.findTransformECC(
                        prev_uint8, curr_uint8, warp_matrix, warp_mode, criteria,
                        inputMask=None, gaussFiltSize=5
                    )
                    
                    # Apply transformation
                    aligned = cv2.warpAffine(
                        patch_sequence[i], warp_matrix,
                        (self.patch_size, self.patch_size),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                    
                    registered.append(aligned)
                    quality_scores.append(cc)
                    
                except cv2.error:
                    # ECC failed, use original
                    registered.append(patch_sequence[i])
                    quality_scores.append(0.0)
                    
            elif method == 'optical_flow':
                # Optical flow registration
                flow = cv2.calcOpticalFlowFarneback(
                    prev_uint8, curr_uint8, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=5, poly_n=7, poly_sigma=1.5, flags=0
                )
                
                # Create warping map
                h, w = patch_sequence[i].shape
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (map_x - flow[..., 0]).astype(np.float32)
                map_y = (map_y - flow[..., 1]).astype(np.float32)
                
                aligned = cv2.remap(
                    patch_sequence[i], map_x, map_y,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                
                registered.append(aligned)
                
                # Quality: inverse of flow magnitude
                flow_mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                quality_scores.append(1.0 / (flow_mag + 1.0))
        
        avg_quality = np.mean(quality_scores) if quality_scores else 1.0
        return registered, avg_quality
    
    def compute_derivatives(self, u):
        """Compute spatial derivatives"""
        # Use central differences
        ux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * self.dx)
        uy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * self.dy)
        uxx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / (self.dx**2)
        uyy = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / (self.dy**2)
        
        return ux, uy, uxx, uyy
    
    def compute_time_derivative(self, patches, idx):
        """Compute time derivative at frame idx"""
        if idx == 0:
            ut = (patches[1] - patches[0]) / self.dt
        elif idx == len(patches) - 1:
            ut = (patches[-1] - patches[-2]) / self.dt
        else:
            ut = (patches[idx+1] - patches[idx-1]) / (2 * self.dt)
        
        return ut
    
    def build_library(self, u, ux, uy, uxx, uyy):
        """Build candidate term library"""
        laplacian = uxx + uyy
        
        terms = [
            np.ones_like(u),      # constant
            u,                    # u
            ux,                   # u_x
            uy,                   # u_y
            uxx,                  # u_xx
            uyy,                  # u_yy
            laplacian,            # ∇²u
            u**2,                 # u²
            u * ux,               # u·u_x
            u * uy,               # u·u_y
            u * laplacian,        # u·∇²u
        ]
        
        term_names = [
            '1', 'u', 'u_x', 'u_y', 'u_xx', 'u_yy', '∇²u',
            'u²', 'u·u_x', 'u·u_y', 'u·∇²u'
        ]
        
        return np.column_stack(terms), term_names
    
    def discover_pde_for_patch(self, patch_sequence, skip_boundary=5, subsample=4,
                               alpha=0.01, registration_method='none'):
        """
        Discover PDE for a single patch location across time
        
        Returns:
        --------
        coeffs : ndarray or None
            Discovered coefficients (None if failed)
        quality : float
            Quality of discovery
        """
        # Apply smoothing instead of registration (camera is stable!)
        if registration_method == 'none':
            # No additional smoothing - already pre-denoised!
            registered_patches = patch_sequence
            reg_quality = 1.0
        else:
            # Register patches (for comparison)
            registered_patches, reg_quality = self.register_patch_sequence(
                patch_sequence, method=registration_method
            )
        
        # Build training data
        X_list = []
        y_list = []
        
        for i in range(1, len(registered_patches) - 1):
            u = registered_patches[i]
            
            # Time derivative
            ut = self.compute_time_derivative(registered_patches, i)
            
            # Spatial derivatives
            ux, uy, uxx, uyy = self.compute_derivatives(u)
            
            # Build library
            library, _ = self.build_library(u, ux, uy, uxx, uyy)
            
            # Mask boundaries
            h, w = u.shape
            mask = np.ones((h, w), dtype=bool)
            mask[:skip_boundary, :] = False
            mask[-skip_boundary:, :] = False
            mask[:, :skip_boundary] = False
            mask[:, -skip_boundary:] = False
            
            # Subsample
            if subsample > 1:
                submask = np.zeros_like(mask)
                submask[::subsample, ::subsample] = True
                mask = mask & submask
            
            idx = np.where(mask)
            
            # Flatten
            ut_flat = ut[idx]
            library_2d = library.reshape(h, w, -1)
            library_flat = library_2d[idx]
            
            X_list.append(library_flat)
            y_list.append(ut_flat)
        
        if not X_list:
            return None, 0.0
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Remove invalid
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        
        if len(y) < 100:  # Too few points
            return None, 0.0
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        try:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_scaled, y)
            coeffs = model.coef_ / scaler.scale_
            
            # Compute quality (R²)
            y_pred = X @ coeffs
            r2 = r2_score(y, y_pred)
            quality = max(0, r2) * reg_quality  # Combined quality
            
            return coeffs, quality
        except:
            return None, 0.0
    
    def discover_pde_patch_ensemble(self, alpha=0.01, min_patches=5,
                                   registration_method='none', max_patches=None):
        """
        Discover PDE using ensemble of patches
        
        Parameters:
        -----------
        alpha : float
            Regularization strength
        min_patches : int
            Minimum number of patches needed for reliable estimate
        registration_method : str
            'none' (just smoothing), 'ecc', or 'optical_flow'
        max_patches : int or None
            Maximum patches to process (for speed)
        """
        print("\n" + "="*70)
        print("PATCH-BASED PDE DISCOVERY")
        print("="*70)
        print(f"Patch size: {self.patch_size}×{self.patch_size}")
        print(f"Overlap: {self.overlap} pixels")
        print(f"Registration method: {registration_method}")
        print(f"Regularization: {alpha}")
        
        start_time = time.time()
        
        # Extract all patches
        print("\nExtracting patches from all frames...")
        all_frame_patches = []
        for img in self.images:
            patches = self.extract_patches(img)
            all_frame_patches.append(patches)
        
        n_patches = len(all_frame_patches[0])
        print(f"Total patches per frame: {n_patches}")
        
        if max_patches and n_patches > max_patches:
            # Randomly sample patches
            import random
            indices = random.sample(range(n_patches), max_patches)
            print(f"Randomly sampling {max_patches} patches for speed")
        else:
            indices = range(n_patches)
        
        # Process each patch location
        print(f"\nProcessing {len(indices)} patch locations...")
        
        patch_coeffs = []
        patch_qualities = []
        term_names = None
        
        for patch_idx in indices:
            # Get patch sequence across time
            patch_sequence = [frame_patches[patch_idx][0] 
                            for frame_patches in all_frame_patches]
            
            # Discover PDE for this patch
            coeffs, quality = self.discover_pde_for_patch(
                patch_sequence,
                alpha=alpha,
                registration_method=registration_method
            )
            
            if coeffs is not None and quality > -0.5:  # More lenient threshold (R² can be negative)
                patch_coeffs.append(coeffs)
                patch_qualities.append(quality)
                
                if term_names is None:
                    # Get term names (same for all patches)
                    u_dummy = np.zeros((10, 10))
                    ux, uy, uxx, uyy = self.compute_derivatives(u_dummy)
                    _, term_names = self.build_library(u_dummy, ux, uy, uxx, uyy)
            
            if (len(patch_coeffs)) % 10 == 0:
                print(f"  Processed {len(patch_coeffs)} valid patches...")
        
        if len(patch_coeffs) < min_patches:
            print(f"\n✗ Too few valid patches ({len(patch_coeffs)} < {min_patches})")
            return None, None, {}
        
        print(f"\n✓ Successfully processed {len(patch_coeffs)} patches")
        
        # Aggregate coefficients using weighted averaging
        patch_coeffs = np.array(patch_coeffs)
        patch_qualities = np.array(patch_qualities)
        
        # Normalize weights
        weights = patch_qualities / patch_qualities.sum()
        
        # Weighted average
        coeffs_ensemble = np.average(patch_coeffs, axis=0, weights=weights)
        
        # Coefficient uncertainty (weighted std)
        coeffs_std = np.sqrt(np.average((patch_coeffs - coeffs_ensemble)**2, 
                                       axis=0, weights=weights))
        
        # Zero out uncertain coefficients
        threshold = np.median(coeffs_std) * 2
        uncertain_mask = coeffs_std > threshold
        coeffs_ensemble[uncertain_mask] = 0
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("ENSEMBLE RESULTS")
        print("="*70)
        print(f"Valid patches: {len(patch_coeffs)}")
        print(f"Average quality: {patch_qualities.mean():.4f}")
        print(f"Quality std: {patch_qualities.std():.4f}")
        print(f"Time: {elapsed:.2f}s")
        
        print("\n" + "="*70)
        print("DISCOVERED PDE (Ensemble)")
        print("="*70)
        self.print_equation(coeffs_ensemble, term_names, coeffs_std)
        
        metrics = {
            'n_patches': len(patch_coeffs),
            'avg_quality': patch_qualities.mean(),
            'quality_std': patch_qualities.std(),
            'time': elapsed,
            'coeffs_std': coeffs_std
        }
        
        return coeffs_ensemble, term_names, metrics
    
    def print_equation(self, coeffs, term_names, coeffs_std=None, threshold=1e-7):
        """Print discovered equation with uncertainty"""
        print("\nu_t = ", end="")
        
        terms = []
        for i, (c, name) in enumerate(zip(coeffs, term_names)):
            if np.abs(c) > threshold:
                sign = "+" if c >= 0 and len(terms) > 0 else ""
                if coeffs_std is not None:
                    terms.append(f"{sign} ({c:.6e} ± {coeffs_std[i]:.2e})·{name}")
                else:
                    terms.append(f"{sign} {c:.6e}·{name}")
        
        if len(terms) == 0:
            print("0  (no significant terms)")
        else:
            print("\n      ".join(terms))
        
        print(f"\nActive terms: {len(terms)}/{len(coeffs)}")
        
        # Show most confident terms
        if coeffs_std is not None and len(terms) > 0:
            confidence = np.abs(coeffs) / (coeffs_std + 1e-10)
            top_idx = np.argsort(confidence)[::-1][:5]
            
            print("\nMost confident terms:")
            for i, idx in enumerate(top_idx):
                if np.abs(coeffs[idx]) > threshold:
                    conf = confidence[idx]
                    print(f"  {i+1}. {term_names[idx]:12s}: {coeffs[idx]:+.6e} (confidence: {conf:.1f})")
    
    def plot_results(self, coeffs, term_names, metrics, output_file):
        """Visualize patch-based results"""
        fig = plt.figure(figsize=(16, 10))
        
        # Show sample frames
        for i in range(3):
            idx = i * (len(self.images) // 3)
            plt.subplot(2, 4, i+1)
            plt.imshow(self.images[idx], cmap='gray', vmin=0, vmax=1)
            plt.title(f'Frame {idx}')
            plt.axis('off')
        
        # Show patch grid on one frame
        plt.subplot(2, 4, 4)
        img_with_grid = self.images[len(self.images)//2].copy()
        h, w = img_with_grid.shape
        # Draw patch boundaries
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                cv2.rectangle(img_with_grid, (x, y), 
                            (x+self.patch_size, y+self.patch_size), 
                            1.0, 2)
        plt.imshow(img_with_grid, cmap='gray')
        plt.title(f'Patch Grid ({metrics["n_patches"]} patches)')
        plt.axis('off')
        
        # Coefficient bar plot
        plt.subplot(2, 2, 3)
        active = np.abs(coeffs) > 1e-7
        if np.any(active):
            plt.barh(np.array(term_names)[active], coeffs[active])
            plt.xlabel('Coefficient Value')
            plt.title('Active Terms')
            plt.grid(True, alpha=0.3)
        
        # Coefficient uncertainty
        if 'coeffs_std' in metrics:
            plt.subplot(2, 2, 4)
            plt.scatter(np.abs(coeffs), metrics['coeffs_std'], alpha=0.6)
            plt.xlabel('|Coefficient|')
            plt.ylabel('Uncertainty (std)')
            plt.title('Coefficient Uncertainty')
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved results to: {output_file}")


def main():
    """Main execution"""

    IMAGE_FOLDER = DATA_DIR
    OUTPUT_FOLDER = OUTPUT_DIR
    
    print("="*70)
    print("PATCH-BASED SINDy WITH LOCAL REGISTRATION")
    print("="*70)
    
    # Initialize with patch parameters
    sindy = PatchBasedSINDy(
        dt=1.0,
        dx=0.1,
        dy=0.1,
        patch_size=256,    # Patch size (larger = fewer patches, more data per patch)
        overlap=64         # Overlap for smooth transitions
    )
    
    # Load all images
    sindy.load_images(IMAGE_FOLDER, max_images=None)
    
    # Pre-denoise images (MINIMAL smoothing to preserve signal)
    sindy.preprocess_images(spatial_sigma=0.3, temporal_window=3)
    
    # Try different methods
    methods = [
        ('none', 0.01),        # No registration, just smoothing
    ]
    
    results = []
    
    for method, alpha in methods:
        print("\n" + "#"*70)
        print(f"# TRYING: {method.upper()} registration (alpha={alpha})")
        print("#"*70)
        
        try:
            coeffs, terms, metrics = sindy.discover_pde_patch_ensemble(
                alpha=alpha,
                min_patches=5,
                registration_method=method,
                max_patches=None  # Use ALL patches for robustness
            )
            
            if coeffs is not None:
                results.append((method, coeffs, terms, metrics))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Show best result
    if results:
        best = max(results, key=lambda x: x[3]['avg_quality'])
        method, coeffs, terms, metrics = best
        
        print("\n" + "="*70)
        print("BEST RESULT")
        print("="*70)
        print(f"Method: {method}")
        print(f"Quality: {metrics['avg_quality']:.4f}")
        sindy.print_equation(coeffs, terms, metrics.get('coeffs_std'))
        
        # Plot
        sindy.plot_results(coeffs, terms, metrics, 
                          OUTPUT_FOLDER / 'patch_based_sindy_results.png')
    else:
        print("\n✗ No valid results obtained")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
