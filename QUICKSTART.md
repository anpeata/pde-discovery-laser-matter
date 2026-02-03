# Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/anpeata/pde-discovery-laser-matter.git
cd pde-discovery-laser-matter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run Examples

### Minimal Example
```bash
python examples/basic_usage.py
```

### Full Pipeline
```bash
python scripts/run_all.py
```

### Individual Components

```bash
# Ground-truth benchmark
python scripts/ks2d_stridge_benchmark.py --method blockwise

# Model comparison
python scripts/analyze_results.py

# Patch-based validation
python scripts/patch_based_pde_discovery.py
```

## Explore Notebooks

```bash
jupyter notebook notebooks/
```

**Start with:**
- `01_interactive_exploration.ipynb` - Data exploration
- `02_main_analysis_pipeline.ipynb` - Complete workflow

## Expected Output

- **figures/** - 22 visualizations generated
- **results/** - Model metrics (JSON, TXT)
- Console output with discovered PDEs

## Troubleshooting

**Missing packages?**
```bash
pip install --upgrade -r requirements.txt
```

**TIFF data not found?**
- Real data in: `data/` (51 TIFF frames)
- Synthetic data generated automatically

**Need help?**
- See [README.md](README.md) for overview
- Check [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for details

---

**Next Steps:** Review generated figures in `figures/` directory
