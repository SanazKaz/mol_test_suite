# Molecular Test Suite

A comprehensive analysis pipeline for molecular property calculation, validation, and visualization.

## Overview

This repository provides an orchestrated analysis pipeline for generated molecular structures, including:

- **Property Analysis**: QED, SA score, LogP, MW, Lipinski violations, diversity metrics
- **Validation**: PoseBusters binding pose quality checks
- **Diversity Scoring**: Silliness scoring against ChEMBL reference dataset
- **Similarity Analysis**: SuCOS shape and feature-based similarity (optional)
- **Visualization**: Property distributions, PoseBusters comparisons, 2D molecular grids

## Quick Start

### Basic Usage

Run the complete analysis pipeline on a directory of SDF files:

```bash
python main.py --input data/qed_sigmoid/DiffSBDD_test_pockets/
```

This will:
1. Analyze molecular properties (QED, SA, LogP, MW, etc.)
2. Validate poses with PoseBusters
3. Calculate silliness scores
4. Generate all plots
5. Create an organized results directory with summary report

### Advanced Usage

```bash
# Custom output directory
python main.py --input data/molecules/ --output-dir results/analysis_20240101/

# Filter specific pockets
python main.py --input data/ --pockets 6v0u 6cm4 6luq

# Select properties to plot
python main.py --input data/ --properties qed sa logp

# Include SuCOS similarity analysis
python main.py --input data/ --reference-mol data/reference_ligand.sdf

# Skip plotting (faster for large datasets)
python main.py --input data/ --skip-plots
```

## Pipeline Steps

### 1. Property Analysis
**Script**: `scripts/qed_calc.py`

Calculates comprehensive molecular properties:
- QED (drug-likeness)
- SA score (synthetic accessibility, normalized 0-1)
- LogP (lipophilicity)
- Molecular weight
- H-bond donors/acceptors
- Rotatable bonds
- Lipinski violations
- Pairwise Tanimoto diversity

**Outputs**:
- `molecular_property_csvs/props_method.csv` - Per-molecule properties
- `molecular_property_csvs/counts_per_pocket_method.csv` - Aggregate counts
- `molecular_property_csvs/diversity_per_pocket.csv` - Diversity metrics

### 2. PoseBusters Validation
**Script**: `scripts/posebusters_check.py`

Validates binding poses against PoseBusters quality checks.

**Outputs**:
- `PB_results/*_PB_results.csv` - Raw PoseBusters results
- `PB_results/*_PB_failure_count.csv` - Failure analysis

### 3. Silliness Scoring
**Script**: `scripts/silliness.py`

Scores molecular diversity relative to ChEMBL drugs reference dataset.

**Outputs**:
- `silliness_scores/*_silliness.csv` - Silliness scores per molecule

### 4. SuCOS Similarity Analysis (Optional)
**Script**: `scripts/sucos_analysis.py`

Calculates shape and feature-based similarity to a reference molecule.

**Note**: Currently requires manual configuration of paths in the script.

### 5. Visualization

#### Property Distribution Plots
**Script**: `scripts/plotting/property_distribution_plot.py`

Generates violin plots comparing property distributions across methods with statistical tests (Mann-Whitney U, FDR correction).

#### PoseBusters Comparison
**Script**: `scripts/plotting/martin_style_pb_plot.py`

Creates stacked percentage bar charts of PoseBusters pass/fail rates with chi-squared significance tests (Martin style with hatched bars).

#### 2D Molecular Grids
**Script**: `scripts/plotting/mols_2d_display.py`

Renders 2D molecular structure grids with legends.

**Outputs**:
- `figures/property_distributions/*.png` - Property violin plots
- `figures/posebusters_comparison/*.png` - PoseBusters comparison
- `figures/*.svg` - Vector graphics versions

## Output Directory Structure

```
<output_dir>/
├── molecular_property_csvs/
│   ├── props_method.csv
│   ├── counts_per_pocket_method.csv
│   └── diversity_per_pocket.csv
├── PB_results/
│   ├── *_PB_results.csv
│   └── *_PB_failure_count.csv
├── silliness_scores/
│   └── *_silliness.csv
├── figures/
│   ├── property_distributions/
│   │   ├── qed_distribution.png
│   │   ├── sa_distribution.png
│   │   └── ...
│   └── posebusters_comparison/
│       └── posebusters_comparison.png
├── logs/
│   └── run_<timestamp>.log
└── summary_report_<timestamp>.txt
```

## Individual Script Usage

### Property Analysis

```bash
# Scan entire directory
python scripts/qed_calc.py --scan data/qed_sigmoid/DiffSBDD_test_pockets/

# Manual specification
python scripts/qed_calc.py \
  --add DiffSBDD data/DiffSBDD_6v0u_30_nodes.sdf auto \
  --add PRISM data/PRISM_7e2z_30_nodes.sdf auto
```

### PoseBusters Validation

```bash
# Single file
python scripts/posebusters_check.py data/molecules.sdf

# Directory
python scripts/posebusters_check.py data/qed_sigmoid/DiffSBDD_test_pockets/

# Multiple files
python scripts/posebusters_check.py data/*.sdf
```

### Silliness Scoring

```bash
python scripts/silliness.py \
  --sdf data/molecules.sdf \
  --output results/silliness.csv
```

### Property Distribution Plots

```bash
python scripts/plotting/property_distribution_plot.py \
  --csv molecular_property_csvs/props_method.csv \
  --property qed \
  --pockets 6v0u 6cm4 6luq
```

### PoseBusters Comparison Plots

```bash
python scripts/plotting/martin_style_pb_plot.py \
  --input "DiffSBDD" PB_results/DiffSBDD_PB_results.csv \
  --input "PRISM" PB_results/PRISM_PB_results.csv \
  -o figures/pb_comparison.png
```

## File Naming Conventions

The pipeline automatically infers method names and pocket IDs from filenames:

**Format**: `<METHOD>_<POCKET>_<NODES>_nodes.sdf`

Examples:
- `DiffSBDD_6v0u_30_nodes.sdf` → Method: DiffSBDD, Pocket: 6v0u
- `PRISM_7e2z_20_nodes.sdf` → Method: PRISM, Pocket: 7e2z

## Requirements

### Core Dependencies
- Python 3.8+
- RDKit
- pandas
- numpy
- matplotlib
- seaborn
- scipy

### Specialized Libraries
- PoseBusters
- statannotations
- Pillow (PIL)

### Internal Utilities
- `utils/SA_Score/` - Synthetic accessibility scoring
- `repos/silly_walks/` - Molecular diversity scoring
- `repos/SuCOS/` - Shape/feature similarity (optional)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd mol_test_suite

# Install dependencies (example with pip)
pip install -r requirements.txt

# If using conda
conda env create -f environment.yml
conda activate mol_test_suite
```

## Tips and Best Practices

### For Large Datasets
- Use `--skip-plots` flag during initial analysis
- Run plotting separately after verifying property analysis
- Consider processing subsets of pockets separately

### For Method Comparisons
- Ensure consistent naming conventions across SDF files
- Use the same pocket IDs for fair comparisons
- Generate plots with `--pockets` flag to focus on specific targets

### Error Handling
- Check the log file in `logs/run_<timestamp>.log` for detailed error messages
- Review the summary report for overview of completed/failed steps
- Individual scripts can be re-run independently if a step fails

### SuCOS Analysis
- Currently requires manual path configuration in `scripts/sucos_analysis.py`
- Update `REF_SDF`, `DIFF_SDF`, and `PRISM_SDF` variables
- Run script manually for SuCOS analysis

## Contributing

Contributions are welcome! Please ensure:
1. Scripts follow the established pattern for CLI arguments
2. Output files follow the naming conventions
3. Documentation is updated for new features
4. Error handling is comprehensive

## License

[Specify your license here]

## Citation

If you use this pipeline in your research, please cite:
[Add citation information]

## Contact

[Add contact information]
