# Reproducibility Guide

## Overview

This document provides complete instructions for reproducing all analyses in "The Coordination Constraint: Hospital-Level Variance as the Primary Determinant of Organ Procurement Performance."

## System Requirements

### Software Dependencies

```
Python >= 3.9
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
statsmodels >= 0.13.0
openpyxl >= 3.0.0  # For Excel file reading
```

### Installation

```bash
# Clone repository
git clone https://github.com/2460bazinga/hospital-opo-interface-variance.git
cd hospital-opo-interface-variance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy statsmodels openpyxl
```

## Data Access

### ORCHID Dataset (Primary Analysis)

The ORCHID (Organ Retrieval and Collection of Health Information for Donation) dataset requires credentialed access through PhysioNet.

**Steps to obtain access:**

1. Create a PhysioNet account at https://physionet.org/
2. Complete the required CITI training in human subjects research
3. Sign the data use agreement for ORCHID
4. Download ORCHID v2.1.1 from https://physionet.org/content/orchid/2.1.1/

**Required file:**
- `OPOReferrals.csv` - Place in `./data/` directory

### OSR 2024 Data (Validation Analysis)

The 2024 validation data is derived from publicly available SRTR OPO-Specific Reports.

**Source:** https://www.srtr.org/reports/opo-specific-reports/

**Required file:**
- `OSR_final_tables2505.xlsx` - Included in repository

## Directory Structure

After setup, your directory should look like:

```
hospital-opo-interface-variance/
├── data/
│   └── OPOReferrals.csv          # ORCHID data (user must obtain)
├── coordination_analysis.py       # Primary ORCHID analysis
├── hospital_variance_analysis.py  # Hospital-level analysis
├── osr_2024_variance_analysis.py  # 2024 validation
├── reality_check.py               # Plausibility validation
├── OSR_final_tables2505.xlsx      # OSR 2024 data (included)
└── outputs/                       # Generated outputs
```

## Running the Analyses

### 1. Primary ORCHID Analysis

```bash
python coordination_analysis.py
```

**Outputs:**
- `coordination_analysis_outputs/01_sample_flow.csv` - STROBE participant flow
- `coordination_analysis_outputs/02_shapley.csv` - Shapley decomposition results
- `coordination_analysis_outputs/03_pathways.csv` - DBD/DCD stratification
- `coordination_analysis_outputs/04_variance.csv` - Variance decomposition
- `coordination_analysis_outputs/05_iv.csv` - Instrumental variable results
- `coordination_analysis_outputs/06_counterfactual.csv` - Counterfactual estimates
- `coordination_analysis_outputs/07_temporal_context.txt` - Temporal caveats

**Expected runtime:** ~2-5 minutes (depending on bootstrap iterations)

### 2. Hospital-Level Variance Analysis

```bash
python hospital_variance_analysis.py
```

**Outputs:**
- `outputs/hospital_approach_rates.csv` - Hospital-level statistics
- `outputs/within_opo_gaps.csv` - Within-OPO performance gaps

### 3. OSR 2024 Validation

```bash
python osr_2024_variance_analysis.py
```

**Outputs:**
- `osr_2024_analysis/within_opo_gaps.csv` - 2024 within-OPO gaps
- `osr_2024_analysis/opo_stats.csv` - OPO-level statistics
- `osr_2024_analysis/zero_conversion_hospitals.csv` - Zero-conversion hospitals

### 4. Plausibility Validation

```bash
python reality_check.py
```

**Outputs:** Console output with plausibility assessment

## Reproducing Key Results

### Table 1: Cohort Characteristics

Run `coordination_analysis.py` and examine `01_sample_flow.csv`.

### Table 2: Shapley Decomposition

Run `coordination_analysis.py` and examine `02_shapley.csv`.

Key values:
- Sorting: 63.5% (95% CI: 62.9-64.0%)
- Authorization: 27.9%
- Procurement: 4.9%
- Placement: 3.8%

### Table 3: Variance Decomposition

Run `coordination_analysis.py` and examine `04_variance.csv`.

Key values:
- ICC: 0.348
- Within-OPO variance: 65%
- Between-OPO variance: 35%

### Figure 6: 2024 Validation

Run `osr_2024_variance_analysis.py`.

Key values:
- ICC (2024): 0.069
- Within-OPO variance (2024): 93%
- Zero-conversion hospitals: 2,136 (all 55 OPOs)

## Configuration Options

All scripts use dataclass-based configuration. To modify parameters:

```python
from coordination_analysis import Config

# Create custom configuration
config = Config(
    data_dir='./custom/path',
    n_bootstrap=5000,  # More bootstrap iterations
    min_hospital_referrals=30,  # Stricter threshold
)

# Run with custom config
results = main(config)
```

## Random Seed

All analyses use `random_seed=42` for reproducibility. Bootstrap confidence intervals should be identical across runs with the same seed.

## Verification

To verify your results match the published findings:

| Metric | Expected Value | Tolerance |
|--------|---------------|-----------|
| MSC cohort size | 88,237 | Exact |
| Shapley sorting % | 63.5% | ±0.5% |
| ICC (ORCHID) | 0.348 | ±0.005 |
| IV estimate | 0.199 | ±0.01 |
| First-stage F | 575 | ±10 |
| ICC (OSR 2024) | 0.069 | ±0.005 |

## Troubleshooting

### "File not found" errors

Ensure ORCHID data is placed in `./data/OPOReferrals.csv`.

### Memory errors

The full analysis requires approximately 4GB RAM. If memory is limited:
- Reduce `n_bootstrap` to 500
- Run analyses sequentially rather than in parallel

### Different results

- Verify Python and package versions match requirements
- Ensure random seed is set before analysis
- Check that data files are complete and uncorrupted

## Contact

For questions about reproducibility, please open an issue in this repository.

---

*Last updated: January 2026*
