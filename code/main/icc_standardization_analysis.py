#!/usr/bin/env python3
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"
ICC Standardization Analysis for ORCHID Dataset

This script calculates the Intraclass Correlation Coefficient (ICC) for multiple
outcome measures in the ORCHID dataset, enabling standardized comparison with
the OSR 2024 national data.

Purpose:
--------
To address the methodological concern that the original analysis compared
different outcome measures (approach rate in ORCHID vs. donor rate in OSR 2024),
this script calculates ICC for BOTH measures in ORCHID:
  1. Approach Rate ICC (process measure)
  2. Donor/Transplant Rate ICC (outcome measure)

This allows for:
  - Within-ORCHID comparison of process vs. outcome variance structure
  - Apples-to-apples comparison of ORCHID donor rate ICC vs. OSR 2024 donor rate ICC

Requirements:
-------------
- pandas
- numpy
- statsmodels (for mixed effects models)
- scipy (for statistical tests)

Usage:
------
    python icc_standardization_analysis.py --data /path/to/orchid_data.csv

The script expects a CSV with at minimum these columns:
    - opo_id: OPO identifier
    - hospital_id: Hospital identifier
    - approached: Binary (0/1) indicating if family was approached
    - transplanted: Binary (0/1) indicating if at least one organ was transplanted
    - age: Patient age
    - bmi: Patient BMI

Author: Noah Parrish
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ICCResult:
    """Container for ICC analysis results."""
    outcome: str
    icc: float
    between_opo_variance: float
    within_opo_variance: float
    total_variance: float
    n_observations: int
    n_hospitals: int
    n_opos: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    method: str = "ANOVA"


def calculate_icc_anova(
    data: pd.DataFrame,
    outcome_col: str,
    hospital_col: str = "hospital_id",
    opo_col: str = "opo_id"
) -> ICCResult:
    """
    Calculate ICC using ANOVA-based method for nested data.
    
    This implements the ICC(1) formula for nested designs where hospitals
    are nested within OPOs.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the outcome and grouping variables
    outcome_col : str
        Name of the binary outcome column (0/1)
    hospital_col : str
        Name of the hospital identifier column
    opo_col : str
        Name of the OPO identifier column
        
    Returns
    -------
    ICCResult
        Dataclass containing ICC and variance components
    """
    # Aggregate to hospital level first (mean outcome per hospital)
    hospital_rates = data.groupby([opo_col, hospital_col]).agg(
        rate=(outcome_col, 'mean'),
        n=(outcome_col, 'count')
    ).reset_index()
    
    # Calculate overall mean
    grand_mean = hospital_rates['rate'].mean()
    
    # Calculate between-OPO variance (variance of OPO means)
    opo_means = hospital_rates.groupby(opo_col)['rate'].mean()
    between_opo_var = opo_means.var(ddof=1)
    
    # Calculate within-OPO variance (variance of hospital rates within OPOs)
    within_opo_vars = []
    for opo in hospital_rates[opo_col].unique():
        opo_data = hospital_rates[hospital_rates[opo_col] == opo]['rate']
        if len(opo_data) > 1:
            within_opo_vars.append(opo_data.var(ddof=1))
    within_opo_var = np.mean(within_opo_vars) if within_opo_vars else 0
    
    # Calculate ICC
    total_var = between_opo_var + within_opo_var
    icc = between_opo_var / total_var if total_var > 0 else 0
    
    return ICCResult(
        outcome=outcome_col,
        icc=icc,
        between_opo_variance=between_opo_var,
        within_opo_variance=within_opo_var,
        total_variance=total_var,
        n_observations=len(data),
        n_hospitals=data[hospital_col].nunique(),
        n_opos=data[opo_col].nunique(),
        method="ANOVA (nested)"
    )


def calculate_icc_mixed_model(
    data: pd.DataFrame,
    outcome_col: str,
    hospital_col: str = "hospital_id",
    opo_col: str = "opo_id"
) -> ICCResult:
    """
    Calculate ICC using mixed-effects logistic regression.
    
    This is the more rigorous approach for binary outcomes, fitting a
    random-intercept model with OPO as the grouping variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the outcome and grouping variables
    outcome_col : str
        Name of the binary outcome column (0/1)
    hospital_col : str
        Name of the hospital identifier column
    opo_col : str
        Name of the OPO identifier column
        
    Returns
    -------
    ICCResult
        Dataclass containing ICC and variance components
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("Warning: statsmodels not available, falling back to ANOVA method")
        return calculate_icc_anova(data, outcome_col, hospital_col, opo_col)
    
    # Aggregate to hospital level
    hospital_rates = data.groupby([opo_col, hospital_col]).agg(
        rate=(outcome_col, 'mean'),
        n=(outcome_col, 'count')
    ).reset_index()
    
    # Fit mixed model: rate ~ 1 + (1|OPO)
    # This estimates the variance attributable to OPO
    try:
        model = MixedLM(
            hospital_rates['rate'],
            np.ones(len(hospital_rates)),  # Intercept only
            groups=hospital_rates[opo_col]
        )
        result = model.fit(method='lbfgs', maxiter=1000)
        
        # Extract variance components
        between_opo_var = float(result.cov_re.iloc[0, 0])
        residual_var = result.scale
        total_var = between_opo_var + residual_var
        icc = between_opo_var / total_var if total_var > 0 else 0
        
        return ICCResult(
            outcome=outcome_col,
            icc=icc,
            between_opo_variance=between_opo_var,
            within_opo_variance=residual_var,
            total_variance=total_var,
            n_observations=len(data),
            n_hospitals=data[hospital_col].nunique(),
            n_opos=data[opo_col].nunique(),
            method="Mixed-effects model"
        )
    except Exception as e:
        print(f"Warning: Mixed model failed ({e}), falling back to ANOVA method")
        return calculate_icc_anova(data, outcome_col, hospital_col, opo_col)


def filter_medically_suitable_cohort(
    data: pd.DataFrame,
    age_col: str = "age",
    bmi_col: str = "bmi",
    age_range: Tuple[int, int] = (0, 70),
    bmi_range: Tuple[float, float] = (15.0, 45.0)
) -> pd.DataFrame:
    """
    Filter data to the Medically Suitable Cohort (MSC).
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw ORCHID data
    age_col : str
        Name of the age column
    bmi_col : str
        Name of the BMI column
    age_range : tuple
        (min_age, max_age) inclusive
    bmi_range : tuple
        (min_bmi, max_bmi) inclusive
        
    Returns
    -------
    pd.DataFrame
        Filtered data containing only MSC cases
    """
    mask = (
        (data[age_col] >= age_range[0]) &
        (data[age_col] <= age_range[1]) &
        (data[bmi_col] >= bmi_range[0]) &
        (data[bmi_col] <= bmi_range[1])
    )
    return data[mask].copy()


def run_standardization_analysis(
    data: pd.DataFrame,
    approach_col: str = "approached",
    transplant_col: str = "transplanted",
    hospital_col: str = "hospital_id",
    opo_col: str = "opo_id",
    method: str = "both"
) -> Dict[str, ICCResult]:
    """
    Run the full ICC standardization analysis.
    
    Calculates ICC for both approach rate and transplant rate to enable
    standardized comparison.
    
    Parameters
    ----------
    data : pd.DataFrame
        ORCHID data (ideally filtered to MSC)
    approach_col : str
        Name of the approach indicator column
    transplant_col : str
        Name of the transplant indicator column
    hospital_col : str
        Name of the hospital identifier column
    opo_col : str
        Name of the OPO identifier column
    method : str
        "anova", "mixed", or "both"
        
    Returns
    -------
    dict
        Dictionary of ICCResult objects keyed by outcome name
    """
    results = {}
    
    outcomes = [
        (approach_col, "Approach Rate"),
        (transplant_col, "Transplant Rate")
    ]
    
    for col, name in outcomes:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found in data, skipping {name}")
            continue
            
        if method in ("anova", "both"):
            result = calculate_icc_anova(data, col, hospital_col, opo_col)
            results[f"{name} (ANOVA)"] = result
            
        if method in ("mixed", "both"):
            result = calculate_icc_mixed_model(data, col, hospital_col, opo_col)
            results[f"{name} (Mixed Model)"] = result
    
    return results


def format_results_table(results: Dict[str, ICCResult]) -> str:
    """
    Format results as a publication-ready table.
    
    Parameters
    ----------
    results : dict
        Dictionary of ICCResult objects
        
    Returns
    -------
    str
        Formatted table string
    """
    lines = [
        "=" * 80,
        "ICC STANDARDIZATION ANALYSIS RESULTS",
        "=" * 80,
        "",
        f"{'Outcome':<35} {'ICC':>8} {'Between-OPO':>12} {'Within-OPO':>12} {'Method':>10}",
        "-" * 80
    ]
    
    for name, result in results.items():
        within_pct = (1 - result.icc) * 100
        between_pct = result.icc * 100
        lines.append(
            f"{name:<35} {result.icc:>8.3f} {between_pct:>11.1f}% {within_pct:>11.1f}% {result.method:>10}"
        )
    
    lines.extend([
        "-" * 80,
        "",
        "INTERPRETATION:",
        "  ICC = proportion of variance attributable to differences BETWEEN OPOs",
        "  1 - ICC = proportion of variance occurring WITHIN OPOs (between hospitals)",
        "",
        "KEY COMPARISON:",
        "  If ORCHID Transplant Rate ICC ≈ OSR 2024 Donor Rate ICC (0.069),",
        "  this validates the structural finding across datasets.",
        "",
        "=" * 80
    ])
    
    return "\n".join(lines)


def save_results_csv(results: Dict[str, ICCResult], output_path: Path) -> None:
    """Save results to CSV for further analysis."""
    rows = []
    for name, result in results.items():
        rows.append({
            "outcome": name,
            "icc": result.icc,
            "between_opo_variance_pct": result.icc * 100,
            "within_opo_variance_pct": (1 - result.icc) * 100,
            "between_opo_variance_raw": result.between_opo_variance,
            "within_opo_variance_raw": result.within_opo_variance,
            "total_variance": result.total_variance,
            "n_observations": result.n_observations,
            "n_hospitals": result.n_hospitals,
            "n_opos": result.n_opos,
            "method": result.method
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate ICC for ORCHID data to enable standardized comparison with OSR 2024"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to ORCHID data CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="icc_standardization_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["anova", "mixed", "both"],
        default="both",
        help="ICC calculation method"
    )
    parser.add_argument(
        "--approach-col",
        type=str,
        default="approached",
        help="Name of the approach indicator column"
    )
    parser.add_argument(
        "--transplant-col",
        type=str,
        default="transplanted",
        help="Name of the transplant indicator column"
    )
    parser.add_argument(
        "--hospital-col",
        type=str,
        default="hospital_id",
        help="Name of the hospital identifier column"
    )
    parser.add_argument(
        "--opo-col",
        type=str,
        default="opo_id",
        help="Name of the OPO identifier column"
    )
    parser.add_argument(
        "--age-col",
        type=str,
        default="age",
        help="Name of the age column"
    )
    parser.add_argument(
        "--bmi-col",
        type=str,
        default="bmi",
        help="Name of the BMI column"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip MSC filtering (use all data)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data)
    print(f"Loaded {len(data):,} records")
    
    # Filter to MSC if requested
    if not args.no_filter:
        print("Filtering to Medically Suitable Cohort (age 0-70, BMI 15-45)...")
        data = filter_medically_suitable_cohort(
            data,
            age_col=args.age_col,
            bmi_col=args.bmi_col
        )
        print(f"MSC contains {len(data):,} records")
    
    # Run analysis
    print("\nRunning ICC analysis...")
    results = run_standardization_analysis(
        data,
        approach_col=args.approach_col,
        transplant_col=args.transplant_col,
        hospital_col=args.hospital_col,
        opo_col=args.opo_col,
        method=args.method
    )
    
    # Print results
    print("\n" + format_results_table(results))
    
    # Save results
    save_results_csv(results, Path(args.output))
    
    # Print comparison guidance
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR MANUSCRIPT UPDATE:")
    print("=" * 80)
    print("""
1. Compare ORCHID Transplant Rate ICC to OSR 2024 Donor Rate ICC (0.069)
   - If similar: Strong validation that the structural finding holds across
     both process and outcome measures, and across time periods
   - If different: Discuss why (e.g., different funnel stages, temporal changes)

2. Update manuscript Table 2 (Variance Decomposition) to include:
   | Dataset              | Outcome         | Between-OPO | Within-OPO |
   |----------------------|-----------------|-------------|------------|
   | ORCHID (2015-2021)   | Approach Rate   | XX.X%       | XX.X%      |
   | ORCHID (2015-2021)   | Transplant Rate | XX.X%       | XX.X%      |
   | OSR 2024 (National)  | Donor Rate      | 6.9%        | 93.1%      |

3. Update Figure 1 (ICC Variance Decomposition) to include all three measures

4. Add a paragraph in Methods explaining the standardization approach
""")


if __name__ == "__main__":
    main()
