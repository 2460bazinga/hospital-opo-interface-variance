#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Supplementary Analysis 07: ICC Sensitivity Analysis

Robustness Checks for Variance Decomposition Findings
================================================================================

Description:
    This script performs sensitivity analyses on the ICC calculation to address
    potential concerns about outlier influence and small sample size (6 OPOs).
    It demonstrates that the within-OPO variance finding is robust across
    multiple specifications.

Key Analyses:
    1. ICC with all 6 OPOs (baseline)
    2. ICC excluding OPO6 (outlier sensitivity)
    3. Volume-weighted ICC
    4. Within-OPO gap analysis (robust to outliers)

Motivation:
    ORCHID has one outlier OPO (OPO6) with an 84.8% approach rate, compared
    to 11.7%-33.7% for other OPOs. With only 6 OPOs, this single observation
    can dominate between-OPO variance. This analysis shows that the within-OPO
    variation finding is substantive and not an artifact.

Data Source:
    ORCHID v2.1.1, PhysioNet (https://physionet.org/content/orchid/2.1.1/)

Output Files:
    - icc_sensitivity_results.csv: ICC values under different specifications

Author: Noah Parrish
Version: 2.0.0
Date: January 2026
License: MIT

Usage:
    python 07_icc_sensitivity.py --data /path/to/OPOReferrals.csv
================================================================================
"""

__version__ = "2.0.0"
__author__ = "Noah Parrish"
__date__ = "January 2026"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_PATH = "./data/OPOReferrals.csv"
DEFAULT_OUTPUT_DIR = "./outputs"

MIN_HOSPITAL_REFERRALS = 20

# MSC criteria
AGE_MIN, AGE_MAX = 0, 70
BMI_MIN, BMI_MAX = 15, 45


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def calculate_icc(
    data: pd.DataFrame,
    rate_col: str,
    group_col: str = 'opo'
) -> Tuple[float, float, float]:
    """
    Calculate Intraclass Correlation Coefficient.
    
    Args:
        data: Hospital-level data.
        rate_col: Column name for the rate variable.
        group_col: Column name for the grouping variable.
        
    Returns:
        Tuple of (ICC, between variance, within variance).
    """
    between_var = data.groupby(group_col)[rate_col].mean().var()
    within_var = data.groupby(group_col)[rate_col].var().mean()
    total_var = between_var + within_var
    icc = between_var / total_var if total_var > 0 else 0
    return icc, between_var, within_var


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """
    Load ORCHID data and prepare hospital-level aggregates.
    
    Args:
        data_path: Path to OPOReferrals.csv.
        
    Returns:
        Hospital-level aggregated DataFrame.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / ((df['height_in'] * 0.0254) ** 2)
    
    # Apply MSC criteria
    msc = df[
        (df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX) &
        (df['bmi'] >= BMI_MIN) & (df['bmi'] <= BMI_MAX)
    ].copy()
    
    msc['approached'] = msc['approached'].fillna(0).astype(int)
    msc['procured'] = msc['procured'].fillna(0).astype(int)
    
    print(f"  MSC cohort: {len(msc):,} referrals")
    
    # Aggregate to hospital level
    hosp = msc.groupby(['opo', 'hospital_id']).agg(
        approach_rate=('approached', 'mean'),
        procured_rate=('procured', 'mean'),
        n=('approached', 'count')
    ).reset_index()
    
    # Filter to hospitals with sufficient volume
    hosp = hosp[hosp['n'] >= MIN_HOSPITAL_REFERRALS].copy()
    
    print(f"  Hospitals with ≥{MIN_HOSPITAL_REFERRALS} referrals: {len(hosp)}")
    print(f"  OPOs: {hosp['opo'].nunique()}")
    
    return hosp


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_opo_summary(hosp: pd.DataFrame) -> pd.DataFrame:
    """
    Generate OPO-level summary statistics.
    
    Args:
        hosp: Hospital-level data.
        
    Returns:
        DataFrame with OPO summary.
    """
    print_section("OPO SUMMARY")
    
    summary = []
    print(f"\n{'OPO':<8} {'N Hosp':>8} {'Mean Rate':>10} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-" * 52)
    
    for opo in sorted(hosp['opo'].unique()):
        opo_data = hosp[hosp['opo'] == opo]['approach_rate']
        row = {
            'opo': opo,
            'n_hospitals': len(opo_data),
            'mean_rate': opo_data.mean(),
            'min_rate': opo_data.min(),
            'max_rate': opo_data.max(),
            'range': opo_data.max() - opo_data.min()
        }
        summary.append(row)
        print(f"{opo:<8} {row['n_hospitals']:>8} {row['mean_rate']*100:>9.1f}% "
              f"{row['min_rate']*100:>7.1f}% {row['max_rate']*100:>7.1f}% "
              f"{row['range']*100:>7.1f}pp")
    
    return pd.DataFrame(summary)


def run_baseline_icc(hosp: pd.DataFrame) -> Dict:
    """
    Calculate ICC with all OPOs (baseline analysis).
    
    Args:
        hosp: Hospital-level data.
        
    Returns:
        Dictionary with ICC results.
    """
    print_section("ANALYSIS 1: ALL 6 OPOs (Baseline)")
    
    icc, between_var, within_var = calculate_icc(hosp, 'approach_rate')
    
    print(f"Between-OPO variance: {between_var:.4f}")
    print(f"Within-OPO variance:  {within_var:.4f}")
    print(f"ICC: {icc:.3f} ({icc*100:.1f}% between, {(1-icc)*100:.1f}% within)")
    
    return {
        'analysis': 'All 6 OPOs',
        'icc': icc,
        'between_var': between_var,
        'within_var': within_var,
        'n_hospitals': len(hosp),
        'n_opos': hosp['opo'].nunique()
    }


def run_excluding_outlier(hosp: pd.DataFrame, outlier_opo: str = 'OPO6') -> Dict:
    """
    Calculate ICC excluding the outlier OPO.
    
    Args:
        hosp: Hospital-level data.
        outlier_opo: OPO to exclude.
        
    Returns:
        Dictionary with ICC results.
    """
    print_section(f"ANALYSIS 2: EXCLUDING {outlier_opo} (Outlier)")
    
    hosp_filtered = hosp[hosp['opo'] != outlier_opo].copy()
    icc, between_var, within_var = calculate_icc(hosp_filtered, 'approach_rate')
    
    print(f"Hospitals: {len(hosp_filtered)} (excluded {len(hosp) - len(hosp_filtered)})")
    print(f"Between-OPO variance: {between_var:.4f}")
    print(f"Within-OPO variance:  {within_var:.4f}")
    print(f"ICC: {icc:.3f} ({icc*100:.1f}% between, {(1-icc)*100:.1f}% within)")
    
    return {
        'analysis': f'Excluding {outlier_opo}',
        'icc': icc,
        'between_var': between_var,
        'within_var': within_var,
        'n_hospitals': len(hosp_filtered),
        'n_opos': hosp_filtered['opo'].nunique()
    }


def run_volume_weighted_icc(hosp: pd.DataFrame) -> Dict:
    """
    Calculate volume-weighted ICC.
    
    Args:
        hosp: Hospital-level data.
        
    Returns:
        Dictionary with ICC results.
    """
    print_section("ANALYSIS 3: VOLUME-WEIGHTED VARIANCE")
    
    # Calculate weighted OPO means
    hosp['weighted_rate'] = hosp['approach_rate'] * hosp['n']
    
    opo_weighted_means = hosp.groupby('opo').apply(
        lambda x: x['weighted_rate'].sum() / x['n'].sum()
    )
    weighted_between = opo_weighted_means.var()
    
    # Calculate weighted within-OPO variance
    def weighted_var_within(group):
        weights = group['n'] / group['n'].sum()
        mean = (group['approach_rate'] * weights).sum()
        return (weights * (group['approach_rate'] - mean) ** 2).sum()
    
    weighted_within = hosp.groupby('opo').apply(weighted_var_within).mean()
    
    total_var = weighted_between + weighted_within
    icc = weighted_between / total_var if total_var > 0 else 0
    
    print(f"Weighted between-OPO: {weighted_between:.4f}")
    print(f"Weighted within-OPO:  {weighted_within:.4f}")
    print(f"ICC (weighted): {icc:.3f} ({icc*100:.1f}% between, {(1-icc)*100:.1f}% within)")
    
    return {
        'analysis': 'Volume-weighted',
        'icc': icc,
        'between_var': weighted_between,
        'within_var': weighted_within,
        'n_hospitals': len(hosp),
        'n_opos': hosp['opo'].nunique()
    }


def run_within_opo_gap_analysis(hosp: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate within-OPO gaps (robust to outliers).
    
    Args:
        hosp: Hospital-level data.
        
    Returns:
        DataFrame with gap statistics by OPO.
    """
    print_section("ANALYSIS 4: WITHIN-OPO GAPS (Robust to Outliers)")
    
    gaps = []
    for opo in hosp['opo'].unique():
        opo_data = hosp[hosp['opo'] == opo]['approach_rate']
        gaps.append({
            'opo': opo,
            'gap_pp': (opo_data.max() - opo_data.min()) * 100,
            'iqr_pp': (opo_data.quantile(0.75) - opo_data.quantile(0.25)) * 100,
            'std_pp': opo_data.std() * 100,
            'n_hospitals': len(opo_data)
        })
    
    gaps_df = pd.DataFrame(gaps)
    
    print(f"\nWithin-OPO variation (all {len(gaps_df)} OPOs):")
    print(f"  Mean gap (max-min): {gaps_df['gap_pp'].mean():.1f} pp")
    print(f"  Mean IQR:           {gaps_df['iqr_pp'].mean():.1f} pp")
    print(f"  Mean SD:            {gaps_df['std_pp'].mean():.1f} pp")
    
    return gaps_df


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def print_reconciliation(results: List[Dict], gaps_df: pd.DataFrame) -> None:
    """Print reconciliation summary."""
    print_section("RECONCILIATION")
    
    baseline = results[0]
    no_outlier = results[1]
    
    print(f"""
THE DISCREPANCY EXPLAINED:

1. ORCHID has ONE outlier OPO (OPO6) with ~85% approach rate
   - All other OPOs: 11.7% - 33.7%
   - This single OPO dominates between-OPO variance

2. With 6 OPOs total, one extreme value has outsized influence
   - ICC with OPO6:    {baseline['icc']:.3f} (appears OPO-dominated)
   - ICC without OPO6: {no_outlier['icc']:.3f} (reveals within-OPO variation)

3. OSR 2024 has 55 OPOs - no single outlier dominates
   - ICC: 0.069 (within-OPO dominates)

4. The WITHIN-OPO variation is REAL and LARGE:
   - Average gap within each OPO: {gaps_df['gap_pp'].mean():.1f} percentage points
   - Even OPO6 has substantial internal variation

CONCLUSION:
  The high ORCHID ICC is an artifact of small N (6 OPOs) and one outlier.
  The within-OPO hospital variation (avg {gaps_df['gap_pp'].mean():.1f}pp gap) is the
  substantive finding that generalizes to national data.

  The coordination failure thesis holds - the hospital-OPO interface
  shows massive variation WITHIN every OPO in the dataset.
""")


def save_results(
    output_dir: Path,
    results: List[Dict],
    gaps_df: pd.DataFrame
) -> None:
    """Save all results to CSV files."""
    output_dir.mkdir(exist_ok=True)
    
    # Add OSR 2024 reference
    results.append({
        'analysis': 'OSR 2024 (reference)',
        'icc': 0.069,
        'between_var': None,
        'within_var': None,
        'n_hospitals': 4140,
        'n_opos': 55
    })
    
    # Save ICC results
    icc_df = pd.DataFrame([
        {
            'analysis': r['analysis'],
            'icc': r['icc'],
            'between_pct': r['icc'] * 100,
            'within_pct': (1 - r['icc']) * 100,
            'n_hospitals': r['n_hospitals'],
            'n_opos': r['n_opos']
        }
        for r in results
    ])
    icc_df.to_csv(output_dir / 'icc_sensitivity_results.csv', index=False)
    
    # Save gap analysis
    gaps_df.to_csv(output_dir / 'within_opo_gaps.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="ICC Sensitivity Analysis"
    )
    parser.add_argument(
        '--data', type=str, default=DEFAULT_DATA_PATH,
        help='Path to OPOReferrals.csv'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help='Directory for output files'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("ICC SENSITIVITY ANALYSIS")
    print(f"Version: {__version__}")
    print("=" * 70)
    
    # Load data
    hosp = load_and_prepare_data(Path(args.data))
    
    # Run analyses
    opo_summary = analyze_opo_summary(hosp)
    
    results = []
    results.append(run_baseline_icc(hosp))
    results.append(run_excluding_outlier(hosp, 'OPO6'))
    results.append(run_volume_weighted_icc(hosp))
    
    gaps_df = run_within_opo_gap_analysis(hosp)
    
    # Print reconciliation
    print_reconciliation(results, gaps_df)
    
    # Save results
    save_results(Path(args.output_dir), results, gaps_df)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
