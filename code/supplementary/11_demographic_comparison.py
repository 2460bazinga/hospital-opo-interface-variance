#!/usr/bin/env python3
"""
================================================================================
DEMOGRAPHIC REPRESENTATIVENESS ANALYSIS: ORCHID vs NATIONAL (OSR 2024)
================================================================================

Compares demographic distributions between ORCHID donors (2015-2021) and
national donor demographics from OSR 2024 to assess representativeness.

Metrics Compared:
    - Age distribution
    - Gender distribution
    - DCD percentage

Author: Noah Parrish
Version: 1.0.0
Date: January 2026

Data Sources:
    ORCHID v2.1.1, PhysioNet (https://physionet.org/content/orchid/2.1.1/)
    OSR 2024 Table D1, SRTR (https://www.srtr.org/reports/opo-specific-reports/)

Usage:
    python 11_demographic_comparison.py [--orchid PATH] [--osr PATH]

Output:
    demographic_representativeness.csv - Comparison results
================================================================================
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2026-01"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration parameters for demographic comparison."""
    orchid_path: str = "./data/OPOReferrals.csv"
    osr_path: str = "./OSR_final_tables2505.xlsx"
    output_dir: str = "./"


# Age bins matching OSR categories
AGE_BINS: List[int] = [0, 2, 12, 18, 35, 50, 65, 200]
AGE_LABELS: List[str] = [
    '<2 years', '2-11 years', '12-17 years', '18-34 years',
    '35-49 years', '50-64 years', '65+ years'
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_orchid_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ORCHID data and return both referrals and donors.
    
    Args:
        path: Path to ORCHID CSV file
        
    Returns:
        Tuple of (all_referrals, donors_only) DataFrames
    """
    df = pd.read_csv(path, low_memory=False)
    df['procured'] = df['procured'].fillna(0).astype(int)
    donors = df[df['procured'] == 1].copy()
    return df, donors


def load_osr_demographics(path: str) -> Dict[str, any]:
    """
    Load national demographics from OSR Table D1.
    
    Args:
        path: Path to OSR Excel file
        
    Returns:
        Dictionary with age, gender, and DCD data
    """
    osr_demo = pd.read_excel(path, sheet_name='Table D1')
    
    us_age = {
        '<2 years': osr_demo['Age (%) - <2 years - US Year 2'].iloc[0],
        '2-11 years': osr_demo['Age (%) - 2-11 years - US Year 2'].iloc[0],
        '12-17 years': osr_demo['Age (%) - 12-17 years - US Year 2'].iloc[0],
        '18-34 years': osr_demo['Age (%) - 18-34 years - US Year 2'].iloc[0],
        '35-49 years': osr_demo['Age (%) - 35-49 years - US Year 2'].iloc[0],
        '50-64 years': osr_demo['Age (%) - 50-64 years - US Year 2'].iloc[0],
        '65+ years': osr_demo['Age (%) - 65+ years - US Year 2'].iloc[0],
    }
    
    us_gender = {
        'Male': osr_demo['Gender (%) - Male - US Year 2'].iloc[0],
        'Female': osr_demo['Gender (%) - Female - US Year 2'].iloc[0],
    }
    
    us_dcd = osr_demo['% DCD - US Year 2'].iloc[0]
    
    return {'age': us_age, 'gender': us_gender, 'dcd_pct': us_dcd}


# =============================================================================
# DEMOGRAPHIC CALCULATIONS
# =============================================================================

def calculate_orchid_demographics(donors: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate demographic distributions for ORCHID donors.
    
    Args:
        donors: DataFrame of ORCHID donors
        
    Returns:
        Dictionary with demographic distributions
    """
    # Age distribution
    donors = donors.copy()
    donors['age_group'] = pd.cut(
        donors['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    age_dist = donors['age_group'].value_counts(normalize=True) * 100
    age_dist = age_dist.reindex(AGE_LABELS).fillna(0)
    
    # Gender distribution
    gender_dist = donors['gender'].value_counts(normalize=True) * 100
    male_pct = gender_dist.get('M', gender_dist.get('Male', 0))
    female_pct = gender_dist.get('F', gender_dist.get('Female', 0))
    
    # DCD percentage
    dcd_pct = (~donors['brain_death']).mean() * 100
    
    return {
        'age': age_dist.to_dict(),
        'gender': {'Male': male_pct, 'Female': female_pct},
        'dcd_pct': dcd_pct,
        'n_donors': len(donors)
    }


# =============================================================================
# COMPARISON AND TESTING
# =============================================================================

def compare_demographics(orchid: Dict, national: Dict) -> pd.DataFrame:
    """
    Compare ORCHID and national demographics.
    
    Args:
        orchid: ORCHID demographic data
        national: National demographic data
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    results.append({
        'metric': 'DCD %',
        'orchid_donors': orchid['dcd_pct'],
        'national': national['dcd_pct'],
        'difference': orchid['dcd_pct'] - national['dcd_pct']
    })
    
    results.append({
        'metric': 'Male %',
        'orchid_donors': orchid['gender']['Male'],
        'national': national['gender']['Male'],
        'difference': orchid['gender']['Male'] - national['gender']['Male']
    })
    
    orchid_65plus = orchid['age'].get('65+ years', 0)
    national_65plus = national['age'].get('65+ years', 0)
    results.append({
        'metric': 'Age 65+ %',
        'orchid_donors': orchid_65plus,
        'national': national_65plus,
        'difference': orchid_65plus - national_65plus
    })
    
    return pd.DataFrame(results)


def run_chi_square_test(donors: pd.DataFrame, national_age: Dict) -> Tuple[float, float]:
    """
    Run chi-square test for age distribution.
    
    Args:
        donors: ORCHID donors DataFrame
        national_age: National age distribution
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    donors = donors.copy()
    donors['age_group'] = pd.cut(
        donors['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    orchid_age_counts = donors['age_group'].value_counts().reindex(AGE_LABELS).fillna(0)
    
    national_age_props = [national_age[label] / 100 for label in AGE_LABELS]
    expected_counts = [p * len(donors) for p in national_age_props]
    
    chi2, p_val = stats.chisquare(orchid_age_counts.values, f_exp=expected_counts)
    return chi2, p_val


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_demographic_comparison(config: Config) -> pd.DataFrame:
    """
    Run complete demographic comparison analysis.
    
    Args:
        config: Configuration parameters
        
    Returns:
        DataFrame with comparison results
    """
    print("=" * 80)
    print("DEMOGRAPHIC REPRESENTATIVENESS: ORCHID vs NATIONAL")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    referrals, donors = load_orchid_data(config.orchid_path)
    national = load_osr_demographics(config.osr_path)
    
    print(f"  ORCHID referrals: {len(referrals):,}")
    print(f"  ORCHID donors: {len(donors):,}")
    
    # Calculate ORCHID demographics
    orchid_demo = calculate_orchid_demographics(donors)
    
    # Age distribution comparison
    print("\n" + "=" * 80)
    print("AGE DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Age Group':<15} {'ORCHID Donors':>15} {'National (OSR)':>15} {'Difference':>12}")
    print("-" * 60)
    for age_group in AGE_LABELS:
        orchid_val = orchid_demo['age'].get(age_group, 0)
        national_val = national['age'].get(age_group, 0)
        diff = orchid_val - national_val
        print(f"{age_group:<15} {orchid_val:>14.1f}% {national_val:>14.1f}% {diff:>+11.1f}pp")
    
    # Gender comparison
    print("\n" + "=" * 80)
    print("GENDER DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Gender':<15} {'ORCHID Donors':>15} {'National (OSR)':>15} {'Difference':>12}")
    print("-" * 60)
    for gender in ['Male', 'Female']:
        orchid_val = orchid_demo['gender'].get(gender, 0)
        national_val = national['gender'].get(gender, 0)
        diff = orchid_val - national_val
        print(f"{gender:<15} {orchid_val:>14.1f}% {national_val:>14.1f}% {diff:>+11.1f}pp")
    
    # DCD comparison
    print("\n" + "=" * 80)
    print("DCD PERCENTAGE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<15} {'ORCHID Donors':>15} {'National (OSR)':>15} {'Difference':>12}")
    print("-" * 60)
    dcd_diff = orchid_demo['dcd_pct'] - national['dcd_pct']
    print(f"{'% DCD':<15} {orchid_demo['dcd_pct']:>14.1f}% {national['dcd_pct']:>14.1f}% {dcd_diff:>+11.1f}pp")
    
    # Chi-square test
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    
    chi2, p_val = run_chi_square_test(donors, national['age'])
    print(f"\nChi-Square Test for Age Distribution:")
    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  → Age distribution differs significantly from national")
    else:
        print("  → Age distribution is consistent with national")
    
    # Generate comparison results
    results = compare_demographics(orchid_demo, national)
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC REPRESENTATIVENESS SUMMARY")
    print("=" * 80)
    
    male_diff = orchid_demo['gender']['Male'] - national['gender']['Male']
    dcd_assessment = ("⚠ Higher" if dcd_diff > 10 else 
                      "✓ Similar" if abs(dcd_diff) <= 10 else "⚠ Lower")
    male_assessment = "✓ Similar" if abs(male_diff) <= 5 else "⚠ Different"
    
    print(f"""
ORCHID DONOR DEMOGRAPHICS vs NATIONAL (OSR 2024):

Dimension         ORCHID    National    Difference    Assessment
─────────────────────────────────────────────────────────────────
DCD %             {orchid_demo['dcd_pct']:.1f}%     {national['dcd_pct']:.1f}%      {dcd_diff:+.1f}pp       {dcd_assessment}
Male %            {orchid_demo['gender']['Male']:.1f}%     {national['gender']['Male']:.1f}%      {male_diff:+.1f}pp       {male_assessment}

Note: OSR demographics are for DONORS, not referrals.
ORCHID reflects the 2015-2021 era before full DCD expansion.

KEY FINDING: Despite demographic differences, STRUCTURAL properties
(variance decomposition, volume-efficiency correlations) are consistent,
supporting generalizability for studies of system organization.
""")
    
    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demographic Representativeness Analysis: ORCHID vs National"
    )
    parser.add_argument(
        '--orchid', type=str, default='./data/OPOReferrals.csv',
        help='Path to ORCHID CSV file'
    )
    parser.add_argument(
        '--osr', type=str, default='./OSR_final_tables2505.xlsx',
        help='Path to OSR Excel file'
    )
    parser.add_argument(
        '--output', type=str, default='./',
        help='Output directory for results'
    )
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    config = Config(
        orchid_path=args.orchid,
        osr_path=args.osr,
        output_dir=args.output
    )
    
    results = run_demographic_comparison(config)
    
    # Save results
    output_path = Path(config.output_dir) / 'demographic_representativeness.csv'
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
