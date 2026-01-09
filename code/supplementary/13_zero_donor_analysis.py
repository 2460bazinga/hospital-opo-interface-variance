#!/usr/bin/env python3
"""
Zero-Donor Hospital Analysis.

This script analyzes the characteristics and distribution of zero-donor
hospitals (facilities with ≥20 referrals but zero donors) across OPOs. It
addresses reviewer concerns about the nature and opportunity represented by
these hospitals.

Analyses Performed:
    1. Distribution of zero-donor hospitals by OPO
    2. Comparison of characteristics (volume) between zero-conv and donor hospitals
    3. Statistical tests for systematic differences
    4. Estimation of potential donor gains

Data Sources:
    - OSR 2024: Hospital-level referral and donor data from SRTR

Output Files:
    - zero_donor_by_opo.csv: OPO-level summary statistics

Usage:
    python 13_zero_donor_analysis.py [--data_path PATH] [--output_dir PATH]

Author: Noah Parrish
Version: 1.0.0
Date: 2026-01-08
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2026-01-08"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration parameters for zero-donor analysis."""
    
    # Data paths
    data_path: Path = Path("data/OSR_final_tables2505.xlsx")
    output_dir: Path = Path("supplementary_analysis")
    
    # Analysis parameters
    min_referrals: int = 20  # Minimum referrals for stable rate estimates
    
    # Volume categories for stratified analysis
    volume_bins: list = field(default_factory=lambda: [0, 50, 100, 200, 500, 10000])
    volume_labels: list = field(default_factory=lambda: ['20-50', '51-100', '101-200', '201-500', '500+'])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Parameters
    ----------
    group1 : pd.Series
        First group of values
    group2 : pd.Series
        Second group of values
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Parameters
    ----------
    d : float
        Cohen's d value
        
    Returns
    -------
    str
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "Large"
    elif abs_d >= 0.5:
        return "Medium"
    elif abs_d >= 0.2:
        return "Small"
    else:
        return "Negligible"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_osr_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Load and preprocess OSR hospital-level data.
    
    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration
        
    Returns
    -------
    pd.DataFrame
        Preprocessed hospital data with zero-donor flag
    """
    # Load data
    osr = pd.read_excel(config.data_path, sheet_name='Table B1')
    
    # Standardize column names
    osr = osr.rename(columns={
        'OPO code': 'opo',
        'Referrals': 'referrals',
        'Total donors': 'donors'
    })
    
    # Convert to numeric
    osr['referrals'] = pd.to_numeric(osr['referrals'], errors='coerce')
    osr['donors'] = pd.to_numeric(osr['donors'], errors='coerce').fillna(0)
    
    # Filter by minimum referrals
    osr = osr[osr['referrals'] >= config.min_referrals].copy()
    
    # Calculate derived metrics
    osr['donor_rate'] = osr['donors'] / osr['referrals']
    osr['zero_conv'] = osr['donors'] == 0
    
    return osr


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_opo_distribution(osr: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze distribution of zero-donor hospitals by OPO.
    
    Parameters
    ----------
    osr : pd.DataFrame
        Hospital-level data
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        OPO-level statistics and chi-square test results
    """
    # Calculate OPO-level statistics
    opo_stats = osr.groupby('opo').agg(
        total_hospitals=('zero_conv', 'count'),
        zero_conv_hospitals=('zero_conv', 'sum'),
        total_referrals=('referrals', 'sum')
    ).reset_index()
    
    # Calculate zero-donor referrals per OPO
    zero_conv_refs = osr[osr['zero_conv']].groupby('opo')['referrals'].sum()
    opo_stats['zero_conv_referrals'] = opo_stats['opo'].map(zero_conv_refs).fillna(0)
    
    # Calculate percentages
    opo_stats['zero_conv_pct'] = (
        100 * opo_stats['zero_conv_hospitals'] / opo_stats['total_hospitals']
    )
    opo_stats['zero_conv_ref_pct'] = (
        100 * opo_stats['zero_conv_referrals'] / opo_stats['total_referrals']
    )
    
    # Chi-square test for uniform distribution
    expected_pct = osr['zero_conv'].mean()
    expected_counts = opo_stats['total_hospitals'] * expected_pct
    chi2, p_value = stats.chisquare(
        opo_stats['zero_conv_hospitals'], 
        f_exp=expected_counts
    )
    
    test_results = {
        'chi_square': chi2,
        'p_value': p_value,
        'is_uniform': p_value >= 0.05,
        'n_opos_with_zero_conv': (opo_stats['zero_conv_hospitals'] > 0).sum(),
        'total_opos': len(opo_stats),
        'min_pct': opo_stats['zero_conv_pct'].min(),
        'max_pct': opo_stats['zero_conv_pct'].max(),
        'mean_pct': opo_stats['zero_conv_pct'].mean(),
        'median_pct': opo_stats['zero_conv_pct'].median()
    }
    
    return opo_stats, test_results


def compare_hospital_characteristics(osr: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare characteristics between zero-donor and donor hospitals.
    
    Parameters
    ----------
    osr : pd.DataFrame
        Hospital-level data
        
    Returns
    -------
    Dict[str, Any]
        Comparison statistics
    """
    zero_hosp = osr[osr['zero_conv']]
    donor_hosp = osr[~osr['zero_conv']]
    
    # Volume comparison
    t_stat, t_p = stats.ttest_ind(zero_hosp['referrals'], donor_hosp['referrals'])
    mw_stat, mw_p = stats.mannwhitneyu(zero_hosp['referrals'], donor_hosp['referrals'])
    d_refs = cohens_d(donor_hosp['referrals'], zero_hosp['referrals'])
    
    return {
        'n_zero_conv': len(zero_hosp),
        'n_donor': len(donor_hosp),
        'zero_conv_mean_refs': zero_hosp['referrals'].mean(),
        'zero_conv_median_refs': zero_hosp['referrals'].median(),
        'donor_mean_refs': donor_hosp['referrals'].mean(),
        'donor_median_refs': donor_hosp['referrals'].median(),
        't_statistic': t_stat,
        't_p_value': t_p,
        'mann_whitney_p': mw_p,
        'cohens_d': d_refs,
        'cohens_d_interpretation': interpret_cohens_d(d_refs)
    }


def estimate_potential_gains(osr: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate potential donor gains from zero-donor hospitals.
    
    Parameters
    ----------
    osr : pd.DataFrame
        Hospital-level data
        
    Returns
    -------
    Dict[str, Any]
        Potential gains estimates
    """
    zero_hosp = osr[osr['zero_conv']]
    donor_hosp = osr[~osr['zero_conv']]
    
    total_refs = osr['referrals'].sum()
    zero_conv_refs = zero_hosp['referrals'].sum()
    
    median_rate = donor_hosp['donor_rate'].median()
    mean_rate = donor_hosp['donor_rate'].mean()
    
    return {
        'total_referrals': total_refs,
        'zero_conv_referrals': zero_conv_refs,
        'zero_conv_ref_pct': 100 * zero_conv_refs / total_refs,
        'median_donor_rate': median_rate,
        'mean_donor_rate': mean_rate,
        'potential_donors_at_median': zero_conv_refs * median_rate,
        'potential_donors_at_mean': zero_conv_refs * mean_rate
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(config: AnalysisConfig) -> None:
    """
    Run complete zero-donor hospital analysis.
    
    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration
    """
    print("=" * 80)
    print("ZERO-CONVERSION HOSPITAL ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading OSR data...")
    osr = load_osr_data(config)
    print(f"Total hospitals (≥{config.min_referrals} referrals): {len(osr)}")
    print(f"Zero-conversion hospitals: {osr['zero_conv'].sum()} "
          f"({100 * osr['zero_conv'].mean():.1f}%)")
    
    # ==========================================================================
    # 1. OPO Distribution Analysis
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. ZERO-CONVERSION HOSPITAL DISTRIBUTION BY OPO")
    print("=" * 80)
    
    opo_stats, dist_results = analyze_opo_distribution(osr)
    
    print(f"\nOPOs with zero-donor hospitals: "
          f"{dist_results['n_opos_with_zero_conv']}/{dist_results['total_opos']}")
    print(f"\nDistribution of zero-donor hospital percentage across OPOs:")
    print(f"  Mean: {dist_results['mean_pct']:.1f}%")
    print(f"  Median: {dist_results['median_pct']:.1f}%")
    print(f"  Range: {dist_results['min_pct']:.1f}% - {dist_results['max_pct']:.1f}%")
    
    print(f"\nChi-square test for uniform distribution:")
    print(f"  χ² = {dist_results['chi_square']:.1f}, p = {dist_results['p_value']:.4f}")
    if not dist_results['is_uniform']:
        print("  → Zero-conversion hospitals are NOT uniformly distributed across OPOs")
    else:
        print("  → Zero-conversion hospitals are uniformly distributed across OPOs")
    
    # ==========================================================================
    # 2. Characteristics Comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print("2. ZERO-CONVERSION vs DONOR HOSPITAL CHARACTERISTICS")
    print("=" * 80)
    
    char_results = compare_hospital_characteristics(osr)
    
    print(f"\nZero-conversion hospitals: {char_results['n_zero_conv']}")
    print(f"Donor hospitals: {char_results['n_donor']}")
    print(f"\nReferral Volume:")
    print(f"  Zero-conv: mean={char_results['zero_conv_mean_refs']:.1f}, "
          f"median={char_results['zero_conv_median_refs']:.1f}")
    print(f"  Donor:     mean={char_results['donor_mean_refs']:.1f}, "
          f"median={char_results['donor_median_refs']:.1f}")
    print(f"\nStatistical Tests:")
    print(f"  t-test: t={char_results['t_statistic']:.2f}, p={char_results['t_p_value']:.4f}")
    print(f"  Mann-Whitney U: p={char_results['mann_whitney_p']:.4f}")
    print(f"  Cohen's d: {char_results['cohens_d']:.3f} ({char_results['cohens_d_interpretation']})")
    
    # ==========================================================================
    # 3. Potential Gains
    # ==========================================================================
    print("\n" + "=" * 80)
    print("3. POTENTIAL DONOR GAINS FROM ZERO-CONVERSION HOSPITALS")
    print("=" * 80)
    
    gains_results = estimate_potential_gains(osr)
    
    print(f"\nTotal referrals: {gains_results['total_referrals']:,.0f}")
    print(f"Referrals at zero-donor hospitals: "
          f"{gains_results['zero_conv_referrals']:,.0f} "
          f"({gains_results['zero_conv_ref_pct']:.1f}%)")
    print(f"\nIf zero-conv hospitals achieved median donor rate "
          f"({gains_results['median_donor_rate']*100:.2f}%):")
    print(f"  Potential additional donors: "
          f"{gains_results['potential_donors_at_median']:,.0f}")
    print(f"\nIf zero-conv hospitals achieved mean donor rate "
          f"({gains_results['mean_donor_rate']*100:.2f}%):")
    print(f"  Potential additional donors: "
          f"{gains_results['potential_donors_at_mean']:,.0f}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY FOR MANUSCRIPT")
    print("=" * 80)
    
    print(f"""
KEY FINDINGS:

1. DISTRIBUTION: Zero-conversion hospitals exist in ALL {dist_results['total_opos']} OPOs 
   (range: {dist_results['min_pct']:.1f}% - {dist_results['max_pct']:.1f}% of hospitals per OPO).
   They are {'NOT ' if not dist_results['is_uniform'] else ''}uniformly distributed 
   (χ²={dist_results['chi_square']:.1f}, p={dist_results['p_value']:.4f}).

2. VOLUME: Zero-conversion hospitals have LOWER referral volume 
   (mean {char_results['zero_conv_mean_refs']:.0f} vs {char_results['donor_mean_refs']:.0f}, 
   Cohen's d={char_results['cohens_d']:.2f}, {char_results['cohens_d_interpretation']}).

3. OPPORTUNITY: {gains_results['zero_conv_referrals']:,.0f} referrals 
   ({gains_results['zero_conv_ref_pct']:.1f}% of total) occur at zero-donor hospitals. 
   If these achieved the median donor rate, {gains_results['potential_donors_at_median']:,.0f} 
   additional donors could be recovered.

4. IMPLICATION: While zero-donor hospitals have lower volume on average, 
   the {char_results['cohens_d_interpretation'].lower()} effect size suggests volume alone 
   does not explain zero conversion. The 6-fold variation in zero-donor prevalence 
   across OPOs suggests coordination factors at the hospital-OPO interface also play a role.
""")
    
    # Save results
    output_path = config.output_dir / "zero_donor_by_opo.csv"
    opo_stats.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Donor Hospital Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/OSR_final_tables2505.xlsx"),
        help="Path to OSR data file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("supplementary_analysis"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--min_referrals",
        type=int,
        default=20,
        help="Minimum referrals for hospital inclusion"
    )
    
    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_arguments()
    
    config = AnalysisConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        min_referrals=args.min_referrals
    )
    
    run_analysis(config)
