#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Supplementary Analysis 03: Potential Gains Analysis

Counterfactual Estimation of Donor Gains from Performance Improvement
================================================================================

Description:
    This script estimates the potential gains in organ donors if lower-performing
    hospitals improved to match higher-performing hospitals within the same OPO.
    It also analyzes timing mechanisms and their relationship to outcomes.

Key Analyses:
    1. Potential gains simulation (hospitals reaching OPO percentiles)
    2. Zero-conversion hospital analysis
    3. Timing mechanism analysis (time to approach)
    4. Hospital-level timing correlations

Data Sources:
    - ORCHID v2.1.1 (for timing analysis)
    - OSR 2024 (for potential gains simulation)

Output Files:
    - potential_gains_results.csv: Counterfactual donor estimates
    - timing_analysis_results.csv: Timing mechanism findings

Author: Noah Parrish
Version: 2.0.0
Date: January 2026
License: MIT

Usage:
    python 03_potential_gains_analysis.py
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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ORCHID_PATH = "./data/OPOReferrals.csv"
DEFAULT_OSR_PATH = "./data/OSR_final_tables2505.xlsx"
DEFAULT_OUTPUT_DIR = "./outputs"

MIN_HOSPITAL_REFERRALS = 20
MAX_HOURS_TO_APPROACH = 72  # Filter for reasonable timing values


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(title)
    print(char * width)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_orchid_data(filepath: Path) -> pd.DataFrame:
    """
    Load ORCHID data for timing analysis.
    
    Args:
        filepath: Path to OPOReferrals.csv.
        
    Returns:
        Preprocessed DataFrame.
    """
    print(f"Loading ORCHID data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Standardize binary columns
    for col in ['approached', 'authorized', 'procured', 'transplanted']:
        df[col] = df[col].fillna(0).astype(int)
    
    # Parse time columns
    time_cols = ['time_referred', 'time_approached', 'time_authorized', 'time_procured']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"  Loaded {len(df):,} referrals")
    return df


def load_osr_data(filepath: Path) -> pd.DataFrame:
    """
    Load OSR 2024 data for potential gains analysis.
    
    Args:
        filepath: Path to OSR Excel file.
        
    Returns:
        Preprocessed DataFrame.
    """
    print(f"Loading OSR data from: {filepath}")
    
    try:
        osr = pd.read_excel(filepath, sheet_name='Table B1')
    except Exception:
        osr = pd.read_excel(filepath)
    
    # Standardize column names
    osr = osr.rename(columns={
        'OPO code': 'opo',
        'Referrals': 'referrals',
        'Total donors': 'donors'
    })
    
    # Convert to numeric
    osr['referrals'] = pd.to_numeric(osr['referrals'], errors='coerce')
    osr['donors'] = pd.to_numeric(osr['donors'], errors='coerce').fillna(0)
    
    # Filter and calculate rate
    osr = osr[osr['referrals'] >= MIN_HOSPITAL_REFERRALS].copy()
    osr['donor_rate'] = osr['donors'] / osr['referrals']
    
    print(f"  Loaded {len(osr):,} hospitals with ≥{MIN_HOSPITAL_REFERRALS} referrals")
    return osr


# =============================================================================
# 1. POTENTIAL GAINS SIMULATION
# =============================================================================

def calculate_opo_percentiles(osr: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate OPO-level percentiles for donor rates.
    
    Args:
        osr: OSR hospital-level data.
        
    Returns:
        DataFrame with percentile columns added.
    """
    def add_percentiles(group):
        group = group.copy()
        group['p50'] = group['donor_rate'].quantile(0.50)
        group['p75'] = group['donor_rate'].quantile(0.75)
        group['p90'] = group['donor_rate'].quantile(0.90)
        group['p_max'] = group['donor_rate'].max()
        return group
    
    return osr.groupby('opo', group_keys=False).apply(add_percentiles)


def run_potential_gains_simulation(osr: pd.DataFrame) -> Dict:
    """
    Simulate potential donor gains under different scenarios.
    
    Scenarios:
    1. All hospitals reach OPO median (50th percentile)
    2. All hospitals reach OPO 75th percentile
    3. All hospitals reach OPO 90th percentile
    4. All hospitals reach OPO best performer
    
    Args:
        osr: OSR hospital-level data with percentiles.
        
    Returns:
        Dictionary with simulation results.
    """
    print_section("1. POTENTIAL GAINS SIMULATION")
    
    # Add percentiles
    osr_with_pct = calculate_opo_percentiles(osr)
    
    # Current state
    current_donors = osr_with_pct['donors'].sum()
    current_referrals = osr_with_pct['referrals'].sum()
    
    print(f"\nCurrent State:")
    print(f"  Hospitals: {len(osr_with_pct):,}")
    print(f"  Referrals: {current_referrals:,.0f}")
    print(f"  Donors: {current_donors:,.0f}")
    print(f"  National rate: {100*current_donors/current_referrals:.2f}%")
    
    # Simulate scenarios
    results = {'current_donors': current_donors}
    
    scenarios = [
        ('p50', 'OPO Median', 0.50),
        ('p75', 'OPO 75th Percentile', 0.75),
        ('p90', 'OPO 90th Percentile', 0.90),
        ('p_max', 'OPO Best Performer', 1.00)
    ]
    
    print("\nPotential Gains Scenarios:")
    print(f"{'Scenario':<25} {'Donors':>12} {'Gain':>12} {'% Increase':>12}")
    print("-" * 65)
    
    for col, name, pct in scenarios:
        # Each hospital reaches at least the target percentile
        scenario_donors = osr_with_pct.apply(
            lambda row: max(row['donors'], row['referrals'] * row[col]),
            axis=1
        ).sum()
        
        gain = scenario_donors - current_donors
        pct_increase = 100 * gain / current_donors
        
        results[f'scenario_{col}'] = {
            'name': name,
            'donors': scenario_donors,
            'gain': gain,
            'pct_increase': pct_increase
        }
        
        print(f"{name:<25} {scenario_donors:>12,.0f} {gain:>+12,.0f} {pct_increase:>+11.1f}%")
    
    return results


def analyze_zero_donor_hospitals(osr: pd.DataFrame) -> Dict:
    """
    Analyze hospitals with zero donors despite sufficient referrals.
    
    Args:
        osr: OSR hospital-level data with percentiles.
        
    Returns:
        Dictionary with zero-donor analysis results.
    """
    print_section("2. ZERO-CONVERSION HOSPITAL ANALYSIS")
    
    # Add percentiles if not present
    if 'p50' not in osr.columns:
        osr = calculate_opo_percentiles(osr)
    
    # Identify zero-donor hospitals
    zero_conv = osr[osr['donors'] == 0].copy()
    
    print(f"\nZero-Donor Hospitals:")
    print(f"  Count: {len(zero_conv):,}")
    print(f"  Referrals: {zero_conv['referrals'].sum():,.0f}")
    print(f"  OPOs affected: {zero_conv['opo'].nunique()}")
    
    # Potential if they reached OPO median
    potential_at_median = (zero_conv['referrals'] * zero_conv['p50']).sum()
    
    print(f"\nPotential Gains:")
    print(f"  If zero-conv hospitals reached OPO median: +{potential_at_median:,.0f} donors")
    
    # Distribution by OPO
    zero_by_opo = zero_conv.groupby('opo').agg(
        n_hospitals=('donors', 'count'),
        total_referrals=('referrals', 'sum')
    ).reset_index()
    
    print(f"\nZero-Donor by OPO (top 10):")
    top_opos = zero_by_opo.nlargest(10, 'total_referrals')
    for _, row in top_opos.iterrows():
        print(f"  {row['opo']}: {row['n_hospitals']} hospitals, {row['total_referrals']:,.0f} referrals")
    
    return {
        'n_zero_donor': len(zero_conv),
        'referrals_at_zero': int(zero_conv['referrals'].sum()),
        'opos_affected': zero_conv['opo'].nunique(),
        'potential_at_median': potential_at_median
    }


# =============================================================================
# 3. TIMING MECHANISM ANALYSIS
# =============================================================================

def run_timing_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze timing mechanisms and their relationship to outcomes.
    
    Args:
        df: ORCHID dataset with time columns.
        
    Returns:
        Dictionary with timing analysis results.
    """
    print_section("3. TIMING MECHANISM ANALYSIS")
    
    # Filter to approached cases with valid timing
    approached = df[df['approached'] == 1].copy()
    
    # Calculate time to approach (in hours)
    approached['hours_to_approach'] = (
        approached['time_approached'] - approached['time_referred']
    ).dt.total_seconds() / 3600
    
    # Filter to reasonable values
    valid_timing = approached[
        (approached['hours_to_approach'] >= 0) &
        (approached['hours_to_approach'] <= MAX_HOURS_TO_APPROACH)
    ].copy()
    
    print(f"\nTiming Statistics:")
    print(f"  Approached cases with valid timing: {len(valid_timing):,}")
    print(f"  Mean time to approach: {valid_timing['hours_to_approach'].mean():.1f} hours")
    print(f"  Median time to approach: {valid_timing['hours_to_approach'].median():.1f} hours")
    print(f"  25th percentile: {valid_timing['hours_to_approach'].quantile(0.25):.1f} hours")
    print(f"  75th percentile: {valid_timing['hours_to_approach'].quantile(0.75):.1f} hours")
    
    # Correlation: time to approach vs. transplant success
    r_individual, p_individual = pearsonr(
        valid_timing['hours_to_approach'],
        valid_timing['transplanted']
    )
    
    print(f"\nIndividual-Level Correlation:")
    print(f"  Time to approach vs. transplant: r = {r_individual:.3f}, p = {p_individual:.4f}")
    
    # Hospital-level analysis
    hosp_timing = valid_timing.groupby(['opo', 'hospital_id']).agg(
        mean_hours=('hours_to_approach', 'mean'),
        median_hours=('hours_to_approach', 'median'),
        n_approached=('approached', 'sum'),
        n_transplanted=('transplanted', 'sum')
    ).reset_index()
    
    hosp_timing['success_rate'] = hosp_timing['n_transplanted'] / hosp_timing['n_approached']
    hosp_timing = hosp_timing[hosp_timing['n_approached'] >= 10]
    
    r_hospital, p_hospital = pearsonr(
        hosp_timing['mean_hours'],
        hosp_timing['success_rate']
    )
    
    print(f"\nHospital-Level Correlation:")
    print(f"  Mean time to approach vs. success rate: r = {r_hospital:.3f}, p = {p_hospital:.4f}")
    print(f"  Hospitals analyzed: {len(hosp_timing)}")
    
    # Interpretation
    if r_hospital < 0 and p_hospital < 0.05:
        interpretation = "Faster approach is associated with better outcomes"
    elif r_hospital > 0 and p_hospital < 0.05:
        interpretation = "Slower approach is associated with better outcomes (unexpected)"
    else:
        interpretation = "No significant relationship between timing and outcomes"
    
    print(f"\nInterpretation: {interpretation}")
    
    return {
        'n_valid_timing': len(valid_timing),
        'mean_hours': valid_timing['hours_to_approach'].mean(),
        'median_hours': valid_timing['hours_to_approach'].median(),
        'r_individual': r_individual,
        'p_individual': p_individual,
        'r_hospital': r_hospital,
        'p_hospital': p_hospital,
        'interpretation': interpretation
    }


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_results(
    output_dir: Path,
    potential_gains: Dict,
    zero_conv: Dict,
    timing: Dict
) -> None:
    """Save all results to CSV files."""
    output_dir.mkdir(exist_ok=True)
    
    # Potential gains results
    gains_data = []
    for key, value in potential_gains.items():
        if key.startswith('scenario_'):
            gains_data.append({
                'scenario': value['name'],
                'donors': value['donors'],
                'gain': value['gain'],
                'pct_increase': value['pct_increase']
            })
    
    pd.DataFrame(gains_data).to_csv(
        output_dir / 'potential_gains_results.csv',
        index=False
    )
    
    # Zero-conversion results
    pd.DataFrame([zero_conv]).to_csv(
        output_dir / 'zero_donor_results.csv',
        index=False
    )
    
    # Timing results
    pd.DataFrame([timing]).to_csv(
        output_dir / 'timing_analysis_results.csv',
        index=False
    )
    
    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Potential Gains Analysis"
    )
    parser.add_argument(
        '--orchid', type=str, default=DEFAULT_ORCHID_PATH,
        help='Path to ORCHID OPOReferrals.csv'
    )
    parser.add_argument(
        '--osr', type=str, default=DEFAULT_OSR_PATH,
        help='Path to OSR Excel file'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help='Directory for output files'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("POTENTIAL GAINS ANALYSIS")
    print(f"Version: {__version__}")
    print("=" * 70)
    
    # Load data
    try:
        osr = load_osr_data(Path(args.osr))
        potential_gains = run_potential_gains_simulation(osr)
        zero_conv = analyze_zero_donor_hospitals(osr)
    except FileNotFoundError as e:
        print(f"\nWarning: OSR data not found ({e})")
        print("Skipping potential gains analysis.")
        potential_gains = {}
        zero_conv = {}
    
    try:
        df = load_orchid_data(Path(args.orchid))
        timing = run_timing_analysis(df)
    except FileNotFoundError as e:
        print(f"\nWarning: ORCHID data not found ({e})")
        print("Skipping timing analysis.")
        timing = {}
    
    # Save results
    if potential_gains or timing:
        save_results(
            Path(args.output_dir),
            potential_gains,
            zero_conv,
            timing
        )
    
    # Summary
    print_section("SUMMARY FOR MANUSCRIPT")
    
    if potential_gains and 'scenario_p75' in potential_gains:
        p75 = potential_gains['scenario_p75']
        print(f"""
POTENTIAL GAINS:
  If all hospitals reached their OPO's 75th percentile:
    Additional donors: +{p75['gain']:,.0f}
    Percentage increase: +{p75['pct_increase']:.1f}%
""")
    
    if zero_conv:
        print(f"""
ZERO-CONVERSION HOSPITALS:
  Count: {zero_conv['n_zero_donor']:,}
  Referrals: {zero_conv['referrals_at_zero']:,}
  Potential at OPO median: +{zero_conv['potential_at_median']:,.0f} donors
""")
    
    if timing:
        print(f"""
TIMING MECHANISM:
  Median time to approach: {timing['median_hours']:.1f} hours
  Hospital-level correlation: r = {timing['r_hospital']:.3f}
  {timing['interpretation']}
""")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
