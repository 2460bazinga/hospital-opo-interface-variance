#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Supplementary Analysis 09: Shapley Value Decomposition

Contextualizing the Procurement Funnel: Stage Exits
================================================================================

Description:
    This script implements Shapley value decomposition to fairly attribute
    referral exits across the four stages of the procurement process:
    1. Sorting (identification and approach)
    2. Authorization (family consent)
    3. Procurement (organ recovery)
    4. Placement (matching and transplantation)

    The Shapley value provides a game-theoretic solution to the attribution
    problem, accounting for the multiplicative nature of the procurement
    funnel and the interdependencies between stages.

Methodology:
    For a multiplicative model where:
        Final = Referrals × r₁ × r₂ × r₃ × r₄
    
    The Shapley value for each stage is its average marginal contribution
    across all possible orderings of stages.

Reference:
    Shapley, L. S. (1953). A Value for n-person Games. In H. W. Kuhn &
    A. W. Tucker (Eds.), Contributions to the Theory of Games II.

Data Source:
    ORCHID v2.1.1, PhysioNet (https://physionet.org/content/orchid/2.1.1/)

Output Files:
    - shapley_results.csv: Shapley values for each stage

Author: Noah Parrish
Version: 2.0.0
Date: January 2026
License: MIT

Usage:
    python 09_shapley_decomposition.py --data /path/to/OPOReferrals.csv
================================================================================
"""

__version__ = "2.0.0"
__author__ = "Noah Parrish"
__date__ = "January 2026"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import math
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_PATH = "./data/OPOReferrals.csv"
DEFAULT_OUTPUT_DIR = "./outputs"

# MSC criteria for sensitivity analysis
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


# =============================================================================
# SHAPLEY VALUE CALCULATION
# =============================================================================

def calculate_product(
    rates: Dict[str, float],
    active_stages: Set[str]
) -> float:
    """
    Calculate product of rates for active stages.
    
    Inactive stages are treated as having perfect (100%) conversion.
    
    Args:
        rates: Dictionary mapping stage names to conversion rates.
        active_stages: Set of stage names to include in the product.
        
    Returns:
        Product of rates for active stages.
    """
    product = 1.0
    for stage, rate in rates.items():
        if stage in active_stages:
            product *= rate
        # else: multiply by 1.0 (perfect conversion)
    return product


def compute_shapley_values(
    stages: Dict[str, float],
    n_referrals: int
) -> Dict[str, float]:
    """
    Compute Shapley values for multiplicative exits attribution.
    
    The Shapley value for each stage represents its fair share of the
    total exits, accounting for all possible orderings of stages.
    
    Args:
        stages: Dictionary mapping stage names to conversion rates.
        n_referrals: Number of initial referrals.
        
    Returns:
        Dictionary mapping stage names to Shapley values (exits attribution).
    """
    stage_names = list(stages.keys())
    n = len(stage_names)
    shapley = {s: 0.0 for s in stage_names}
    
    # Iterate over all permutations
    for perm in permutations(stage_names):
        for i, stage in enumerate(perm):
            # Coalition before adding this stage
            coalition_before = set(perm[:i])
            coalition_after = coalition_before | {stage}
            
            # Value = expected donors (referrals × product of rates)
            value_before = n_referrals * calculate_product(stages, coalition_before)
            value_after = n_referrals * calculate_product(stages, coalition_after)
            
            # Marginal contribution (exits caused by this stage)
            marginal = value_before - value_after
            shapley[stage] += marginal
    
    # Average over all permutations
    n_perms = math.factorial(n)
    for stage in shapley:
        shapley[stage] /= n_perms
    
    return shapley


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess ORCHID data.
    
    Args:
        data_path: Path to OPOReferrals.csv.
        
    Returns:
        Preprocessed DataFrame.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Standardize binary columns
    for col in ['approached', 'authorized', 'procured', 'transplanted']:
        df[col] = df[col].fillna(0).astype(int)
    
    # Calculate BMI for MSC filtering
    df['bmi'] = df['weight_kg'] / ((df['height_in'] * 0.0254) ** 2)
    df['bmi'] = df['bmi'].replace([np.inf, -np.inf], np.nan)
    
    print(f"  Total referrals: {len(df):,}")
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_funnel(df: pd.DataFrame, cohort_name: str = "All Referrals") -> Dict:
    """
    Analyze the procurement funnel and calculate stage rates.
    
    Args:
        df: Dataset to analyze.
        cohort_name: Name of the cohort for reporting.
        
    Returns:
        Dictionary with funnel statistics and stage rates.
    """
    n_referrals = len(df)
    n_approached = df['approached'].sum()
    n_authorized = df['authorized'].sum()
    n_procured = df['procured'].sum()
    n_transplanted = df['transplanted'].sum()
    
    # Calculate stage-to-stage conversion rates
    stages = {
        'Sorting': n_approached / n_referrals if n_referrals > 0 else 0,
        'Authorization': n_authorized / n_approached if n_approached > 0 else 0,
        'Procurement': n_procured / n_authorized if n_authorized > 0 else 0,
        'Placement': n_transplanted / n_procured if n_procured > 0 else 0
    }
    
    return {
        'cohort': cohort_name,
        'n_referrals': n_referrals,
        'n_approached': n_approached,
        'n_authorized': n_authorized,
        'n_procured': n_procured,
        'n_transplanted': n_transplanted,
        'stages': stages
    }


def run_shapley_analysis(funnel: Dict) -> Dict:
    """
    Run Shapley decomposition on funnel data.
    
    Args:
        funnel: Dictionary with funnel statistics from analyze_funnel().
        
    Returns:
        Dictionary with Shapley results.
    """
    shapley = compute_shapley_values(funnel['stages'], funnel['n_referrals'])
    total_exits = sum(shapley.values())
    
    return {
        'shapley': shapley,
        'total_exits': total_exits,
        'percentages': {
            stage: 100 * exits / total_exits if total_exits > 0 else 0
            for stage, exits in shapley.items()
        }
    }


def print_funnel_analysis(funnel: Dict) -> None:
    """Print funnel analysis results."""
    print(f"""
PROCUREMENT FUNNEL ({funnel['cohort']}):

  Referrals:     {funnel['n_referrals']:>10,}  (100.0%)
  Approached:    {funnel['n_approached']:>10,}  ({100*funnel['n_approached']/funnel['n_referrals']:>5.1f}%)
  Authorized:    {funnel['n_authorized']:>10,}  ({100*funnel['n_authorized']/funnel['n_referrals']:>5.1f}%)
  Procured:      {funnel['n_procured']:>10,}  ({100*funnel['n_procured']/funnel['n_referrals']:>5.1f}%)
  Transplanted:  {funnel['n_transplanted']:>10,}  ({100*funnel['n_transplanted']/funnel['n_referrals']:>5.1f}%)

STAGE CONVERSION RATES:

  Referral → Approached:     {100*funnel['stages']['Sorting']:>5.1f}%
  Approached → Authorized:   {100*funnel['stages']['Authorization']:>5.1f}%
  Authorized → Procured:     {100*funnel['stages']['Procurement']:>5.1f}%
  Procured → Transplanted:   {100*funnel['stages']['Placement']:>5.1f}%
""")


def print_shapley_results(shapley_result: Dict, cohort_name: str) -> None:
    """Print Shapley decomposition results."""
    print(f"\nSHAPLEY VALUE ATTRIBUTION ({cohort_name})")
    print("-" * 50)
    print(f"\n{'Stage':<25} {'Exits':>12} {'Percent':>10}")
    print("-" * 50)
    
    for stage, exits in sorted(
        shapley_result['shapley'].items(),
        key=lambda x: -x[1]
    ):
        pct = shapley_result['percentages'][stage]
        print(f"{stage:<25} {exits:>12,.0f} {pct:>9.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<25} {shapley_result['total_exits']:>12,.0f} {100.0:>9.1f}%")


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_results(
    output_dir: Path,
    all_refs_result: Dict,
    msc_result: Dict
) -> None:
    """Save Shapley results to CSV."""
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # All referrals
    for stage, exits in all_refs_result["shapley"].items():
        results.append({
            "analysis": "All Referrals",
            "stage": stage,
            "exits": exits,
            "percent": all_refs_result["percentages"][stage]
        })
    
    # MSC
    for stage, exits in msc_result["shapley"].items():
        results.append({
            "analysis": "MSC Only",
            "stage": stage,
            "exits": exits,
            "percent": msc_result["percentages"][stage]
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "shapley_results.csv", index=False)

def create_shapley_figure(
    output_dir: Path,
    shapley_result: Dict,
    cohort_name: str,
    filename: str = "Figure2_Shapley_Decomposition.png"
) -> None:
    """Create and save a bar chart of Shapley decomposition results."""
    stages = list(shapley_result["shapley"].keys())
    exitses = [shapley_result["shapley"][stage] for stage in stages]
    percentages = [shapley_result["percentages"][stage] for stage in stages]

    # Sort by exits percentage in descending order
    sorted_indices = np.argsort(percentages)[::-1]
    stages = [stages[i] for i in sorted_indices]
    exitses = [exitses[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(stages))

    ax.barh(y_pos, percentages, color="#2c7bb6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()  # Stages with highest exits at the top
    ax.set_xlabel("Percentage of Total Referral Exits")
    ax.set_title(f"Shapley Decomposition of Referral Exits by Stage ({cohort_name})")

    # Add percentage labels to bars
    for i, v in enumerate(percentages):
        ax.text(v + 1, i, f"{v:.1f}%", color="black", va="center")

    ax.set_xlim(0, max(percentages) * 1.2) # Adjust x-axis limit for labels

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_dir / filename}")
    
    print(f"\nResults saved to: {output_dir / 'shapley_results.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Shapley Value Decomposition Analysis"
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
    print("SHAPLEY VALUE DECOMPOSITION")
    print(f"Version: {__version__}")
    print("=" * 70)
    
    # Load data
    df = load_data(Path(args.data))
    
    # Analysis 1: All referrals
    print_section("ANALYSIS 1: ALL REFERRALS")
    funnel_all = analyze_funnel(df, "All Referrals")
    print_funnel_analysis(funnel_all)
    shapley_all = run_shapley_analysis(funnel_all)
    print_shapley_results(shapley_all, "All Referrals")
    
    # Analysis 2: MSC cohort
    print_section("ANALYSIS 2: MEDICALLY SUITABLE COHORT (MSC)")
    msc = df[
        (df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX) &
        (df['bmi'] >= BMI_MIN) & (df['bmi'] <= BMI_MAX)
    ].copy()
    
    funnel_msc = analyze_funnel(msc, "MSC")
    print_funnel_analysis(funnel_msc)
    shapley_msc = run_shapley_analysis(funnel_msc)
    print_shapley_results(shapley_msc, "MSC")
    
    # Comparison
    print_section("COMPARISON: ALL REFERRALS vs MSC")
    print(f"""
{'Stage':<25} {'All Refs %':>12} {'MSC %':>12} {'Difference':>12}
{'-'*65}""")
    
    for stage in funnel_all['stages'].keys():
        all_pct = shapley_all['percentages'][stage]
        msc_pct = shapley_msc['percentages'][stage]
        diff = all_pct - msc_pct
        print(f"{stage:<25} {all_pct:>11.1f}% {msc_pct:>11.1f}% {diff:>+11.1f}pp")
    
    # Summary
    print_section("SUMMARY FOR MANUSCRIPT")
    
    sorting_pct = shapley_all['percentages']['Sorting']
    auth_pct = shapley_all['percentages']['Authorization']
    proc_pct = shapley_all['percentages']['Procurement']
    place_pct = shapley_all['percentages']['Placement']
    
    print(f"""
SHAPLEY ATTRIBUTION (All Referrals):

  Sorting (Approach):   {sorting_pct:>5.1f}%  ← Primary bottleneck
  Authorization:        {auth_pct:>5.1f}%
  Procurement:          {proc_pct:>5.1f}%
  Placement:            {place_pct:>5.1f}%

KEY INSIGHT:
  The Sorting stage (identification and approach) accounts for {sorting_pct:.1f}%
  of total referral exits. This is the largest single contributor, supporting
  the coordination constraint hypothesis.

  The approach rate for all referrals ({100*funnel_all['stages']['Sorting']:.1f}%) represents
  the primary opportunity for improvement in organ procurement.
""")
    
    # Save results
    save_results(Path(args.output_dir), shapley_all, shapley_msc)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

def create_shapley_figure(
    output_dir: Path,
    shapley_result: Dict,
    cohort_name: str,
    filename: str = "Figure2_Shapley_Decomposition.png"
) -> None:
    """Create and save a bar chart of Shapley decomposition results."""
    stages = list(shapley_result["shapley"].keys())
    exitses = [shapley_result["shapley"][stage] for stage in stages]
    percentages = [shapley_result["percentages"][stage] for stage in stages]

    # Sort by exits percentage in descending order
    sorted_indices = np.argsort(percentages)[::-1]
    stages = [stages[i] for i in sorted_indices]
    exitses = [exitses[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(stages))

    ax.barh(y_pos, percentages, color="#2c7bb6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()  # Stages with highest exits at the top
    ax.set_xlabel("Percentage of Total Referral Exits")
    ax.set_title(f"Shapley Decomposition of Referral Exits by Stage ({cohort_name})")

    # Add percentage labels to bars
    for i, v in enumerate(percentages):
        ax.text(v + 1, i, f"{v:.1f}%", color="black", va="center")

    ax.set_xlim(0, max(percentages) * 1.2) # Adjust x-axis limit for labels

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()
    print(f"✓ Figure saved: {output_dir / filename}")
