#!/usr/bin/env python3
from __future__ import annotations
"""
__date__ = "2025-01-08"
Plausibility Assessment for Coordination Constraint Estimates.

This module validates the magnitude of coordination constraint estimates
against known national statistics and benchmarks.

Purpose
-------
Ensure that analytical estimates are internally consistent and externally
plausible when compared to published national statistics from SRTR/OPTN.

Notes
-----
This validation is essential for credibility. Estimates that are wildly
inconsistent with known benchmarks require investigation and explanation.
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"


from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for plausibility validation."""
    
    data_path: Path = Path('./data/OPOReferrals.csv')
    
    # MSC criteria
    age_max: int = 70
    bmi_min: float = 15.0
    bmi_max: float = 45.0
    
    # ORCHID sample characteristics
    orchid_n_opos: int = 6
    orchid_years: int = 6  # 2015-2021
    
    # National context (for comparison)
    national_n_opos: int = 57
    national_deceased_donors_per_year: int = 13500  # SRTR approximate
    national_organs_transplanted_per_year: int = 35000  # SRTR approximate
    national_potential_donors_estimate: int = 40000  # Common estimate
    
    # IV estimate from primary analysis
    iv_estimate: float = 0.208


CONFIG = ValidationConfig()


# =============================================================================
# Data Loading
# =============================================================================

def load_data(config: ValidationConfig = CONFIG) -> pd.DataFrame:
    """Load ORCHID referral data."""
    return pd.read_csv(config.data_path, low_memory=False)


def apply_msc_criteria(df: pd.DataFrame, config: ValidationConfig = CONFIG) -> pd.DataFrame:
    """Apply medically suitable candidate criteria."""
    age_valid = (df['age'] > 0) & (df['age'] < config.age_max)
    
    height_m = df['height_in'] * 0.0254
    bmi = np.where(
        (df['weight_kg'] > 0) & (height_m > 0),
        df['weight_kg'] / (height_m ** 2),
        np.nan
    )
    bmi_valid = ((bmi >= config.bmi_min) & (bmi <= config.bmi_max)) | pd.isna(bmi)
    
    return df[age_valid & bmi_valid].copy()


# =============================================================================
# Validation Computations
# =============================================================================

def compute_annual_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual summary statistics."""
    annual = df.groupby('referral_year').agg({
        'patient_id': 'count',
        'approached': 'sum',
        'authorized': 'sum',
        'procured': 'sum',
        'transplanted': 'sum'
    }).rename(columns={'patient_id': 'referrals'})
    
    return annual.astype(int)


def compute_national_extrapolation(
    orchid_value: float,
    config: ValidationConfig = CONFIG
) -> float:
    """Extrapolate ORCHID value to national scale."""
    return orchid_value * (config.national_n_opos / config.orchid_n_opos)


def validate_estimates(
    df: pd.DataFrame,
    msc: pd.DataFrame,
    config: ValidationConfig = CONFIG
) -> Dict:
    """
    Validate coordination constraint estimates against benchmarks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full ORCHID dataset.
    msc : pd.DataFrame
        MSC-filtered dataset.
    config : ValidationConfig
        Configuration parameters.
    
    Returns
    -------
    dict
        Validation results and plausibility assessments.
    """
    # ORCHID basics
    annual = compute_annual_statistics(df)
    avg_referrals_per_year = annual['referrals'].mean()
    avg_transplanted_per_year = annual['transplanted'].mean()
    
    # National extrapolation
    national_referrals_est = compute_national_extrapolation(avg_referrals_per_year, config)
    national_tx_est = compute_national_extrapolation(avg_transplanted_per_year, config)
    
    # MSC statistics
    msc_not_approached = len(msc) - msc['approached'].sum()
    msc_per_year = msc_not_approached / config.orchid_years
    
    # Apply IV estimate
    potential_tx_upper_bound = msc_not_approached * config.iv_estimate
    
    # Realistic counterfactual (all OPOs at best rate)
    best_opo_rate = msc.groupby('opo')['approached'].mean().max()
    current_rate = msc['approached'].mean()
    rate_gap = best_opo_rate - current_rate
    
    realistic_additional_approaches = len(msc) * rate_gap
    realistic_additional_tx = realistic_additional_approaches * config.iv_estimate
    
    # Magnitude checks
    actual_tx = msc['transplanted'].sum()
    pct_improvement = 100 * realistic_additional_tx / actual_tx
    
    # National projection of realistic estimate
    national_additional_tx = compute_national_extrapolation(
        realistic_additional_tx / config.orchid_years,
        config
    )
    pct_of_national = 100 * national_additional_tx / config.national_organs_transplanted_per_year
    
    return {
        'orchid_summary': {
            'total_referrals': len(df),
            'years_covered': f"{df['referral_year'].min()}-{df['referral_year'].max()}",
            'n_opos': config.orchid_n_opos,
            'avg_referrals_per_year': avg_referrals_per_year,
            'avg_transplanted_per_year': avg_transplanted_per_year,
        },
        'msc_summary': {
            'total_msc': len(msc),
            'msc_pct_of_total': 100 * len(msc) / len(df),
            'msc_not_approached': msc_not_approached,
            'msc_transplanted': msc['transplanted'].sum(),
        },
        'counterfactual': {
            'current_approach_rate': current_rate,
            'best_opo_rate': best_opo_rate,
            'rate_gap': rate_gap,
            'additional_approaches': realistic_additional_approaches,
            'additional_tx_orchid': realistic_additional_tx,
            'pct_improvement_orchid': pct_improvement,
        },
        'national_projection': {
            'extrapolated_referrals_per_year': national_referrals_est,
            'extrapolated_tx_per_year': national_tx_est,
            'additional_tx_per_year_national': national_additional_tx,
            'pct_of_current_national': pct_of_national,
        },
        'plausibility': {
            'tx_ratio_vs_known': national_tx_est / config.national_deceased_donors_per_year,
            'msc_ratio_vs_40k_claim': compute_national_extrapolation(
                len(msc) / config.orchid_years, config
            ) / config.national_potential_donors_estimate,
            'is_plausible': pct_of_national < 20,  # Modest claim threshold
        }
    }


# =============================================================================
# Reporting
# =============================================================================

def print_validation_report(results: Dict) -> None:
    """Print formatted validation report."""
    print("=" * 70)
    print("PLAUSIBILITY VALIDATION REPORT")
    print("=" * 70)
    
    orchid = results['orchid_summary']
    print(f"""
ORCHID Dataset Summary:
  Total referrals: {orchid['total_referrals']:,}
  Years covered: {orchid['years_covered']}
  OPOs: {orchid['n_opos']}
  Average referrals/year: {orchid['avg_referrals_per_year']:,.0f}
  Average transplanted/year: {orchid['avg_transplanted_per_year']:,.0f}
""")
    
    msc = results['msc_summary']
    print(f"""
MSC Cohort:
  Total MSC referrals: {msc['total_msc']:,}
  MSC as % of all referrals: {msc['msc_pct_of_total']:.1f}%
  MSC not approached: {msc['msc_not_approached']:,}
  MSC transplanted: {msc['msc_transplanted']:,}
""")
    
    cf = results['counterfactual']
    print(f"""
Counterfactual Analysis (All OPOs at Best Rate):
  Current approach rate: {cf['current_approach_rate']:.1%}
  Best OPO rate: {cf['best_opo_rate']:.1%}
  Gap: {cf['rate_gap']:.1%}
  Additional approaches: {cf['additional_approaches']:,.0f}
  Additional transplants (ORCHID sample): {cf['additional_tx_orchid']:,.0f}
  Improvement: {cf['pct_improvement_orchid']:.1f}%
""")
    
    nat = results['national_projection']
    print(f"""
National Projection (Extrapolated):
  Additional TX/year nationally: {nat['additional_tx_per_year_national']:,.0f}
  As % of current national: {nat['pct_of_current_national']:.1f}%
""")
    
    plaus = results['plausibility']
    status = "PLAUSIBLE" if plaus['is_plausible'] else "REQUIRES REVIEW"
    print(f"""
Plausibility Assessment: {status}
  TX ratio vs known national: {plaus['tx_ratio_vs_known']:.2f}x
  MSC ratio vs 40,000 claim: {plaus['msc_ratio_vs_40k_claim']:.1%}
""")
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The counterfactual claim is modest: if all OPOs approached at the best
OPO's rate, the improvement would be approximately {:.0f}% of current
transplants. This is NOT a claim that "10x more donors are possible."

The IV estimate ({:.1%} transplant probability per approach) constrains
the estimates to plausible magnitudes.

Key limitation: These estimates apply to the ORCHID sample (2015-2021)
and should not be directly extrapolated to the current landscape, which
has transformed substantially (6.4x referral growth, DCD expansion).
""".format(
        results['counterfactual']['pct_improvement_orchid'],
        CONFIG.iv_estimate
    ))


# =============================================================================
# Main
# =============================================================================

def main(config: ValidationConfig = CONFIG) -> Dict:
    """Execute plausibility validation."""
    print("Loading data...")
    df = load_data(config)
    
    print("Applying MSC criteria...")
    msc = apply_msc_criteria(df, config)
    
    print("Validating estimates...")
    results = validate_estimates(df, msc, config)
    
    print_validation_report(results)
    
    return results


if __name__ == "__main__":
    results = main()
