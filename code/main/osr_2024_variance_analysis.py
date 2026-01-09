#!/usr/bin/env python3
from __future__ import annotations
"""
__date__ = "2025-01-08"
2024 OSR Variance Decomposition Analysis.

This module validates the structural finding from ORCHID (within-OPO variance
dominates between-OPO variance) using contemporary national data from the
SRTR OPO-Specific Reports (OSR) 2024.

Purpose
-------
Provide external validation of the core structural finding using an
independent, contemporary, national-scale dataset.

Data Source
-----------
SRTR OPO-Specific Reports (2024)
https://www.srtr.org/reports/opo-specific-reports/

Notes
-----
The OSR data captures donor rates (donors per 100 referrals), which differs
from the ORCHID approach rate outcome. This analysis provides corroboration
of the structural pattern, not exact replication of the ORCHID findings.
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OSRConfig:
    """Configuration for OSR 2024 analysis."""
    
    data_path: Path = Path('./OSR_final_tables2505.xlsx')
    output_dir: Path = Path('./osr_2024_analysis')
    
    # Filtering parameters
    min_referrals: int = 20
    min_hospitals_per_opo: int = 3
    
    # ORCHID comparison values
    orchid_icc: float = 0.348
    orchid_pct_within: float = 65.0


CONFIG = OSRConfig()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VarianceDecomposition:
    """Results from ICC variance decomposition."""
    
    total_variance: float
    between_opo_variance: float
    within_opo_variance: float
    icc: float
    n_hospitals: int
    n_opos: int
    
    @property
    def pct_between(self) -> float:
        return 100 * self.icc
    
    @property
    def pct_within(self) -> float:
        return 100 * (1 - self.icc)


@dataclass
class NationalSummary:
    """National-level summary statistics."""
    
    total_referrals: int
    total_donors: int
    total_dbd: int
    total_dcd: int
    
    @property
    def donor_rate(self) -> float:
        return 100 * self.total_donors / self.total_referrals if self.total_referrals > 0 else 0
    
    @property
    def dcd_share(self) -> float:
        return 100 * self.total_dcd / self.total_donors if self.total_donors > 0 else 0


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_osr_data(config: OSRConfig = CONFIG) -> pd.DataFrame:
    """
    Load and prepare OSR 2024 hospital-level data.
    
    Parameters
    ----------
    config : OSRConfig
        Configuration parameters.
    
    Returns
    -------
    pd.DataFrame
        Cleaned hospital-level data.
    """
    xlsx = pd.ExcelFile(config.data_path)
    hospital_data = pd.read_excel(xlsx, sheet_name='Table B1')
    
    # Standardize column names
    hospital_data = hospital_data.rename(columns={
        'OPO code': 'opo',
        'Hospital name': 'hospital',
        'Referrals': 'referrals',
        'Total donors': 'donors',
        'DCD donors': 'dcd_donors',
        'DBD donors': 'dbd_donors',
        'Donors per 100 referrals': 'donor_rate'
    })
    
    # Convert to numeric
    hospital_data['referrals'] = pd.to_numeric(hospital_data['referrals'], errors='coerce')
    hospital_data['donors'] = pd.to_numeric(hospital_data['donors'], errors='coerce')
    hospital_data['dcd_donors'] = pd.to_numeric(hospital_data['dcd_donors'], errors='coerce')
    hospital_data['dbd_donors'] = pd.to_numeric(hospital_data['dbd_donors'], errors='coerce')
    
    return hospital_data


def filter_hospitals(
    hospital_data: pd.DataFrame,
    config: OSRConfig = CONFIG
) -> pd.DataFrame:
    """Filter to hospitals meeting minimum referral threshold."""
    filtered = hospital_data[hospital_data['referrals'] >= config.min_referrals].copy()
    filtered['donor_rate_computed'] = 100 * filtered['donors'] / filtered['referrals']
    return filtered


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_national_summary(hospital_data: pd.DataFrame) -> NationalSummary:
    """Compute national-level summary statistics."""
    return NationalSummary(
        total_referrals=int(hospital_data['referrals'].sum()),
        total_donors=int(hospital_data['donors'].sum()),
        total_dbd=int(hospital_data['dbd_donors'].sum()),
        total_dcd=int(hospital_data['dcd_donors'].sum())
    )


def compute_variance_decomposition(
    hospital_data: pd.DataFrame
) -> VarianceDecomposition:
    """
    Decompose donor rate variance into between-OPO and within-OPO components.
    
    Parameters
    ----------
    hospital_data : pd.DataFrame
        Filtered hospital-level data with donor rates.
    
    Returns
    -------
    VarianceDecomposition
        Variance decomposition results.
    """
    total_var = hospital_data['donor_rate_computed'].var()
    opo_means = hospital_data.groupby('opo')['donor_rate_computed'].mean()
    between_var = opo_means.var()
    within_var = hospital_data.groupby('opo')['donor_rate_computed'].var().mean()
    
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
    
    return VarianceDecomposition(
        total_variance=total_var,
        between_opo_variance=between_var,
        within_opo_variance=within_var,
        icc=icc,
        n_hospitals=len(hospital_data),
        n_opos=hospital_data['opo'].nunique()
    )


def compute_opo_statistics(hospital_data: pd.DataFrame) -> pd.DataFrame:
    """Compute OPO-level aggregate statistics."""
    opo_stats = hospital_data.groupby('opo').agg({
        'referrals': 'sum',
        'donors': 'sum',
        'hospital': 'count'
    })
    opo_stats['opo_rate'] = 100 * opo_stats['donors'] / opo_stats['referrals']
    return opo_stats


def compute_within_opo_gaps(
    hospital_data: pd.DataFrame,
    config: OSRConfig = CONFIG
) -> pd.DataFrame:
    """Compute within-OPO performance gaps."""
    gaps = []
    
    for opo in hospital_data['opo'].unique():
        opo_hospitals = hospital_data[hospital_data['opo'] == opo]
        
        if len(opo_hospitals) >= config.min_hospitals_per_opo:
            gaps.append({
                'opo': opo,
                'n_hospitals': len(opo_hospitals),
                'best_rate': opo_hospitals['donor_rate_computed'].max(),
                'worst_rate': opo_hospitals['donor_rate_computed'].min(),
                'gap': opo_hospitals['donor_rate_computed'].max() - opo_hospitals['donor_rate_computed'].min()
            })
    
    return pd.DataFrame(gaps)


def identify_zero_donor_hospitals(
    hospital_data: pd.DataFrame,
    config: OSRConfig = CONFIG
) -> Dict:
    """
    Identify hospitals with zero donors despite meeting referral threshold.
    
    This analysis identifies hospitals that, despite receiving a meaningful
    volume of referrals, produced zero donors—a key indicator of coordination
    constraints at the hospital-OPO interface.
    """
    zero_donor = hospital_data[
        (hospital_data['referrals'] >= config.min_referrals) &
        (hospital_data['donors'] == 0)
    ]
    
    # Check if every OPO has at least one zero-donor hospital
    opos_with_zero = zero_donor['opo'].nunique()
    total_opos = hospital_data['opo'].nunique()
    
    return {
        'n_zero_donor_hospitals': len(zero_donor),
        'total_referrals_at_zero_hospitals': int(zero_donor['referrals'].sum()),
        'opos_with_zero_donor': opos_with_zero,
        'total_opos': total_opos,
        'all_opos_have_zero_donor': opos_with_zero == total_opos,
        'zero_donor_hospitals': zero_donor
    }


# =============================================================================
# Comparison and Validation
# =============================================================================

def compare_to_orchid(
    var_decomp: VarianceDecomposition,
    gaps: pd.DataFrame,
    config: OSRConfig = CONFIG
) -> Dict:
    """
    Compare OSR 2024 findings to ORCHID (2015-2021) findings.
    
    Returns
    -------
    dict
        Comparison results and validation assessment.
    """
    structural_finding_validated = var_decomp.pct_within > 50
    
    return {
        'orchid': {
            'icc': config.orchid_icc,
            'pct_within': config.orchid_pct_within,
            'outcome_variable': 'Approach rate',
        },
        'osr_2024': {
            'icc': var_decomp.icc,
            'pct_within': var_decomp.pct_within,
            'outcome_variable': 'Donor rate',
        },
        'comparison': {
            'icc_difference': var_decomp.icc - config.orchid_icc,
            'pct_within_difference': var_decomp.pct_within - config.orchid_pct_within,
            'structural_finding_validated': structural_finding_validated,
        },
        'interpretation': (
            "Structural finding validated: Within-OPO variance dominates in both datasets."
            if structural_finding_validated else
            "Structural finding not validated: Pattern has shifted since ORCHID era."
        ),
        'caveats': [
            "OSR 'donor rate' differs from ORCHID 'approach rate' (different funnel stages)",
            "Comparison is indicative, not exact replication",
            "OSR captures end-to-end conversion; ORCHID captured process steps"
        ]
    }


# =============================================================================
# Reporting
# =============================================================================

def print_analysis_report(
    national: NationalSummary,
    var_decomp: VarianceDecomposition,
    opo_stats: pd.DataFrame,
    gaps: pd.DataFrame,
    zero_conv: Dict,
    comparison: Dict
) -> None:
    """Print formatted analysis report."""
    print("=" * 70)
    print("2024 OSR VARIANCE DECOMPOSITION ANALYSIS")
    print("Validation of ORCHID Structural Findings")
    print("=" * 70)
    
    print(f"""
National Summary (2024):
  Total referrals: {national.total_referrals:,}
  Total donors: {national.total_donors:,}
  DBD: {national.total_dbd:,} ({100 - national.dcd_share:.1f}%)
  DCD: {national.total_dcd:,} ({national.dcd_share:.1f}%)
  National donor rate: {national.donor_rate:.2f} per 100 referrals
""")
    
    print(f"""
Variance Decomposition:
  Hospitals analyzed: {var_decomp.n_hospitals:,}
  OPOs: {var_decomp.n_opos}
  
  Total variance: {var_decomp.total_variance:.4f}
  Between-OPO variance: {var_decomp.between_opo_variance:.4f}
  Within-OPO variance: {var_decomp.within_opo_variance:.4f}
  
  ICC: {var_decomp.icc:.3f}
  Between-OPO: {var_decomp.pct_between:.0f}%
  Within-OPO: {var_decomp.pct_within:.0f}%
""")
    
    print(f"""
OPO-Level Variation:
  Rate range: {opo_stats['opo_rate'].min():.2f} - {opo_stats['opo_rate'].max():.2f}
  Ratio (best/worst): {opo_stats['opo_rate'].max() / opo_stats['opo_rate'].min():.2f}x
""")
    
    print(f"""
Within-OPO Hospital Gaps:
  OPOs with ≥3 hospitals: {len(gaps)}
  Average within-OPO gap: {gaps['gap'].mean():.1f} percentage points
  Median within-OPO gap: {gaps['gap'].median():.1f} percentage points
""")
    
    print(f"""
Zero-Donor Hospitals:
  Hospitals with ≥20 referrals and 0 donors: {zero_conv['n_zero_donor_hospitals']:,}
  Total referrals at these hospitals: {zero_conv['total_referrals_at_zero_hospitals']:,}
  OPOs with at least one zero-donor hospital: {zero_conv['opos_with_zero_donor']}/{zero_conv['total_opos']}
  All OPOs have zero-donor hospitals: {zero_conv['all_opos_have_zero_donor']}
""")
    
    print("=" * 70)
    print("COMPARISON: ORCHID (2015-2021) vs OSR (2024)")
    print("=" * 70)
    
    orchid = comparison['orchid']
    osr = comparison['osr_2024']
    print(f"""
                            ORCHID          OSR 2024
                            (2015-2021)     (National)
-----------------------------------------------------------
Outcome variable            {orchid['outcome_variable']:<15} {osr['outcome_variable']}
ICC                         {orchid['icc']:.3f}           {osr['icc']:.3f}
% Between-OPO               {orchid['pct_within']:.0f}%             {100-osr['pct_within']:.0f}%
% Within-OPO                {100-orchid['pct_within']:.0f}%             {osr['pct_within']:.0f}%
""")
    
    print("=" * 70)
    print("VALIDATION ASSESSMENT")
    print("=" * 70)
    print(f"\n{comparison['interpretation']}\n")
    
    print("Caveats:")
    for caveat in comparison['caveats']:
        print(f"  - {caveat}")


# =============================================================================
# Main
# =============================================================================

def main(config: OSRConfig = CONFIG) -> Dict:
    """Execute OSR 2024 variance decomposition analysis."""
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading OSR 2024 data...")
    hospital_data = load_osr_data(config)
    print(f"  Total hospitals: {len(hospital_data):,}")
    
    filtered = filter_hospitals(hospital_data, config)
    print(f"  Hospitals with ≥{config.min_referrals} referrals: {len(filtered):,}")
    
    # Compute analyses
    print("\nComputing analyses...")
    national = compute_national_summary(hospital_data)
    var_decomp = compute_variance_decomposition(filtered)
    opo_stats = compute_opo_statistics(filtered)
    gaps = compute_within_opo_gaps(filtered, config)
    zero_conv = identify_zero_donor_hospitals(filtered, config)
    comparison = compare_to_orchid(var_decomp, gaps, config)
    
    # Print report
    print_analysis_report(national, var_decomp, opo_stats, gaps, zero_conv, comparison)
    
    # Save outputs
    print(f"\nSaving outputs to {config.output_dir}...")
    gaps.to_csv(config.output_dir / 'within_opo_gaps.csv', index=False)
    opo_stats.to_csv(config.output_dir / 'opo_stats.csv')
    zero_conv['zero_donor_hospitals'].to_csv(
        config.output_dir / 'zero_donor_hospitals.csv',
        index=False
    )
    
    return {
        'national_summary': national,
        'variance_decomposition': var_decomp,
        'opo_stats': opo_stats,
        'gaps': gaps,
        'zero_donor': zero_conv,
        'comparison': comparison
    }


if __name__ == "__main__":
    results = main()
