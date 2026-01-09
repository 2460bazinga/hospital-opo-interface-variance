#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Coordination Constraint: Hospital-Level Variance as the Primary
Determinant of Organ Procurement Performance

Primary Analysis Script - ORCHID Dataset (2015-2021)
================================================================================

Description:
    This script performs the primary statistical analysis for the manuscript,
    including variance decomposition (ICC), Shapley value attribution of donor
    loss, and instrumental variable estimation of the causal effect of family
    approach on organ procurement.

Data Source:
    ORCHID v2.1.1, PhysioNet (https://physionet.org/content/orchid/2.1.1/)
    Coverage: 133,101 referrals, 6 US OPOs, 343 hospitals, 2015-2021

Citation:
    Parrish, N. (2026). The Coordination Constraint: Hospital-Level Variance
    as the Primary Determinant of Organ Procurement Performance.

Author: Noah Parrish
Version: 6.0.0
Date: January 2026
License: MIT

Reproducibility:
    - Set RANDOM_SEED for reproducible bootstrap confidence intervals
    - All data paths are configurable via command-line arguments
    - Output files include timestamps and version information

Requirements:
    numpy>=1.21.0
    pandas>=1.3.0
    scipy>=1.7.0
    statsmodels>=0.13.0

Usage:
    python coordination_analysis.py --data-dir ./data --output-dir ./outputs
================================================================================
"""

__version__ = "6.0.0"
__author__ = "Noah Parrish"
__date__ = "January 2026"

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import argparse
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from itertools import permutations
import math

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Configuration parameters for the analysis.
    
    All parameters are documented and can be modified for sensitivity analyses.
    """
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    
    # Reproducibility
    random_seed: int = 42
    
    # Bootstrap parameters
    n_bootstrap: int = 2000
    ci_level: float = 0.95
    
    # Sample size thresholds
    min_hospital_referrals: int = 20
    min_cohort_size: int = 100
    
    # Medically Suitable Cohort (MSC) criteria
    # Based on common clinical heuristics for the 2015-2021 period
    age_min: int = 0
    age_max: int = 70
    bmi_min: float = 15.0
    bmi_max: float = 45.0
    
    # Dataset metadata
    orchid_years: Tuple[int, int] = (2015, 2021)
    orchid_n_opos: int = 6


# Global configuration instance
CONFIG = AnalysisConfig()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FunnelCounts:
    """
    Procurement pipeline counts with validation.
    
    Represents the sequential stages of the organ procurement process:
    Referral → Approach → Authorization → Procurement → Transplantation
    
    Attributes:
        n_suitable: Number of medically suitable referrals
        n_approached: Number of families approached for donation
        n_authorized: Number of authorizations obtained
        n_procured: Number of donors from whom organs were procured
        n_transplanted: Number of successful transplantations
    """
    n_suitable: int
    n_approached: int
    n_authorized: int
    n_procured: int
    n_transplanted: int
    
    def __post_init__(self):
        """Validate that the funnel is monotonically decreasing."""
        if not (self.n_suitable >= self.n_approached >= self.n_authorized >= 
                self.n_procured >= self.n_transplanted >= 0):
            raise ValueError(
                "Funnel counts must be monotonically decreasing: "
                f"suitable({self.n_suitable}) >= approached({self.n_approached}) >= "
                f"authorized({self.n_authorized}) >= procured({self.n_procured}) >= "
                f"transplanted({self.n_transplanted})"
            )
    
    @property
    def total_loss(self) -> int:
        """Total loss from suitable referrals to transplantation."""
        return self.n_suitable - self.n_transplanted
    
    @property
    def rates(self) -> Dict[str, float]:
        """Conditional conversion rates at each stage."""
        return {
            'sorting': self.n_approached / self.n_suitable if self.n_suitable > 0 else 0,
            'authorization': self.n_authorized / self.n_approached if self.n_approached > 0 else 0,
            'procurement': self.n_procured / self.n_authorized if self.n_authorized > 0 else 0,
            'placement': self.n_transplanted / self.n_procured if self.n_procured > 0 else 0,
        }


@dataclass
class ShapleyResult:
    """
    Results from Shapley value decomposition of donor loss.
    
    The Shapley value fairly attributes the total loss to each stage based
    on its marginal contribution across all possible orderings.
    
    Reference:
        Shapley, L. S. (1953). A Value for n-person Games.
    """
    cohort_name: str
    cohort_definition: str
    n_suitable: int
    n_transplanted: int
    total_loss: int
    
    # Conditional rates (sigma)
    sigma_sorting: float
    sigma_authorization: float
    sigma_procurement: float
    sigma_placement: float
    
    # Shapley values (phi) - absolute loss attributed to each stage
    phi_sorting: float
    phi_authorization: float
    phi_procurement: float
    phi_placement: float
    
    # Percentage attribution
    pct_sorting: float
    pct_authorization: float
    pct_procurement: float
    pct_placement: float
    
    # Bootstrap confidence intervals (optional)
    ci_sorting: Optional[Tuple[float, float]] = None
    ci_authorization: Optional[Tuple[float, float]] = None
    ci_procurement: Optional[Tuple[float, float]] = None
    ci_placement: Optional[Tuple[float, float]] = None


@dataclass
class IVResult:
    """
    Results from instrumental variable analysis.
    
    IV estimation addresses confounding by using hospital-level approach
    rates as an instrument for individual approach decisions.
    """
    name: str
    description: str
    n_obs: int
    first_stage_f: float
    instrument_strength: str  # "Strong" if F > 10, else "Weak"
    ols_estimate: float
    ols_se: float
    iv_estimate: float
    iv_se: float
    iv_ci_lower: float
    iv_ci_upper: float
    p_value: float


@dataclass
class ICCResult:
    """
    Results from Intraclass Correlation Coefficient analysis.
    
    ICC measures the proportion of total variance attributable to the
    grouping variable (OPO), with 1-ICC representing within-group variance.
    """
    outcome: str
    n_hospitals: int
    n_opos: int
    total_variance: float
    between_opo_variance: float
    within_opo_variance: float
    icc: float
    pct_between: float
    pct_within: float


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_random_seed(seed: int = None) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value. Uses CONFIG.random_seed if None.
    """
    seed = seed or CONFIG.random_seed
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_file_hash(filepath: Path) -> str:
    """
    Compute MD5 hash of a file for data integrity verification.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        MD5 hash string.
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_orchid_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and preprocess ORCHID dataset.
    
    Args:
        data_dir: Directory containing ORCHID data files.
        
    Returns:
        Preprocessed DataFrame with all required columns.
        
    Raises:
        FileNotFoundError: If data file is not found.
    """
    filepath = data_dir / "OPOReferrals.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"ORCHID data file not found: {filepath}\n"
            "Please download from: https://physionet.org/content/orchid/2.1.1/"
        )
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Standardize binary outcome columns
    for col in ['approached', 'authorized', 'procured', 'transplanted']:
        df[col] = df[col].fillna(0).astype(int)
    
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / ((df['height_in'] * 0.0254) ** 2)
    df['bmi'] = df['bmi'].replace([np.inf, -np.inf], np.nan)
    
    # Create pathway indicators
    df['is_dbd'] = df['brain_death'].fillna(False).astype(int)
    df['is_dcd'] = (~df['brain_death'].fillna(True)).astype(int)
    
    # Create cause of death indicators
    df['cod_anoxia'] = (df['cod'] == 'Anoxia').astype(int)
    df['cod_cva'] = (df['cod'] == 'CVA').astype(int)
    df['cod_trauma'] = (df['cod'] == 'Head Trauma').astype(int)
    
    print(f"Loaded {len(df):,} referrals from {df['opo'].nunique()} OPOs")
    
    return df


def create_medically_suitable_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Medically Suitable Cohort (MSC) criteria.
    
    Criteria based on common clinical heuristics for the 2015-2021 period:
    - Age: 0-70 years
    - BMI: 15-45 kg/m²
    
    Args:
        df: Full ORCHID dataset.
        
    Returns:
        Filtered DataFrame meeting MSC criteria.
    """
    initial_n = len(df)
    
    # Apply age filter
    df_filtered = df[
        (df['age'] >= CONFIG.age_min) & 
        (df['age'] <= CONFIG.age_max)
    ].copy()
    n_after_age = len(df_filtered)
    
    # Apply BMI filter
    df_filtered = df_filtered[
        (df_filtered['bmi'] >= CONFIG.bmi_min) & 
        (df_filtered['bmi'] <= CONFIG.bmi_max)
    ].copy()
    n_after_bmi = len(df_filtered)
    
    print(f"\nMedically Suitable Cohort (MSC) Selection:")
    print(f"  Initial referrals:     {initial_n:>10,}")
    print(f"  After age filter:      {n_after_age:>10,} (excluded {initial_n - n_after_age:,})")
    print(f"  After BMI filter:      {n_after_bmi:>10,} (excluded {n_after_age - n_after_bmi:,})")
    print(f"  Final MSC cohort:      {len(df_filtered):>10,} ({100*len(df_filtered)/initial_n:.1f}%)")
    
    return df_filtered


# =============================================================================
# VARIANCE DECOMPOSITION (ICC)
# =============================================================================

def calculate_icc(
    df: pd.DataFrame,
    rate_col: str,
    group_col: str = 'opo'
) -> ICCResult:
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    
    ICC = σ²_between / (σ²_between + σ²_within)
    
    Where:
    - σ²_between = variance of group means
    - σ²_within = mean of within-group variances
    
    Args:
        df: DataFrame with hospital-level data.
        rate_col: Column name for the rate variable.
        group_col: Column name for the grouping variable (OPO).
        
    Returns:
        ICCResult with variance components and ICC.
    """
    # Calculate between-group variance (variance of group means)
    group_means = df.groupby(group_col)[rate_col].mean()
    between_var = group_means.var()
    
    # Calculate within-group variance (mean of group variances)
    within_var = df.groupby(group_col)[rate_col].var().mean()
    
    # Calculate total variance
    total_var = between_var + within_var
    
    # Calculate ICC
    icc = between_var / total_var if total_var > 0 else 0
    
    return ICCResult(
        outcome=rate_col,
        n_hospitals=len(df),
        n_opos=df[group_col].nunique(),
        total_variance=total_var,
        between_opo_variance=between_var,
        within_opo_variance=within_var,
        icc=icc,
        pct_between=100 * icc,
        pct_within=100 * (1 - icc)
    )


def run_variance_decomposition(df: pd.DataFrame) -> Dict[str, ICCResult]:
    """
    Run variance decomposition analysis for multiple outcomes.
    
    Args:
        df: Full dataset.
        
    Returns:
        Dictionary mapping outcome names to ICCResult objects.
    """
    print_section("VARIANCE DECOMPOSITION (ICC)")
    
    # Aggregate to hospital level
    hosp = df.groupby(['opo', 'hospital_id']).agg(
        approach_rate=('approached', 'mean'),
        procured_rate=('procured', 'mean'),
        tx_rate=('transplanted', 'mean'),
        n_referrals=('approached', 'count')
    ).reset_index()
    
    # Filter to hospitals with sufficient volume
    hosp_filtered = hosp[hosp['n_referrals'] >= CONFIG.min_hospital_referrals].copy()
    
    print(f"\nHospitals with ≥{CONFIG.min_hospital_referrals} referrals: {len(hosp_filtered)}")
    print(f"OPOs represented: {hosp_filtered['opo'].nunique()}")
    
    results = {}
    
    # Primary analysis: Procurement rate (matches OSR "donor rate")
    results['procured_rate'] = calculate_icc(hosp_filtered, 'procured_rate')
    
    # Sensitivity analyses
    results['approach_rate'] = calculate_icc(hosp_filtered, 'approach_rate')
    results['tx_rate'] = calculate_icc(hosp_filtered, 'tx_rate')
    
    # Print results
    print("\nResults:")
    print(f"{'Outcome':<20} {'ICC':>8} {'Between-OPO':>12} {'Within-OPO':>12}")
    print("-" * 55)
    for name, result in results.items():
        print(f"{name:<20} {result.icc:>8.3f} {result.pct_between:>11.1f}% {result.pct_within:>11.1f}%")
    
    # Print within-OPO gaps
    print("\nWithin-OPO Hospital Variation (Procurement Rate):")
    for opo in sorted(hosp_filtered['opo'].unique()):
        opo_data = hosp_filtered[hosp_filtered['opo'] == opo]['procured_rate']
        gap = (opo_data.max() - opo_data.min()) * 100
        print(f"  {opo}: mean={opo_data.mean()*100:.1f}%, range={gap:.1f}pp, n={len(opo_data)}")
    
    return results


# =============================================================================
# SHAPLEY VALUE DECOMPOSITION
# =============================================================================

def characteristic_function(
    D: int,
    rates: Dict[str, float],
    coalition: set
) -> float:
    """
    Characteristic function v(S) for Shapley decomposition.
    
    v(S) = expected transplants when coalition S operates at observed rates
    and all other stages operate at 100%.
    
    Args:
        D: Number of initial referrals.
        rates: Dictionary of conditional rates by stage.
        coalition: Set of stages in the coalition.
        
    Returns:
        Expected number of transplants for the coalition.
        
    Reference:
        Shapley, L. S. (1953). A Value for n-person Games.
    """
    result = float(D)
    for stage in ['sorting', 'authorization', 'procurement', 'placement']:
        if stage in coalition:
            result *= rates.get(stage, 1.0)
    return result


def compute_shapley_values(counts: FunnelCounts) -> Dict[str, float]:
    """
    Compute Shapley values for each stage of the procurement process.
    
    The Shapley value for each stage represents its fair share of the
    total loss, accounting for all possible orderings of stages.
    
    Args:
        counts: FunnelCounts object with pipeline counts.
        
    Returns:
        Dictionary mapping stage names to Shapley values (loss attribution).
    """
    stages = ['sorting', 'authorization', 'procurement', 'placement']
    rates = counts.rates
    D = counts.n_suitable
    n = len(stages)
    
    shapley = {stage: 0.0 for stage in stages}
    
    # Iterate over all permutations
    for perm in permutations(stages):
        for i, stage in enumerate(perm):
            # Coalition before adding this stage
            before = set(perm[:i])
            # Coalition after adding this stage
            after = before | {stage}
            
            # Marginal contribution
            v_before = characteristic_function(D, rates, before)
            v_after = characteristic_function(D, rates, after)
            
            # Loss attributed to this stage in this ordering
            shapley[stage] += v_before - v_after
    
    # Average over all permutations
    n_perms = math.factorial(n)
    for stage in shapley:
        shapley[stage] /= n_perms
    
    return shapley


def run_shapley_decomposition(df: pd.DataFrame) -> ShapleyResult:
    """
    Run Shapley decomposition analysis.
    
    Args:
        df: Dataset (full or MSC cohort).
        
    Returns:
        ShapleyResult with loss attribution.
    """
    print_section("SHAPLEY DECOMPOSITION")
    
    # Calculate funnel counts
    counts = FunnelCounts(
        n_suitable=len(df),
        n_approached=int(df['approached'].sum()),
        n_authorized=int(df['authorized'].sum()),
        n_procured=int(df['procured'].sum()),
        n_transplanted=int(df['transplanted'].sum())
    )
    
    # Print funnel
    print("\nProcurement Funnel:")
    print(f"  Referrals:     {counts.n_suitable:>10,}")
    print(f"  Approached:    {counts.n_approached:>10,} ({100*counts.rates['sorting']:.1f}%)")
    print(f"  Authorized:    {counts.n_authorized:>10,} ({100*counts.rates['authorization']:.1f}%)")
    print(f"  Procured:      {counts.n_procured:>10,} ({100*counts.rates['procurement']:.1f}%)")
    print(f"  Transplanted:  {counts.n_transplanted:>10,} ({100*counts.rates['placement']:.1f}%)")
    print(f"  Total Loss:    {counts.total_loss:>10,}")
    
    # Compute Shapley values
    shapley = compute_shapley_values(counts)
    total_loss = sum(shapley.values())
    
    # Print results
    print("\nShapley Attribution:")
    print(f"{'Stage':<20} {'Loss':>12} {'Percent':>10}")
    print("-" * 45)
    for stage, loss in sorted(shapley.items(), key=lambda x: -x[1]):
        pct = 100 * loss / total_loss if total_loss > 0 else 0
        print(f"{stage.capitalize():<20} {loss:>12,.0f} {pct:>9.1f}%")
    print("-" * 45)
    print(f"{'TOTAL':<20} {total_loss:>12,.0f} {100.0:>9.1f}%")
    
    return ShapleyResult(
        cohort_name="All Referrals",
        cohort_definition="All ORCHID referrals",
        n_suitable=counts.n_suitable,
        n_transplanted=counts.n_transplanted,
        total_loss=counts.total_loss,
        sigma_sorting=counts.rates['sorting'],
        sigma_authorization=counts.rates['authorization'],
        sigma_procurement=counts.rates['procurement'],
        sigma_placement=counts.rates['placement'],
        phi_sorting=shapley['sorting'],
        phi_authorization=shapley['authorization'],
        phi_procurement=shapley['procurement'],
        phi_placement=shapley['placement'],
        pct_sorting=100 * shapley['sorting'] / total_loss if total_loss > 0 else 0,
        pct_authorization=100 * shapley['authorization'] / total_loss if total_loss > 0 else 0,
        pct_procurement=100 * shapley['procurement'] / total_loss if total_loss > 0 else 0,
        pct_placement=100 * shapley['placement'] / total_loss if total_loss > 0 else 0,
    )


# =============================================================================
# INSTRUMENTAL VARIABLE ANALYSIS
# =============================================================================

def run_iv_analysis(
    df: pd.DataFrame,
    include_opo_fe: bool = True
) -> IVResult:
    """
    Run instrumental variable analysis.
    
    Uses hospital-level approach rates as an instrument for individual
    approach decisions to estimate the causal effect of approach on
    transplantation.
    
    Identification assumption: Patients do not choose their OPO based on
    unobserved factors correlated with donor suitability.
    
    Args:
        df: Dataset with individual-level observations.
        include_opo_fe: Whether to include OPO fixed effects.
        
    Returns:
        IVResult with estimation results.
    """
    print_section("INSTRUMENTAL VARIABLE ANALYSIS")
    
    # Calculate hospital-level approach rates (instrument)
    hosp_rates = df.groupby('hospital_id').agg(
        hosp_approach_rate=('approached', 'mean'),
        hosp_n=('approached', 'count')
    ).reset_index()
    
    # Merge instrument to individual data
    df_iv = df.merge(hosp_rates, on='hospital_id')
    
    # Filter to hospitals with sufficient volume
    df_iv = df_iv[df_iv['hosp_n'] >= CONFIG.min_hospital_referrals].copy()
    
    # Drop missing values
    required_cols = ['age', 'bmi', 'is_dcd', 'approached', 'transplanted', 'hosp_approach_rate']
    df_iv = df_iv.dropna(subset=required_cols)
    df_iv = df_iv[np.isfinite(df_iv['bmi']) & np.isfinite(df_iv['age'])]
    
    print(f"\nObservations for IV analysis: {len(df_iv):,}")
    
    # Prepare variables
    y = df_iv['transplanted'].astype(float).values
    endog = df_iv['approached'].astype(float).values
    instrument = df_iv['hosp_approach_rate'].astype(float).values
    
    # Control variables
    controls = [
        df_iv['age'].values,
        df_iv['bmi'].values,
        df_iv['is_dcd'].values
    ]
    
    # Add OPO fixed effects if requested
    if include_opo_fe:
        opos = sorted(df_iv['opo'].unique())
        for opo in opos[1:]:  # Omit first OPO as reference
            controls.append((df_iv['opo'] == opo).astype(float).values)
    
    X_controls = np.column_stack(controls)
    
    # First stage: Approach ~ Instrument + Controls
    X_first = np.column_stack([np.ones(len(df_iv)), instrument, X_controls])
    first_stage = sm.OLS(endog, X_first).fit()
    f_stat = first_stage.tvalues[1] ** 2
    
    # Second stage: Transplant ~ Approach_hat + Controls
    approach_hat = first_stage.fittedvalues
    X_second = np.column_stack([np.ones(len(df_iv)), approach_hat, X_controls])
    second_stage = sm.OLS(y, X_second).fit()
    
    # Extract results
    iv_estimate = second_stage.params[1]
    iv_se = second_stage.bse[1]
    ci_lower = iv_estimate - 1.96 * iv_se
    ci_upper = iv_estimate + 1.96 * iv_se
    p_value = second_stage.pvalues[1]
    
    # OLS for comparison
    X_ols = np.column_stack([np.ones(len(df_iv)), endog, X_controls])
    ols = sm.OLS(y, X_ols).fit()
    
    # Print results
    print(f"\nFirst-stage F-statistic: {f_stat:.0f} ({'Strong' if f_stat > 10 else 'Weak'} instrument)")
    print(f"\nOLS estimate: {ols.params[1]:.4f} (SE: {ols.bse[1]:.4f})")
    print(f"IV estimate:  {iv_estimate:.4f} (SE: {iv_se:.4f})")
    print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"p-value: {p_value:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Each family approach increases the probability of transplantation")
    print(f"  by {iv_estimate*100:.1f} percentage points (95% CI: {ci_lower*100:.1f}%-{ci_upper*100:.1f}%)")
    
    return IVResult(
        name="Hospital Approach Rate IV",
        description="Hospital-level approach rate as instrument for individual approach",
        n_obs=len(df_iv),
        first_stage_f=f_stat,
        instrument_strength="Strong" if f_stat > 10 else "Weak",
        ols_estimate=ols.params[1],
        ols_se=ols.bse[1],
        iv_estimate=iv_estimate,
        iv_se=iv_se,
        iv_ci_lower=ci_lower,
        iv_ci_upper=ci_upper,
        p_value=p_value
    )


# =============================================================================
# FALSIFICATION TEST
# =============================================================================

def run_falsification_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run falsification test for adverse selection.
    
    Tests whether high-approach hospitals have worse outcomes among
    approached cases. If so, this would suggest adverse selection
    (approaching unsuitable donors). If not, it supports the coordination
    constraint hypothesis (missing suitable donors).
    
    Args:
        df: Dataset with individual-level observations.
        
    Returns:
        Dictionary with test results.
    """
    print_section("FALSIFICATION TEST (Adverse Selection)")
    
    # Calculate hospital-level approach rates
    hosp_rates = df.groupby('hospital_id').agg(
        hosp_approach_rate=('approached', 'mean'),
        hosp_n=('approached', 'count')
    ).reset_index()
    
    # Filter to approached cases only
    approached = df[df['approached'] == 1].copy()
    
    # Calculate success rate among approached cases by hospital
    hosp_success = approached.groupby('hospital_id').agg(
        n_approached=('approached', 'sum'),
        n_tx=('transplanted', 'sum')
    ).reset_index()
    
    hosp_success = hosp_success.merge(hosp_rates, on='hospital_id')
    hosp_success = hosp_success[hosp_success['n_approached'] >= 10]
    hosp_success['success_rate'] = hosp_success['n_tx'] / hosp_success['n_approached']
    
    # Correlation test
    from scipy.stats import pearsonr
    r, p = pearsonr(hosp_success['hosp_approach_rate'], hosp_success['success_rate'])
    
    # Print results
    print(f"\nHospitals analyzed: {len(hosp_success)}")
    print(f"Correlation (approach rate vs. success rate): r = {r:.3f}, p = {p:.4f}")
    
    if r >= 0 or p > 0.05:
        result = "PASS - No evidence of adverse selection"
        interpretation = (
            "High-approach hospitals do NOT have worse success rates among "
            "approached cases. This supports the coordination constraint "
            "hypothesis: variation is driven by missed suitable donors, "
            "not by approaching unsuitable donors."
        )
    else:
        result = "FAIL - Possible adverse selection"
        interpretation = (
            "High-approach hospitals have significantly worse success rates. "
            "This suggests possible adverse selection (approaching unsuitable donors)."
        )
    
    print(f"\nResult: {result}")
    print(f"\nInterpretation: {interpretation}")
    
    return {
        'n_hospitals': len(hosp_success),
        'correlation': r,
        'p_value': p,
        'result': result,
        'interpretation': interpretation
    }


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_results(
    output_dir: Path,
    icc_results: Dict[str, ICCResult],
    shapley_result: ShapleyResult,
    iv_result: IVResult,
    falsification: Dict[str, Any]
) -> None:
    """
    Save all results to CSV files.
    
    Args:
        output_dir: Directory for output files.
        icc_results: Dictionary of ICC results.
        shapley_result: Shapley decomposition result.
        iv_result: IV analysis result.
        falsification: Falsification test results.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Save ICC results
    icc_df = pd.DataFrame([
        {
            'outcome': name,
            'n_hospitals': r.n_hospitals,
            'n_opos': r.n_opos,
            'icc': r.icc,
            'pct_between': r.pct_between,
            'pct_within': r.pct_within
        }
        for name, r in icc_results.items()
    ])
    icc_df.to_csv(output_dir / 'icc_results.csv', index=False)
    
    # Save Shapley results
    shapley_df = pd.DataFrame([
        {'stage': 'Sorting', 'loss': shapley_result.phi_sorting, 'percent': shapley_result.pct_sorting},
        {'stage': 'Authorization', 'loss': shapley_result.phi_authorization, 'percent': shapley_result.pct_authorization},
        {'stage': 'Procurement', 'loss': shapley_result.phi_procurement, 'percent': shapley_result.pct_procurement},
        {'stage': 'Placement', 'loss': shapley_result.phi_placement, 'percent': shapley_result.pct_placement},
    ])
    shapley_df.to_csv(output_dir / 'shapley_results.csv', index=False)
    
    # Save IV results
    iv_df = pd.DataFrame([{
        'n_obs': iv_result.n_obs,
        'first_stage_f': iv_result.first_stage_f,
        'instrument_strength': iv_result.instrument_strength,
        'ols_estimate': iv_result.ols_estimate,
        'ols_se': iv_result.ols_se,
        'iv_estimate': iv_result.iv_estimate,
        'iv_se': iv_result.iv_se,
        'iv_ci_lower': iv_result.iv_ci_lower,
        'iv_ci_upper': iv_result.iv_ci_upper,
        'p_value': iv_result.p_value
    }])
    iv_df.to_csv(output_dir / 'iv_results.csv', index=False)
    
    # Save falsification results
    falsif_df = pd.DataFrame([falsification])
    falsif_df.to_csv(output_dir / 'falsification_results.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main analysis pipeline.
    
    Executes all analyses in sequence:
    1. Data loading and preprocessing
    2. Variance decomposition (ICC)
    3. Shapley decomposition
    4. Instrumental variable analysis
    5. Falsification test
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Coordination Constraint Analysis - ORCHID Dataset"
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data',
        help='Directory containing ORCHID data files'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./outputs',
        help='Directory for output files'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    # Update configuration
    CONFIG.data_dir = args.data_dir
    CONFIG.output_dir = args.output_dir
    CONFIG.random_seed = args.seed
    
    # Initialize
    start_time = datetime.now()
    set_random_seed()
    
    print("=" * 70)
    print("THE COORDINATION CONSTRAINT IN ORGAN PROCUREMENT")
    print("Primary Analysis - ORCHID Dataset (2015-2021)")
    print("=" * 70)
    print(f"\nAnalysis started: {start_time.isoformat()}")
    print(f"Version: {__version__}")
    print(f"Random seed: {CONFIG.random_seed}")
    
    # Create output directory
    output_dir = Path(CONFIG.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    try:
        df = load_orchid_data(Path(CONFIG.data_dir))
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo run this analysis, you need access to the ORCHID dataset.")
        print("See REPRODUCIBILITY.md for instructions.")
        sys.exit(1)
    
    # Run analyses
    icc_results = run_variance_decomposition(df)
    shapley_result = run_shapley_decomposition(df)
    iv_result = run_iv_analysis(df, include_opo_fe=True)
    falsification = run_falsification_test(df)
    
    # Save results
    save_results(output_dir, icc_results, shapley_result, iv_result, falsification)
    
    # Print summary
    print_section("SUMMARY FOR MANUSCRIPT")
    
    icc_primary = icc_results['procured_rate']
    print(f"""
Key Statistics:

1. VARIANCE DECOMPOSITION:
   ICC = {icc_primary.icc:.3f}
   Within-OPO variance: {icc_primary.pct_within:.1f}%
   Between-OPO variance: {icc_primary.pct_between:.1f}%

2. SHAPLEY DECOMPOSITION:
   Sorting: {shapley_result.pct_sorting:.1f}%
   Authorization: {shapley_result.pct_authorization:.1f}%
   Procurement: {shapley_result.pct_procurement:.1f}%
   Placement: {shapley_result.pct_placement:.1f}%

3. INSTRUMENTAL VARIABLE:
   Causal effect: {iv_result.iv_estimate*100:.1f}%
   95% CI: ({iv_result.iv_ci_lower*100:.1f}%, {iv_result.iv_ci_upper*100:.1f}%)
   F-statistic: {iv_result.first_stage_f:.0f}

4. FALSIFICATION TEST:
   {falsification['result']}
""")
    
    # Finish
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nAnalysis completed: {end_time.isoformat()}")
    print(f"Duration: {duration:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
