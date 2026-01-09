#!/usr/bin/env python3
from __future__ import annotations
"""
__date__ = "2025-01-08"
Hospital-Level Variance Analysis for Organ Procurement Performance.

This module implements variance decomposition and instrumental variable
analysis to test coordination versus selection hypotheses in organ
procurement at the hospital-OPO interface.

Methods
-------
- Intraclass Correlation Coefficient (ICC) decomposition
- Within-OPO hospital variance analysis
- Outcome falsification test
- Hospital-level instrumental variable estimation

References
----------
Chen, B., Lawson, K. A., Finelli, A., & Saarela, O. (2020). Causal variance
    decompositions for institutional comparisons in healthcare. Statistical
    Methods in Medical Research, 29(7), 1972-1986.
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"


import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration parameters for hospital variance analysis."""
    
    data_path: Path = Path('./data/OPOReferrals.csv')
    output_dir: Path = Path('./outputs')
    
    # Cohort criteria
    age_min: int = 0
    age_max: int = 70
    bmi_min: float = 15.0
    bmi_max: float = 45.0
    
    # Minimum sample sizes
    min_hospital_referrals: int = 20
    min_hospital_referrals_strict: int = 30
    min_hospitals_per_opo: int = 3
    
    # Statistical parameters
    confidence_level: float = 0.95
    random_seed: int = 42


CONFIG = AnalysisConfig()


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
        """Percentage of variance attributable to between-OPO differences."""
        return 100 * self.icc
    
    @property
    def pct_within(self) -> float:
        """Percentage of variance attributable to within-OPO differences."""
        return 100 * (1 - self.icc)


@dataclass
class IVEstimate:
    """Results from instrumental variable estimation."""
    
    n_observations: int
    first_stage_f: float
    ols_estimate: float
    ols_se: float
    iv_estimate: float
    iv_se: float
    
    @property
    def iv_ci_lower(self) -> float:
        """Lower bound of 95% confidence interval."""
        return self.iv_estimate - 1.96 * self.iv_se
    
    @property
    def iv_ci_upper(self) -> float:
        """Upper bound of 95% confidence interval."""
        return self.iv_estimate + 1.96 * self.iv_se
    
    @property
    def instrument_strength(self) -> str:
        """Assessment of instrument strength based on F-statistic."""
        return "Strong" if self.first_stage_f > 10 else "Weak"


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_prepare_data(config: AnalysisConfig = CONFIG) -> pd.DataFrame:
    """
    Load ORCHID data and apply medically suitable candidate (MSC) criteria.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration parameters including data path and cohort criteria.
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset containing medically suitable candidates.
    
    Notes
    -----
    MSC criteria applied:
    - Age: 0-70 years
    - BMI: 15-45 kg/m² (where calculable)
    """
    df = pd.read_csv(config.data_path, low_memory=False)
    
    # Apply age criteria
    age_valid = (df['age'] > config.age_min) & (df['age'] < config.age_max)
    
    # Calculate and apply BMI criteria
    height_m = df['height_in'] * 0.0254
    bmi = np.where(
        (df['weight_kg'] > 0) & (height_m > 0),
        df['weight_kg'] / (height_m ** 2),
        np.nan
    )
    bmi_valid = ((bmi >= config.bmi_min) & (bmi <= config.bmi_max)) | pd.isna(bmi)
    
    # Apply filters
    msc = df[age_valid & bmi_valid].copy()
    
    # Ensure binary columns are integer type
    for col in ['approached', 'authorized', 'procured', 'transplanted']:
        msc[col] = msc[col].astype(int)
    
    # Create pathway indicator
    msc['is_dbd'] = (msc['brain_death'] == True).astype(int)
    
    return msc


# =============================================================================
# Variance Decomposition
# =============================================================================

def compute_variance_decomposition(
    df: pd.DataFrame,
    config: AnalysisConfig = CONFIG
) -> Tuple[VarianceDecomposition, pd.DataFrame]:
    """
    Decompose approach rate variance into between-OPO and within-OPO components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns: opo, hospital_id, approached.
    config : AnalysisConfig
        Configuration parameters.
    
    Returns
    -------
    VarianceDecomposition
        Variance decomposition results including ICC.
    pd.DataFrame
        Hospital-level aggregated data.
    
    Notes
    -----
    The Intraclass Correlation Coefficient (ICC) represents the proportion
    of total variance attributable to between-group (OPO) differences:
    
        ICC = σ²_between / (σ²_between + σ²_within)
    
    A low ICC indicates that most variance occurs within OPOs, suggesting
    that hospital-level factors dominate OPO-level factors.
    """
    # Aggregate to hospital level
    hospital_stats = df.groupby(['opo', 'hospital_id']).agg({
        'approached': ['sum', 'count', 'mean'],
        'transplanted': ['sum', 'mean']
    })
    hospital_stats.columns = [
        'n_approached', 'n_referrals', 'approach_rate',
        'n_transplanted', 'transplant_rate'
    ]
    hospital_stats = hospital_stats[
        hospital_stats['n_referrals'] >= config.min_hospital_referrals
    ].reset_index()
    
    # Compute variance components
    total_var = hospital_stats['approach_rate'].var()
    opo_means = hospital_stats.groupby('opo')['approach_rate'].mean()
    between_var = opo_means.var()
    within_var = hospital_stats.groupby('opo')['approach_rate'].var().mean()
    
    # Compute ICC
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
    
    result = VarianceDecomposition(
        total_variance=total_var,
        between_opo_variance=between_var,
        within_opo_variance=within_var,
        icc=icc,
        n_hospitals=len(hospital_stats),
        n_opos=hospital_stats['opo'].nunique()
    )
    
    return result, hospital_stats


def compute_within_opo_gaps(
    hospital_stats: pd.DataFrame,
    config: AnalysisConfig = CONFIG
) -> pd.DataFrame:
    """
    Compute performance gaps between best and worst hospitals within each OPO.
    
    Parameters
    ----------
    hospital_stats : pd.DataFrame
        Hospital-level statistics with approach rates.
    config : AnalysisConfig
        Configuration parameters.
    
    Returns
    -------
    pd.DataFrame
        Within-OPO gap statistics for each OPO.
    """
    gaps = []
    
    for opo in hospital_stats['opo'].unique():
        opo_hospitals = hospital_stats[hospital_stats['opo'] == opo]
        
        if len(opo_hospitals) < config.min_hospitals_per_opo:
            continue
        
        best_rate = opo_hospitals['approach_rate'].max()
        worst_rate = opo_hospitals['approach_rate'].min()
        gap = best_rate - worst_rate
        
        # Compute potential additional approaches if all at best rate
        potential = (best_rate * opo_hospitals['n_referrals']).sum()
        actual = opo_hospitals['n_approached'].sum()
        missed = potential - actual
        
        gaps.append({
            'opo': opo,
            'n_hospitals': len(opo_hospitals),
            'best_rate': best_rate,
            'worst_rate': worst_rate,
            'gap': gap,
            'potential_additional_approaches': missed
        })
    
    return pd.DataFrame(gaps)


# =============================================================================
# Falsification Test
# =============================================================================

def run_outcome_falsification_test(
    df: pd.DataFrame,
    hospital_stats: pd.DataFrame
) -> Dict:
    """
    Test whether high-approach hospitals have worse conversion outcomes.
    
    This falsification test distinguishes between:
    - Selection bias: High-approach hospitals approach unsuitable candidates
    - Coordination constraint: High-approach hospitals capture more suitable candidates
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient-level data.
    hospital_stats : pd.DataFrame
        Hospital-level statistics with approach rates.
    
    Returns
    -------
    dict
        Test results including correlation and regression coefficients.
    
    Notes
    -----
    If the coefficient on hospital approach rate is non-negative (or not
    significantly negative), this supports the coordination constraint
    hypothesis over the selection bias hypothesis.
    """
    # Merge hospital rates to patient data
    hospital_rates = hospital_stats[['opo', 'hospital_id', 'approach_rate']].copy()
    hospital_rates = hospital_rates.rename(columns={'approach_rate': 'hospital_approach_rate'})
    
    merged = df.merge(hospital_rates, on=['opo', 'hospital_id'], how='inner')
    approached_patients = merged[merged['approached'] == 1].copy()
    
    # Compute hospital-level correlation
    hospital_outcomes = approached_patients.groupby('hospital_id').agg({
        'hospital_approach_rate': 'first',
        'transplanted': 'mean'
    })
    correlation = hospital_outcomes['hospital_approach_rate'].corr(
        hospital_outcomes['transplanted']
    )
    
    # Regression with controls
    y = approached_patients['transplanted'].astype(float).values
    X = sm.add_constant(
        approached_patients[['hospital_approach_rate', 'is_dbd', 'age']]
        .astype(float).fillna(0)
    )
    model = sm.OLS(y, X).fit()
    
    coef = model.params['hospital_approach_rate']
    pval = model.pvalues['hospital_approach_rate']
    
    # Interpretation
    if coef >= 0 or pval > 0.05:
        interpretation = "No evidence of adverse selection"
    else:
        interpretation = "Possible adverse selection concern"
    
    return {
        'correlation': correlation,
        'regression_coefficient': coef,
        'regression_pvalue': pval,
        'interpretation': interpretation,
        'n_approached_patients': len(approached_patients)
    }


# =============================================================================
# Instrumental Variable Analysis
# =============================================================================

def run_hospital_iv_analysis(
    df: pd.DataFrame,
    hospital_stats: pd.DataFrame
) -> IVEstimate:
    """
    Estimate causal effect of approach on transplantation using hospital IV.
    
    Uses hospital-level approach rate as an instrument for individual
    approach decisions, controlling for OPO fixed effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient-level data.
    hospital_stats : pd.DataFrame
        Hospital-level statistics with approach rates.
    
    Returns
    -------
    IVEstimate
        IV estimation results.
    
    Notes
    -----
    Identification assumption: Patients do not select hospitals based on
    unobserved factors correlated with both approach probability and
    transplant suitability.
    """
    # Merge hospital rates
    hospital_rates = hospital_stats[['opo', 'hospital_id', 'approach_rate']].rename(
        columns={'approach_rate': 'hospital_approach_rate'}
    )
    
    merged = df.merge(hospital_rates, on=['opo', 'hospital_id'], how='inner')
    
    # Create OPO fixed effects
    merged = pd.get_dummies(merged, columns=['opo'], prefix='opo', drop_first=True)
    opo_cols = [c for c in merged.columns if c.startswith('opo_')]
    
    # Prepare analysis variables
    cols = ['transplanted', 'approached', 'hospital_approach_rate', 'age', 'is_dbd'] + opo_cols
    analysis_data = merged[cols].dropna().astype(float)
    
    y = analysis_data['transplanted'].values
    approached = analysis_data['approached'].values
    instrument = analysis_data['hospital_approach_rate'].values
    controls = analysis_data[['age', 'is_dbd'] + opo_cols].values
    
    n = len(analysis_data)
    
    # First stage: approached ~ instrument + controls
    X_first = sm.add_constant(np.column_stack([instrument, controls]))
    first_stage = sm.OLS(approached, X_first).fit()
    f_stat = first_stage.tvalues[1] ** 2
    
    # Reduced form: transplanted ~ instrument + controls
    reduced_form = sm.OLS(y, X_first).fit()
    
    # IV estimate: reduced form / first stage
    iv_est = reduced_form.params[1] / first_stage.params[1]
    iv_se = abs(reduced_form.bse[1] / first_stage.params[1])
    
    # OLS for comparison
    X_ols = sm.add_constant(np.column_stack([approached, controls]))
    ols = sm.OLS(y, X_ols).fit()
    
    return IVEstimate(
        n_observations=n,
        first_stage_f=f_stat,
        ols_estimate=ols.params[1],
        ols_se=ols.bse[1],
        iv_estimate=iv_est,
        iv_se=iv_se
    )


# =============================================================================
# Main Analysis
# =============================================================================

def main(config: AnalysisConfig = CONFIG) -> Dict:
    """
    Execute complete hospital-level variance analysis.
    
    Parameters
    ----------
    config : AnalysisConfig
        Configuration parameters.
    
    Returns
    -------
    dict
        Complete analysis results.
    """
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading and preparing data...")
    df = load_and_prepare_data(config)
    print(f"  MSC cohort: {len(df):,} referrals")
    print(f"  OPOs: {df['opo'].nunique()}")
    print(f"  Hospitals: {df['hospital_id'].nunique()}")
    
    # Variance decomposition
    print("\nComputing variance decomposition...")
    var_decomp, hospital_stats = compute_variance_decomposition(df, config)
    print(f"  ICC: {var_decomp.icc:.3f}")
    print(f"  Between-OPO: {var_decomp.pct_between:.1f}%")
    print(f"  Within-OPO: {var_decomp.pct_within:.1f}%")
    
    # Within-OPO gaps
    print("\nComputing within-OPO gaps...")
    gaps = compute_within_opo_gaps(hospital_stats, config)
    print(f"  Average gap: {gaps['gap'].mean():.1%}")
    
    # Falsification test
    print("\nRunning outcome falsification test...")
    falsification = run_outcome_falsification_test(df, hospital_stats)
    print(f"  Correlation: {falsification['correlation']:.3f}")
    print(f"  Interpretation: {falsification['interpretation']}")
    
    # IV analysis
    print("\nRunning IV analysis...")
    iv_result = run_hospital_iv_analysis(df, hospital_stats)
    print(f"  First-stage F: {iv_result.first_stage_f:.1f} ({iv_result.instrument_strength})")
    print(f"  IV estimate: {iv_result.iv_estimate:.4f}")
    print(f"  95% CI: ({iv_result.iv_ci_lower:.4f}, {iv_result.iv_ci_upper:.4f})")
    
    # Save outputs
    print("\nSaving outputs...")
    hospital_stats.to_csv(config.output_dir / 'hospital_approach_rates.csv', index=False)
    gaps.to_csv(config.output_dir / 'within_opo_gaps.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"""
Variance Decomposition:
  ICC = {var_decomp.icc:.3f}
  {var_decomp.pct_within:.0f}% of variance is within-OPO (between hospitals)

Within-OPO Performance Gaps:
  Average gap (best vs worst hospital): {gaps['gap'].mean():.1%}
  Total potential additional approaches: {gaps['potential_additional_approaches'].sum():,.0f}

Falsification Test:
  {falsification['interpretation']}

Instrumental Variable Analysis:
  First-stage F: {iv_result.first_stage_f:.1f}
  IV estimate: {iv_result.iv_estimate:.4f} (95% CI: {iv_result.iv_ci_lower:.4f} to {iv_result.iv_ci_upper:.4f})
""")
    
    return {
        'variance_decomposition': var_decomp,
        'hospital_stats': hospital_stats,
        'gaps': gaps,
        'falsification_test': falsification,
        'iv_result': iv_result
    }


if __name__ == "__main__":
    results = main()
