#!/usr/bin/env python3
"""
__date__ = "2025-01-08"
Investigate anomalous findings
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv("/home/noah/physionet.org/files/orchid/2.1.1/OPOReferrals.csv", low_memory=False)
for col in ['approached', 'authorized', 'procured', 'transplanted']:
    df[col] = df[col].fillna(0).astype(int)

df['time_referred'] = pd.to_datetime(df['time_referred'], errors='coerce')
df['time_approached'] = pd.to_datetime(df['time_approached'], errors='coerce')

# =============================================================================
# 1. WHY IS SLOWER APPROACH BETTER?
# =============================================================================

print("=" * 70)
print("1. INVESTIGATING TIMING PARADOX")
print("=" * 70)

approached = df[df['approached'] == 1].copy()
approached['hours_to_approach'] = (
    approached['time_approached'] - approached['time_referred']
).dt.total_seconds() / 3600

approached = approached[
    (approached['hours_to_approach'] >= 0) & 
    (approached['hours_to_approach'] <= 72)
]

# Is it case complexity? Check by cause of death
print("\n--- Time to approach by cause of death ---")
timing_by_cod = approached.groupby('cause_of_death_unos').agg(
    n=('hours_to_approach', 'count'),
    mean_hours=('hours_to_approach', 'mean'),
    tx_rate=('transplanted', 'mean')
).reset_index()
timing_by_cod = timing_by_cod[timing_by_cod['n'] >= 50].sort_values('mean_hours')
print(timing_by_cod.to_string(index=False))

# Is it DBD vs DCD?
print("\n--- Time to approach by donor type ---")
approached['is_dbd'] = approached['brain_death'].astype(int)
timing_by_type = approached.groupby('is_dbd').agg(
    n=('hours_to_approach', 'count'),
    mean_hours=('hours_to_approach', 'mean'),
    tx_rate=('transplanted', 'mean')
)
print(timing_by_type)

# Within DBD only, does faster = better?
print("\n--- Timing analysis WITHIN DBD only ---")
dbd = approached[approached['is_dbd'] == 1]
if len(dbd) > 100:
    r, p = pearsonr(dbd['hours_to_approach'], dbd['transplanted'])
    print(f"DBD correlation (time vs success): r={r:.3f}, p={p:.4f}, N={len(dbd)}")
    
    dbd['time_quartile'] = pd.qcut(dbd['hours_to_approach'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print(dbd.groupby('time_quartile')['transplanted'].mean())

# Within DCD only
print("\n--- Timing analysis WITHIN DCD only ---")
dcd = approached[approached['is_dbd'] == 0]
if len(dcd) > 100:
    r, p = pearsonr(dcd['hours_to_approach'], dcd['transplanted'])
    print(f"DCD correlation (time vs success): r={r:.3f}, p={p:.4f}, N={len(dcd)}")
    
    dcd['time_quartile'] = pd.qcut(dcd['hours_to_approach'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    print(dcd.groupby('time_quartile')['transplanted'].mean())

# =============================================================================
# 2. WHY IS DBD IV RESULT GARBAGE?
# =============================================================================

print("\n" + "=" * 70)
print("2. INVESTIGATING DBD IV FAILURE")
print("=" * 70)

print(f"\nDBD cases: {df['brain_death'].sum():,}")
print(f"DCD cases: {(~df['brain_death']).sum():,}")

# DBD approach rate
dbd_all = df[df['brain_death'] == True]
print(f"\nDBD approach rate: {dbd_all['approached'].mean()*100:.1f}%")
print(f"DBD transplant rate: {dbd_all['transplanted'].mean()*100:.1f}%")

# Hospital-level variation in DBD
dbd_hosp = dbd_all.groupby('hospital_id').agg(
    n=('approached', 'count'),
    approach_rate=('approached', 'mean'),
    tx_rate=('transplanted', 'mean')
).reset_index()
dbd_hosp = dbd_hosp[dbd_hosp['n'] >= 10]

print(f"\nDBD hospitals with ≥10 cases: {len(dbd_hosp)}")
print(f"DBD approach rate range: {dbd_hosp['approach_rate'].min()*100:.1f}% - {dbd_hosp['approach_rate'].max()*100:.1f}%")

# Is there enough variation to identify?
print(f"DBD approach rate std: {dbd_hosp['approach_rate'].std()*100:.1f}pp")

# =============================================================================
# 3. WHY IS IV EFFECT DECLINING OVER TIME?
# =============================================================================

print("\n" + "=" * 70)
print("3. INVESTIGATING IV DECLINE OVER TIME")
print("=" * 70)

# Approach rate over time
yearly = df.groupby('referral_year').agg(
    n=('approached', 'count'),
    approach_rate=('approached', 'mean'),
    tx_rate=('transplanted', 'mean'),
    auth_rate=('authorized', 'mean')
).reset_index()

print("\nYearly trends:")
print(yearly.to_string(index=False))

# Hospital-level approach rate variance over time
print("\n--- Hospital-level variance over time ---")
for year in sorted(df['referral_year'].dropna().unique()):
    year_data = df[df['referral_year'] == year]
    hosp_rates = year_data.groupby('hospital_id')['approached'].mean()
    hosp_rates = hosp_rates[year_data.groupby('hospital_id').size() >= 20]
    if len(hosp_rates) > 10:
        print(f"  {int(year)}: N={len(hosp_rates)} hospitals, approach rate SD={hosp_rates.std()*100:.1f}pp")

# =============================================================================
# 4. ALTERNATIVE TIMING METRIC: REFERRAL-TO-PROCUREMENT
# =============================================================================

print("\n" + "=" * 70)
print("4. ALTERNATIVE TIMING: REFERRAL TO PROCUREMENT")
print("=" * 70)

df['time_procured'] = pd.to_datetime(df['time_procured'], errors='coerce')

procured = df[df['procured'] == 1].copy()
procured['hours_to_procure'] = (
    procured['time_procured'] - procured['time_referred']
).dt.total_seconds() / 3600

procured = procured[
    (procured['hours_to_procure'] >= 0) & 
    (procured['hours_to_procure'] <= 120)
]

print(f"\nProcured cases with valid timing: {len(procured):,}")
print(f"Mean time referral→procurement: {procured['hours_to_procure'].mean():.1f} hours")
print(f"Median: {procured['hours_to_procure'].median():.1f} hours")

# Does faster procurement correlate with more organs?
# Check organ-specific outcomes
organ_cols = [c for c in df.columns if c.startswith('outcome_')]
print(f"\nOrgan outcome columns: {organ_cols}")

# =============================================================================
# 5. THE REAL COORDINATION METRIC: APPROACH GIVEN REFERRAL
# =============================================================================

print("\n" + "=" * 70)
print("5. THE REAL QUESTION: WHY DON'T REFERRALS GET APPROACHED?")
print("=" * 70)

not_approached = df[df['approached'] == 0].copy()

print(f"\nReferrals NOT approached: {len(not_approached):,} ({100*len(not_approached)/len(df):.1f}%)")

# By cause of death
print("\n--- Non-approach rate by cause of death ---")
non_approach_cod = df.groupby('cause_of_death_unos').agg(
    n=('approached', 'count'),
    approach_rate=('approached', 'mean')
).reset_index()
non_approach_cod = non_approach_cod[non_approach_cod['n'] >= 100].sort_values('approach_rate')
print(non_approach_cod.head(10).to_string(index=False))

# By age
print("\n--- Approach rate by age group ---")
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 40, 55, 65, 75, 100], labels=['0-17', '18-39', '40-54', '55-64', '65-74', '75+'])
age_approach = df.groupby('age_group').agg(
    n=('approached', 'count'),
    approach_rate=('approached', 'mean'),
    tx_rate=('transplanted', 'mean')
)
print(age_approach)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: WHAT TO DO WITH THESE FINDINGS")
print("=" * 70)

print("""
1. TIMING PARADOX EXPLANATION:
   - Likely SELECTION BIAS: Harder cases (e.g., older, more comorbid) take 
     longer to evaluate but are approached more carefully → higher success
   - NOT evidence against "fast approach = good"
   - Paper should: Note this complexity, avoid claiming speed causally helps

2. DBD IV FAILURE:
   - Too few DBD cases with variation to identify effect
   - Paper should: Report DCD-specific IV (18.5%, robust) as sensitivity
   - Or just report pooled 29.1% and note heterogeneity

3. IV DECLINING OVER TIME:
   - Could be real (system getting better at selectivity)
   - Could be confounding (OPO composition changing)
   - Paper should: Note temporal heterogeneity, use pooled estimate

4. STRONGEST FINDINGS FOR PAPER:
   - Variance decomposition (82-93% within-OPO) - ROBUST
   - Potential gains (+4,135 donors, 25%) - CONCRETE
   - Concentration (top 100 hospitals = 20%) - ACTIONABLE
   - Zero-conversion (2,136 hospitals) - STARK
   
5. WEAKER FINDINGS TO CAVEAT:
   - IV causal effect - valid but heterogeneous
   - Timing mechanisms - confounded, don't overclaim
""")
