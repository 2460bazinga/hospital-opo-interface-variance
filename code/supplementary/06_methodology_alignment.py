#!/usr/bin/env python3
"""
__date__ = "2025-01-08"
Apples-to-apples comparison: Use ALL referrals in both datasets
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"

import pandas as pd
import numpy as np

print("=" * 70)
print("APPLES-TO-APPLES: ALL REFERRALS (NO MSC FILTER)")
print("=" * 70)

# =============================================================================
# ORCHID - ALL REFERRALS (no MSC filter)
# =============================================================================

df = pd.read_csv("/home/noah/physionet.org/files/orchid/2.1.1/OPOReferrals.csv", low_memory=False)

for col in ['approached', 'authorized', 'procured', 'transplanted']:
    df[col] = df[col].fillna(0).astype(int)

print(f"\nORCHID - ALL REFERRALS (N = {len(df):,})")
print(f"  Approached:   {df['approached'].sum():,} ({100*df['approached'].mean():.2f}%)")
print(f"  Procured:     {df['procured'].sum():,} ({100*df['procured'].mean():.2f}%)")
print(f"  Transplanted: {df['transplanted'].sum():,} ({100*df['transplanted'].mean():.2f}%)")

# Hospital-level aggregation (ALL referrals)
hosp = df.groupby(['opo', 'hospital_id']).agg(
    approach_rate=('approached', 'mean'),
    procured_rate=('procured', 'mean'),
    tx_rate=('transplanted', 'mean'),
    n=('approached', 'count')
).reset_index()

hosp = hosp[hosp['n'] >= 20]

print(f"\nHospitals with ≥20 referrals: {len(hosp)}")
print(f"OPOs: {hosp['opo'].nunique()}")

# ICC for procured rate (closest to OSR "donor")
between = hosp.groupby('opo')['procured_rate'].mean().var()
within = hosp.groupby('opo')['procured_rate'].var().mean()
icc_procured = between / (between + within)

print(f"\nORCHID Procured Rate (all referrals):")
print(f"  Mean rate: {hosp['procured_rate'].mean()*100:.2f}%")
print(f"  ICC: {icc_procured:.3f}")
print(f"  Between-OPO: {icc_procured*100:.1f}%")
print(f"  Within-OPO: {(1-icc_procured)*100:.1f}%")

# Within-OPO gaps
print(f"\nWithin-OPO hospital variation (procured rate):")
for opo in sorted(hosp['opo'].unique()):
    d = hosp[hosp['opo'] == opo]['procured_rate']
    print(f"  {opo}: mean={d.mean()*100:.1f}%, min={d.min()*100:.1f}%, max={d.max()*100:.1f}%, range={(d.max()-d.min())*100:.1f}pp")

# =============================================================================
# OSR 2024
# =============================================================================

print("\n" + "=" * 70)
print("OSR 2024 - ALL REFERRALS")
print("=" * 70)

osr = pd.read_excel('/home/noah/OSR_final_tables2505.xlsx', sheet_name='Table B1')
osr = osr.rename(columns={
    'OPO code': 'opo',
    'Referrals': 'referrals',
    'Total donors': 'donors'
})
osr['referrals'] = pd.to_numeric(osr['referrals'], errors='coerce')
osr['donors'] = pd.to_numeric(osr['donors'], errors='coerce').fillna(0)
osr = osr[osr['referrals'] >= 20].copy()
osr['donor_rate'] = osr['donors'] / osr['referrals']

print(f"\nOSR 2024 (N = {len(osr):,} hospitals)")
print(f"  Total referrals: {osr['referrals'].sum():,.0f}")
print(f"  Total donors: {osr['donors'].sum():,.0f}")
print(f"  Mean donor rate: {osr['donor_rate'].mean()*100:.2f}%")

# ICC
osr_between = osr.groupby('opo')['donor_rate'].mean().var()
osr_within = osr.groupby('opo')['donor_rate'].var().mean()
osr_icc = osr_between / (osr_between + osr_within)

print(f"\nOSR Donor Rate:")
print(f"  ICC: {osr_icc:.3f}")
print(f"  Between-OPO: {osr_icc*100:.1f}%")
print(f"  Within-OPO: {(1-osr_icc)*100:.1f}%")

# =============================================================================
# SIDE BY SIDE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: ORCHID vs OSR (Both using ALL referrals)")
print("=" * 70)

print(f"""
                          ORCHID (2015-2021)    OSR 2024
                          ------------------    --------
Hospitals analyzed:       {len(hosp):>10}          {len(osr):>10}
OPOs:                     {hosp['opo'].nunique():>10}          {osr['opo'].nunique():>10}
Total referrals:          {df.shape[0]:>10,}       {osr['referrals'].sum():>10,.0f}

Outcome measure:          Procured rate         Donor rate
Mean rate:                {hosp['procured_rate'].mean()*100:>9.2f}%         {osr['donor_rate'].mean()*100:>9.2f}%

ICC:                      {icc_procured:>10.3f}          {osr_icc:>10.3f}
Between-OPO variance:     {icc_procured*100:>9.1f}%          {osr_icc*100:>9.1f}%
Within-OPO variance:      {(1-icc_procured)*100:>9.1f}%          {(1-osr_icc)*100:>9.1f}%
""")

# =============================================================================
# THE REAL ISSUE: N=6 OPOs vs N=55 OPOs
# =============================================================================

print("\n" + "=" * 70)
print("THE REAL ISSUE: SAMPLE SIZE")
print("=" * 70)

print(f"""
ORCHID has only 6 OPOs. One outlier (OPO6) dominates the between-OPO variance.

OPO6 procured rate vs others:
""")

for opo in sorted(hosp['opo'].unique()):
    d = hosp[hosp['opo'] == opo]['procured_rate']
    print(f"  {opo}: {d.mean()*100:.1f}%")

print(f"""
OSR has 55 OPOs - no single outlier dominates.

With N=6:
  - One extreme OPO can make ICC look like OPO-level dominates
  - Statistical power to detect within-OPO variance is limited

With N=55:
  - Outliers are diluted
  - Within-OPO variance emerges as dominant pattern

THE WITHIN-OPO GAPS ARE REAL IN BOTH DATASETS:
  ORCHID: avg {hosp.groupby('opo')['procured_rate'].apply(lambda x: x.max()-x.min()).mean()*100:.1f}pp range within each OPO
  OSR 2024: avg {osr.groupby('opo')['donor_rate'].apply(lambda x: x.max()-x.min()).mean()*100:.1f}pp range within each OPO
""")
