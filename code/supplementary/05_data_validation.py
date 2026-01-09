#!/usr/bin/env python3
"""
__date__ = "2025-01-08"
Check what timing info is usable after de-identification
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"

import pandas as pd
import numpy as np

df = pd.read_csv("/home/noah/physionet.org/files/orchid/2.1.1/OPOReferrals.csv", low_memory=False)

print("=" * 70)
print("CHECKING DE-IDENTIFICATION EFFECTS")
print("=" * 70)

# Parse times
for col in ['time_referred', 'time_approached', 'time_authorized', 'time_procured']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Check year distribution
print("\n--- Year Distribution (should be random if shifted) ---")
df['ref_year'] = df['time_referred'].dt.year
print(df['ref_year'].value_counts().sort_index())

# Check if referral_year column matches time_referred year
print("\n--- Does referral_year match time_referred year? ---")
df['year_match'] = df['referral_year'] == df['ref_year']
print(f"Match rate: {df['year_match'].mean()*100:.1f}%")

# Check relative timing (should be preserved)
print("\n--- Relative Timing: Referral → Approach (hours) ---")
approached = df[df['approached'] == 1].copy()
approached['hours_to_approach'] = (
    approached['time_approached'] - approached['time_referred']
).dt.total_seconds() / 3600

# Filter reasonable values
valid = approached[(approached['hours_to_approach'] >= 0) & (approached['hours_to_approach'] <= 168)]
print(f"Valid cases: {len(valid):,}")
print(f"Mean: {valid['hours_to_approach'].mean():.1f} hours")
print(f"Median: {valid['hours_to_approach'].median():.1f} hours")
print(f"Std: {valid['hours_to_approach'].std():.1f} hours")

# Check if relative timing varies by OPO (this should be real variation)
print("\n--- Mean hours to approach BY OPO ---")
opo_timing = valid.groupby('opo')['hours_to_approach'].agg(['mean', 'median', 'count'])
print(opo_timing)

# Check day of week - is it preserved or randomized?
print("\n--- Day of Week Distribution ---")
print(df['referral_day_of_week'].value_counts())

# Does day of week from time_referred match referral_day_of_week column?
df['dow_from_time'] = df['time_referred'].dt.day_name()
df['dow_match'] = df['dow_from_time'] == df['referral_day_of_week']
print(f"\nDay of week match rate: {df['dow_match'].mean()*100:.1f}%")

# Check approach-to-authorization timing
print("\n--- Relative Timing: Approach → Authorization (hours) ---")
authorized = df[(df['approached'] == 1) & (df['authorized'] == 1)].copy()
authorized['hours_approach_to_auth'] = (
    authorized['time_authorized'] - authorized['time_approached']
).dt.total_seconds() / 3600

valid_auth = authorized[
    (authorized['hours_approach_to_auth'] >= 0) & 
    (authorized['hours_approach_to_auth'] <= 72)
]
print(f"Valid cases: {len(valid_auth):,}")
print(f"Mean: {valid_auth['hours_approach_to_auth'].mean():.1f} hours")
print(f"Median: {valid_auth['hours_approach_to_auth'].median():.1f} hours")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 70)
print("CONCLUSION: WHAT TIMING DATA IS USABLE")
print("=" * 70)

print("""
USABLE (relative intervals preserved):
  - Hours from referral to approach
  - Hours from approach to authorization  
  - Hours from authorization to procurement
  - Day of week (if match rate is high)

NOT USABLE (randomized by de-identification):
  - Absolute years
  - Year-over-year trends
  - Seasonal patterns
  - Any analysis requiring real calendar time
""")
