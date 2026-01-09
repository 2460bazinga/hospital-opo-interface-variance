#!/usr/bin/env python3
"""
__date__ = "2025-01-08"
Create publication-ready figures for within-OPO variance analysis
"""

__version__ = "1.0.0"
__author__ = "Noah Parrish"
__date__ = "2025-01-08"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

output_dir = Path('./variance_figures')
output_dir.mkdir(exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

# ORCHID
df = pd.read_csv("/home/noah/physionet.org/files/orchid/2.1.1/OPOReferrals.csv", low_memory=False)
df['procured'] = df['procured'].fillna(0).astype(int)

orchid_hosp = df.groupby(['opo', 'hospital_id']).agg(
    procured_rate=('procured', 'mean'),
    n=('procured', 'count')
).reset_index()
orchid_hosp = orchid_hosp[orchid_hosp['n'] >= 20]

# OSR 2024
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

# =============================================================================
# FIGURE 1: ICC COMPARISON BAR CHART
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

datasets = ['ORCHID\n(2015-2021)', 'OSR National\n(2024)']
within_opo = [82.6, 93.1]
between_opo = [17.4, 6.9]

x = np.arange(len(datasets))
width = 0.6

# Stacked bar
bars1 = ax.bar(x, within_opo, width, label='Within-OPO (Hospital-level)', color='#2c7bb6')
bars2 = ax.bar(x, between_opo, width, bottom=within_opo, label='Between-OPO', color='#d7191c')

# Labels on bars
for i, (w, b) in enumerate(zip(within_opo, between_opo)):
    ax.text(i, w/2, f'{w:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(i, w + b/2, f'{b:.1f}%', ha='center', va='center', fontsize=11, color='white')

ax.set_ylabel('Percentage of Total Variance')
ax.set_title('Variance Decomposition: Where Does Performance Heterogeneity Originate?')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', frameon=False)

# Add ICC annotations
ax.annotate('ICC = 0.174', xy=(0, 102), ha='center', fontsize=10, style='italic')
ax.annotate('ICC = 0.069', xy=(1, 102), ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(output_dir / 'Figure1_ICC_Comparison.png', dpi=300)
plt.savefig(output_dir / 'Figure1_ICC_Comparison.pdf')
plt.close()
print("✓ Figure 1: ICC Comparison")

# =============================================================================
# FIGURE 2: ORCHID WITHIN-OPO HOSPITAL VARIATION (STRIP PLOT)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

opos = sorted(orchid_hosp['opo'].unique())
colors = plt.cm.Set2(np.linspace(0, 1, len(opos)))

for i, opo in enumerate(opos):
    opo_data = orchid_hosp[orchid_hosp['opo'] == opo]['procured_rate'] * 100
    
    # Jittered strip
    jitter = np.random.normal(0, 0.08, len(opo_data))
    ax.scatter([i + jitter[j] for j in range(len(opo_data))], opo_data, 
               alpha=0.6, s=40, color=colors[i], edgecolor='white', linewidth=0.5)
    
    # Mean marker
    ax.scatter(i, opo_data.mean(), s=200, color=colors[i], marker='D', 
               edgecolor='black', linewidth=1.5, zorder=5)
    
    # Range line
    ax.plot([i, i], [opo_data.min(), opo_data.max()], color='black', 
            linewidth=2, alpha=0.3, zorder=1)

ax.set_xticks(range(len(opos)))
ax.set_xticklabels(opos, fontsize=11)
ax.set_xlabel('OPO')
ax.set_ylabel('Hospital Procurement Rate (%)')
ax.set_title('ORCHID: Hospital-Level Procurement Rates Within Each OPO\n(Each dot = one hospital, diamond = OPO mean)')

# Add gap annotations
for i, opo in enumerate(opos):
    opo_data = orchid_hosp[orchid_hosp['opo'] == opo]['procured_rate'] * 100
    gap = opo_data.max() - opo_data.min()
    ax.annotate(f'{gap:.0f}pp\ngap', xy=(i, opo_data.max() + 1), ha='center', 
                fontsize=9, color='gray')

ax.set_ylim(-1, orchid_hosp['procured_rate'].max() * 100 + 8)

plt.tight_layout()
plt.savefig(output_dir / 'Figure2_ORCHID_Within_OPO_Variation.png', dpi=300)
plt.savefig(output_dir / 'Figure2_ORCHID_Within_OPO_Variation.pdf')
plt.close()
print("✓ Figure 2: ORCHID Within-OPO Variation")

# =============================================================================
# FIGURE 3: OSR 2024 WITHIN-OPO VARIATION (BOX PLOT - SAMPLE OF OPOs)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Get OPOs with enough hospitals for meaningful boxes
opo_counts = osr.groupby('opo').size()
opos_with_data = opo_counts[opo_counts >= 20].index.tolist()[:20]  # Top 20 by hospital count

box_data = []
labels = []
for opo in opos_with_data:
    opo_data = osr[osr['opo'] == opo]['donor_rate'] * 100
    box_data.append(opo_data.values)
    labels.append(opo)

bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.5))

# Color boxes
colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(box_data)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('OPO')
ax.set_ylabel('Hospital Donor Rate (%)')
ax.set_title('OSR 2024: Hospital-Level Donor Rate Variation Within OPOs\n(20 largest OPOs by hospital count)')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'Figure3_OSR_Within_OPO_Boxplot.png', dpi=300)
plt.savefig(output_dir / 'Figure3_OSR_Within_OPO_Boxplot.pdf')
plt.close()
print("✓ Figure 3: OSR Within-OPO Box Plot")

# =============================================================================
# FIGURE 4: HISTOGRAM OF WITHIN-OPO GAPS
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ORCHID gaps
orchid_gaps = orchid_hosp.groupby('opo')['procured_rate'].apply(lambda x: (x.max() - x.min()) * 100)

ax = axes[0]
ax.bar(range(len(orchid_gaps)), orchid_gaps.values, color='#2c7bb6', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(orchid_gaps)))
ax.set_xticklabels(orchid_gaps.index, fontsize=10)
ax.set_xlabel('OPO')
ax.set_ylabel('Within-OPO Gap (percentage points)')
ax.set_title(f'ORCHID: Within-OPO Range\n(Mean gap: {orchid_gaps.mean():.1f}pp)')
ax.axhline(orchid_gaps.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {orchid_gaps.mean():.1f}pp')
ax.legend(frameon=False)

# OSR gaps
osr_gaps = osr.groupby('opo')['donor_rate'].apply(lambda x: (x.max() - x.min()) * 100)

ax = axes[1]
ax.hist(osr_gaps, bins=20, color='#d7191c', alpha=0.7, edgecolor='black')
ax.axvline(osr_gaps.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {osr_gaps.mean():.1f}pp')
ax.set_xlabel('Within-OPO Gap (percentage points)')
ax.set_ylabel('Number of OPOs')
ax.set_title(f'OSR 2024: Distribution of Within-OPO Gaps\n(n=55 OPOs)')
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(output_dir / 'Figure4_Within_OPO_Gaps.png', dpi=300)
plt.savefig(output_dir / 'Figure4_Within_OPO_Gaps.pdf')
plt.close()
print("✓ Figure 4: Within-OPO Gaps")

# =============================================================================
# FIGURE 5: SIDE-BY-SIDE DENSITY COMPARISON
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ORCHID
ax = axes[0]
for opo in sorted(orchid_hosp['opo'].unique()):
    opo_data = orchid_hosp[orchid_hosp['opo'] == opo]['procured_rate'] * 100
    ax.hist(opo_data, bins=15, alpha=0.5, label=opo, density=True)

ax.set_xlabel('Hospital Procurement Rate (%)')
ax.set_ylabel('Density')
ax.set_title('ORCHID: Distribution of Hospital Rates by OPO')
ax.legend(title='OPO', frameon=False, fontsize=9)

# OSR - sample 6 OPOs for comparison
ax = axes[1]
sample_opos = osr.groupby('opo').size().nlargest(6).index.tolist()
for opo in sample_opos:
    opo_data = osr[osr['opo'] == opo]['donor_rate'] * 100
    ax.hist(opo_data, bins=15, alpha=0.5, label=opo, density=True)

ax.set_xlabel('Hospital Donor Rate (%)')
ax.set_ylabel('Density')
ax.set_title('OSR 2024: Distribution of Hospital Rates\n(6 largest OPOs)')
ax.legend(title='OPO', frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'Figure5_Rate_Distributions.png', dpi=300)
plt.savefig(output_dir / 'Figure5_Rate_Distributions.pdf')
plt.close()
print("✓ Figure 5: Rate Distributions")

# =============================================================================
# FIGURE 6: KEY FINDING SUMMARY INFOGRAPHIC
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Where Does Performance Variance Originate?', 
        ha='center', va='top', fontsize=18, fontweight='bold')
ax.text(0.5, 0.89, 'Hospital-OPO Interface is the Primary Locus of Heterogeneity',
        ha='center', va='top', fontsize=14, style='italic', color='gray')

# Two columns
# ORCHID
ax.add_patch(plt.Rectangle((0.05, 0.35), 0.4, 0.45, facecolor='#e6f2ff', edgecolor='#2c7bb6', linewidth=2))
ax.text(0.25, 0.77, 'ORCHID\n2015-2021', ha='center', va='top', fontsize=14, fontweight='bold')
ax.text(0.25, 0.65, '6 OPOs\n343 Hospitals\n133,101 Referrals', ha='center', va='top', fontsize=11)

# Pie chart for ORCHID
pie_ax1 = fig.add_axes([0.1, 0.4, 0.25, 0.2])
pie_ax1.pie([82.6, 17.4], labels=['Within-OPO\n82.6%', 'Between-OPO\n17.4%'], 
            colors=['#2c7bb6', '#d7191c'], autopct='', startangle=90,
            textprops={'fontsize': 9})
pie_ax1.set_title('ICC = 0.174', fontsize=10)

# OSR
ax.add_patch(plt.Rectangle((0.55, 0.35), 0.4, 0.45, facecolor='#ffe6e6', edgecolor='#d7191c', linewidth=2))
ax.text(0.75, 0.77, 'OSR National\n2024', ha='center', va='top', fontsize=14, fontweight='bold')
ax.text(0.75, 0.65, '55 OPOs\n4,140 Hospitals\n1.15M Referrals', ha='center', va='top', fontsize=11)

# Pie chart for OSR
pie_ax2 = fig.add_axes([0.6, 0.4, 0.25, 0.2])
pie_ax2.pie([93.1, 6.9], labels=['Within-OPO\n93.1%', 'Between-OPO\n6.9%'], 
            colors=['#2c7bb6', '#d7191c'], autopct='', startangle=90,
            textprops={'fontsize': 9})
pie_ax2.set_title('ICC = 0.069', fontsize=10)

# Bottom key finding
ax.add_patch(plt.Rectangle((0.1, 0.08), 0.8, 0.2, facecolor='#f0f0f0', edgecolor='black', linewidth=1))
ax.text(0.5, 0.24, 'KEY FINDING', ha='center', va='top', fontsize=12, fontweight='bold')
ax.text(0.5, 0.17, '80-93% of variance occurs WITHIN OPOs\n(between hospitals served by the same OPO)',
        ha='center', va='top', fontsize=12)
ax.text(0.5, 0.1, 'The hospital-OPO interface, not the OPO itself,\nis the primary determinant of performance.',
        ha='center', va='top', fontsize=11, style='italic', color='#444444')

plt.savefig(output_dir / 'Figure6_Summary_Infographic.png', dpi=300)
plt.savefig(output_dir / 'Figure6_Summary_Infographic.pdf')
plt.close()
print("✓ Figure 6: Summary Infographic")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FIGURES CREATED")
print("=" * 60)
print(f"""
Output directory: {output_dir}

1. Figure1_ICC_Comparison.png/pdf
   - Stacked bar showing variance decomposition for both datasets

2. Figure2_ORCHID_Within_OPO_Variation.png/pdf  
   - Strip plot showing hospital-level variation within each ORCHID OPO

3. Figure3_OSR_Within_OPO_Boxplot.png/pdf
   - Box plots showing within-OPO variation for 20 largest OSR OPOs

4. Figure4_Within_OPO_Gaps.png/pdf
   - Bar chart (ORCHID) and histogram (OSR) of within-OPO gaps

5. Figure5_Rate_Distributions.png/pdf
   - Overlapping histograms showing rate distributions by OPO

6. Figure6_Summary_Infographic.png/pdf
   - Visual summary of key findings with pie charts
""")
