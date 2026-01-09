#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Figure Generation: ICC Variance Decomposition

Publication-Ready Visualization of Variance Decomposition Findings
================================================================================

Description:
    This script generates a professional visualization comparing the variance
    decomposition findings between ORCHID (2015-2021) and OSR 2024 datasets.
    The figure illustrates the key finding that the majority of performance
    variance occurs within OPOs (at the hospital level), not between OPOs.

Key Statistics:
    ORCHID (2015-2021):
        - ICC = 0.174 (17.4% between-OPO, 82.6% within-OPO)
        - Outcome: Procurement rate
        - n = 343 hospitals, 6 OPOs
    
    OSR 2024 (National):
        - ICC = 0.069 (6.9% between-OPO, 93.1% within-OPO)
        - Outcome: Donor rate
        - n = 4,140 hospitals, 55 OPOs

Output:
    - figures/Figure1_ICC_Comparison.png (300 DPI)
    - figures/Figure1_ICC_Comparison.pdf (vector)

Author: Noah Parrish
Version: 2.0.0
Date: January 2026
License: MIT

Requirements:
    matplotlib>=3.5.0
    numpy>=1.21.0

Usage:
    python create_icc_figure.py
================================================================================
"""

__version__ = "2.0.0"
__author__ = "Noah Parrish"
__date__ = "January 2026"

# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = Path('./figures')

# Data: Corrected ICC values from standardized analysis
DATASETS = {
    'ORCHID\n(2015-2021)': {
        'within_opo': 82.6,
        'between_opo': 17.4,
        'icc': 0.174,
        'outcome': 'Procurement Rate',
        'n_hospitals': 343,
        'n_opos': 6
    },
    'OSR 2024\n(National)': {
        'within_opo': 93.1,
        'between_opo': 6.9,
        'icc': 0.069,
        'outcome': 'Donor Rate',
        'n_hospitals': 4140,
        'n_opos': 55
    }
}

# Color palette (colorblind-friendly)
COLORS = {
    'within': '#2E86AB',   # Blue for within-OPO (hospital-level)
    'between': '#A23B72',  # Magenta for between-OPO
    'text': '#333333',     # Dark gray for text
    'subtitle': '#555555'  # Medium gray for subtitles
}

# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_variance_decomposition_figure():
    """
    Create publication-ready variance decomposition figure.
    
    The figure consists of two panels:
    1. Stacked horizontal bar chart showing variance decomposition
    2. Donut charts for visual impact
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set up the figure with a clean, professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    dataset_names = list(DATASETS.keys())
    within_opo = [DATASETS[d]['within_opo'] for d in dataset_names]
    between_opo = [DATASETS[d]['between_opo'] for d in dataset_names]
    
    # ==========================================================================
    # Left Panel: Stacked Horizontal Bar Chart
    # ==========================================================================
    ax1 = axes[0]
    bar_width = 0.5
    y_pos = np.arange(len(dataset_names))
    
    # Create stacked bars
    bars_within = ax1.barh(
        y_pos, within_opo, bar_width,
        label='Within-OPO (Hospital-Level)',
        color=COLORS['within'],
        edgecolor='white',
        linewidth=2
    )
    bars_between = ax1.barh(
        y_pos, between_opo, bar_width,
        left=within_opo,
        label='Between-OPO',
        color=COLORS['between'],
        edgecolor='white',
        linewidth=2
    )
    
    # Add percentage labels on bars
    for i, (w, b) in enumerate(zip(within_opo, between_opo)):
        # Within-OPO label (centered in the within portion)
        ax1.text(
            w / 2, i, f'{w:.1f}%',
            ha='center', va='center',
            fontsize=16, fontweight='bold', color='white'
        )
        # Between-OPO label (centered in the between portion)
        ax1.text(
            w + b / 2, i, f'{b:.1f}%',
            ha='center', va='center',
            fontsize=14, fontweight='bold', color='white'
        )
    
    # Formatting
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(dataset_names, fontsize=12)
    ax1.set_xlabel('Percentage of Total Variance', fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_title(
        'Variance Decomposition by Dataset',
        fontsize=14, fontweight='bold', pad=15
    )
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=10,
        frameon=True
    )
    
    # Add ICC values as annotations
    for i, name in enumerate(dataset_names):
        icc = DATASETS[name]['icc']
        ax1.annotate(
            f'ICC = {icc:.3f}',
            xy=(100, i),
            xytext=(105, i),
            fontsize=10,
            va='center',
            color=COLORS['text']
        )
    
    # ==========================================================================
    # Right Panel: Donut Charts
    # ==========================================================================
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title(
        'Visual Comparison',
        fontsize=14, fontweight='bold', pad=15
    )
    
    # ORCHID donut chart
    orchid_ax = fig.add_axes([0.55, 0.30, 0.18, 0.45])
    orchid_data = DATASETS['ORCHID\n(2015-2021)']
    orchid_sizes = [orchid_data['within_opo'], orchid_data['between_opo']]
    
    wedges1, _ = orchid_ax.pie(
        orchid_sizes,
        colors=[COLORS['within'], COLORS['between']],
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    orchid_ax.text(
        0, 0,
        f"ORCHID\n{orchid_data['within_opo']:.1f}%\nWithin",
        ha='center', va='center',
        fontsize=11, fontweight='bold'
    )
    orchid_ax.set_title(
        f"ORCHID (2015-2021)\n{orchid_data['outcome']}\nn={orchid_data['n_hospitals']:,} hospitals",
        fontsize=10, fontweight='bold', pad=10
    )
    
    # OSR 2024 donut chart
    osr_ax = fig.add_axes([0.75, 0.30, 0.18, 0.45])
    osr_data = DATASETS['OSR 2024\n(National)']
    osr_sizes = [osr_data['within_opo'], osr_data['between_opo']]
    
    wedges2, _ = osr_ax.pie(
        osr_sizes,
        colors=[COLORS['within'], COLORS['between']],
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    osr_ax.text(
        0, 0,
        f"OSR 2024\n{osr_data['within_opo']:.1f}%\nWithin",
        ha='center', va='center',
        fontsize=11, fontweight='bold'
    )
    osr_ax.set_title(
        f"OSR 2024 (National)\n{osr_data['outcome']}\nn={osr_data['n_hospitals']:,} hospitals",
        fontsize=10, fontweight='bold', pad=10
    )
    
    # ==========================================================================
    # Main Title and Subtitle
    # ==========================================================================
    fig.suptitle(
        'The Hospital-OPO Interface: Primary Locus of Performance Variance',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    fig.text(
        0.5, 0.02,
        'ICC analysis reveals that 82-93% of performance variance occurs within OPOs '
        '(between hospitals served by the same OPO),\n'
        'not between OPOs. This finding challenges the OPO-centric regulatory framework.',
        ha='center', va='bottom',
        fontsize=10, style='italic', color=COLORS['subtitle']
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 0.52, 0.95])
    
    return fig


def save_figure(fig, output_dir: Path = OUTPUT_DIR):
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib Figure object.
        output_dir: Directory for output files.
    """
    output_dir.mkdir(exist_ok=True)
    
    # PNG (high resolution for presentations and web)
    png_path = output_dir / 'Figure1_ICC_Comparison.png'
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"Saved: {png_path}")
    
    # PDF (vector format for publication)
    pdf_path = output_dir / 'Figure1_ICC_Comparison.pdf'
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"Saved: {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate and save the variance decomposition figure."""
    print("=" * 70)
    print("GENERATING ICC VARIANCE DECOMPOSITION FIGURE")
    print("=" * 70)
    
    print("\nData Summary:")
    for name, data in DATASETS.items():
        print(f"\n{name.replace(chr(10), ' ')}:")
        print(f"  ICC: {data['icc']:.3f}")
        print(f"  Within-OPO: {data['within_opo']:.1f}%")
        print(f"  Between-OPO: {data['between_opo']:.1f}%")
        print(f"  Hospitals: {data['n_hospitals']:,}")
        print(f"  OPOs: {data['n_opos']}")
    
    print("\nGenerating figure...")
    fig = create_variance_decomposition_figure()
    
    print("\nSaving figure...")
    save_figure(fig)
    
    print("\n" + "=" * 70)
    print("Figure generation complete!")
    print("=" * 70)
    
    plt.close(fig)


if __name__ == "__main__":
    main()
