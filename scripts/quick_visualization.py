#!/usr/bin/env python3
"""
Quick visualization script using matplotlib.

This creates several visualizations to explore the spatio-temporal data:
1. 100-year mass change time series
2. Ice core δ¹⁸O record with multi-sensory mapping
3. Scale comparison

Use for rapid iteration and pitch preparation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


# =============================================================================
# 1. 100-Year Mass Change Time Series
# =============================================================================

def plot_100yr_mass_change():
    """Plot stylized Greenland mass change over 100 years."""
    data = pd.read_csv(
        DATASETS_DIR / "spatio_temporal_100yr" / "stylized_greenland_mass_changes_1925_2025.csv",
        index_col=0, parse_dates=True
    )
    
    # Calculate total mass change
    total_mass = data.sum(axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Individual glaciers
    ax1 = axes[0]
    for col in data.columns:
        ax1.plot(data.index.year, data[col], label=col, linewidth=1, alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mass Change (1000 Gt)')
    ax1.set_title('Individual Glacier Mass Changes (1925-2025)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Total mass change
    ax2 = axes[1]
    ax2.fill_between(data.index.year, total_mass, 0, 
                     where=(total_mass < 0), color='red', alpha=0.5, label='Mass Loss')
    ax2.plot(data.index.year, total_mass, 'r-', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Mass Change (1000 Gt)')
    ax2.set_title('Total Greenland Ice Sheet Mass Change')
    ax2.text(0.95, 0.95, f'Final: {total_mass.iloc[-1]:.1f} × 10³ Gt', 
             transform=ax2.transAxes, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat'))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('datasets/spatio_temporal_100yr/mass_change_100yr.png', dpi=150)
    print("Saved: datasets/spatio_temporal_100yr/mass_change_100yr.png")
    
    return fig


# =============================================================================
# 2. Ice Core δ¹⁸O Through Time with Multi-Sensory Mapping
# =============================================================================

def plot_icecore_with_sensory():
    """Plot ice core δ¹⁸O with color representing temperature and potential audio mapping."""
    
    # Load ice core temporal driver
    driver = pd.read_csv(DATASETS_DIR / "spatio_temporal_100yr" / "icecore_temporal_driver.csv")
    
    # Also load raw data
    raw = pd.read_csv(DATASETS_DIR / "paleoclimate" / "gisp2" / "gispd18o-noaa.txt", 
                      sep='\t', comment='#')
    raw.columns = ['depth', 'd18O', 'age']
    raw = raw.replace(999999, np.nan).dropna()
    raw['age_ka'] = raw['age'] / 1000
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.5])
    
    # Plot 1: Full δ¹⁸O record (color = temperature proxy)
    ax1 = fig.add_subplot(gs[0, :])
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=driver['d18O'].min(), vmax=driver['d18O'].max())
    
    scatter = ax1.scatter(driver['age_ka'], driver['d18O'], 
                          c=driver['d18O'], cmap=cmap, s=2, alpha=0.7)
    ax1.set_xlabel('Age (thousands of years before present)')
    ax1.set_ylabel('δ¹⁸O (‰ SMOW)')
    ax1.set_title('GISP2 Ice Core δ¹⁸O Record - 110,000 Years of Climate History')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Temperature Proxy (δ¹⁸O)')
    ax1.grid(True, alpha=0.3)
    
    # Annotate key periods
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Present')
    ax1.axvline(x=-11.7, color='blue', linestyle='--', alpha=0.7, label='Last Glacial Maximum')
    ax1.annotate('HOLOCENE', xy=(5, -34), fontsize=12, fontweight='bold', color='green')
    ax1.annotate('LAST\nGLACIAL', xy=(-50, -40), fontsize=10, color='blue')
    ax1.legend(loc='upper right')
    
    # Plot 2: Color hue mapping (for visualization)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(driver['age_ka'], driver['color_hue'], 
                c=driver['color_hue'], cmap='hsv', s=3, alpha=0.5)
    ax2.set_xlabel('Age (ka)')
    ax2.set_ylabel('Color Hue (0-240)')
    ax2.set_title('Color Mapping for Visualization')
    ax2.set_ylim(0, 240)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Audio pitch mapping (for sonification)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(driver['age_ka'], driver['audio_pitch'], 
                c=driver['audio_pitch'], cmap='plasma', s=3, alpha=0.5)
    ax3.set_xlabel('Age (ka)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Audio Pitch Mapping for Sonification')
    ax3.set_ylim(200, 1200)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Vibration intensity mapping
    ax4 = fig.add_subplot(gs[2, :])
    ax4.fill_between(driver['age_ka'], driver['vibration_intensity'], 0, alpha=0.5, color='purple')
    ax4.set_xlabel('Age (thousands of years before present)')
    ax4.set_ylabel('Vibration Intensity (normalized)')
    ax4.set_title('Vibration Intensity Mapping (0-1 scale)')
    ax4.set_xlim(driver['age_ka'].min(), driver['age_ka'].max())
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('datasets/spatio_temporal_100yr/icecore_sensory_mapping.png', dpi=150)
    print("Saved: datasets/spatio_temporal_100yr/icecore_sensory_mapping.png")
    
    return fig


# =============================================================================
# 3. Scale Comparison Plot
# =============================================================================

def plot_scale_comparison():
    """Plot the 4 temporal scales side by side for pitch."""
    
    scales = pd.read_csv(DATASETS_DIR / "spatio_temporal_100yr" / "temporal_scale_comparison.csv")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    colors = {'modern_100yr': 'red', 'holocene_11ka': 'orange', 
              'full_glacial_110ka': 'blue', 'million_yr_potential': 'purple'}
    
    for i, (_, row) in enumerate(scales.iterrows()):
        ax = axes[i]
        
        # Create a simple bar or indicator showing scale
        time_range = row['time_range_years']
        
        # Create a log-scale bar to show relative time scales
        bar_heights = [1]  # single bar
        x_pos = [0]
        
        # Draw a horizontal bar whose length represents time range (log scale)
        log_range = np.log10(time_range)
        
        ax.barh(0, log_range, color=colors[row['scale_name']], alpha=0.7, height=0.5)
        ax.set_xlim(0, 7)  # 10^7 = 10 million years
        ax.set_yticks([])
        ax.set_xlabel('log₁₀(Years)')
        
        # Add time range annotation
        ax.set_title(row['display_name'], fontsize=12, fontweight='bold')
        
        # Add time range text
        if time_range < 1000:
            label = f'{time_range:,} years'
        elif time_range < 1e6:
            label = f'{time_range/1000:.0f} ka'
        else:
            label = f'{time_range/1e6:.1f} Ma'
        
        ax.text(0.05, 0.5, label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='center')
        
        # Data source
        ax.text(0.05, 0.25, row['data_source'], transform=ax.transAxes,
                fontsize=8, va='center', style='italic', wrap=True)
    
    plt.suptitle('Temporal Scales for Greenland Ice Sheet Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('datasets/spatio_temporal_100yr/scale_comparison.png', dpi=150)
    print("Saved: datasets/spatio_temporal_100yr/scale_comparison.png")
    
    return fig


# =============================================================================
# 4. Multi-panel Summary for Pitch
# =============================================================================

def plot_pitch_summary():
    """Create a summary figure for the scientific pitch."""
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig)
    
    # Load data
    mass_data = pd.read_csv(
        DATASETS_DIR / "spatio_temporal_100yr" / "stylized_greenland_mass_changes_1925_2025.csv",
        index_col=0, parse_dates=True
    )
    total_mass = mass_data.sum(axis=1)
    driver = pd.read_csv(DATASETS_DIR / "spatio_temporal_100yr" / "icecore_temporal_driver.csv")
    
    # Panel 1: Modern mass loss (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(mass_data.index.year, total_mass, 0, 
                     where=(total_mass < 0), color='crimson', alpha=0.6)
    ax1.plot(mass_data.index.year, total_mass, 'crimson', linewidth=2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mass Change (1000 Gt)')
    ax1.set_title('100 Years: Modern Observations\n(Synthetic GRACE-style)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: 11ka Holocene (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    holo_mask = (driver['age_ka'] >= 0) & (driver['age_ka'] <= 12)
    holo_driver = driver[holo_mask]
    ax2.plot(holo_driver['age_ka'], holo_driver['d18O'], 'orange', linewidth=0.5)
    ax2.scatter(holo_driver['age_ka'], holo_driver['d18O'], 
                c=holo_driver['d18O'], cmap='coolwarm', s=10, alpha=0.7)
    ax2.set_xlabel('Age (ka)')
    ax2.set_ylabel('δ¹⁸O (‰)')
    ax2.set_title('11,700 Years: Holocene Climate\n(Little Ice Age, Roman Warm Period)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: 110ka Full glacial cycle (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(driver['age_ka'], driver['d18O'], 
                         c=driver['d18O'], cmap='coolwarm', s=2, alpha=0.5)
    ax3.set_xlabel('Age (thousands of years before present)')
    ax3.set_ylabel('δ¹⁸O (‰)')
    ax3.set_title('110,000 Years: Full Glacial Cycle\n(Dansgaard-Oeschger Events)', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Present')
    # Mark DO events region
    ax3.axvspan(-60, -20, alpha=0.1, color='blue', label='Frequent DO Events')
    ax3.legend(loc='lower right')
    plt.colorbar(scatter, ax=ax3, label='δ¹⁸O')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Million-year potential (middle right) - the pitch slide
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(0.5, 0.7, '2.5 Million Years', fontsize=20, fontweight='bold',
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.5, 'ICE SHEET MODEL\nSIMULATION NEEDED', fontsize=16,
             ha='center', transform=ax4.transAxes, color='purple',
             fontweight='bold')
    ax4.text(0.5, 0.25, '• Plio-Pleistocene transitions\n• Greenland ice sheet formation\n• 41-kyr and 100-kyr cycles\n• Climate-biology-geology coupling', 
             fontsize=11, ha='center', va='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Million-Year Potential\n(Why Run the Simulation?)', fontsize=12, fontweight='bold', color='purple')
    
    # Panel 5: The pitch message (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    message = """
    THE PITCH: Even with limited data, we can demonstrate compelling multi-scale visualizations.
    
    Each time scale reveals different dynamics: modern acceleration → Holocene variability → glacial abruptness → (missing) deep time structure.
    
    A million-year simulation would allow exhibition visitors to FEEL the full climate story, from Greenland's formation to potential future collapse.
    """
    ax5.text(0.5, 0.5, message, fontsize=11, ha='center', va='center',
             transform=ax5.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('datasets/spatio_temporal_100yr/pitch_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: datasets/spatio_temporal_100yr/pitch_summary.png")
    
    return fig


# =============================================================================
# 5. Animated GIF - Ice Core Through Time
# =============================================================================

def create_animation():
    """Create an animation showing ice core data from past to present.
    
    The animation starts at the oldest data (~110ka) and sweeps forward
    through time to the most recent (~present), as if traveling forward
    through time from the deep past to now.
    """
    
    driver = pd.read_csv(DATASETS_DIR / "spatio_temporal_100yr" / "icecore_temporal_driver.csv")
    
    # Sort by age_ka DESCENDING: oldest (110ka) first, most recent (near 0) last
    # age_ka = thousands of years before present
    # Higher age_ka = older time, so descending gives us past → present
    sorted_driver = driver.sort_values('age_ka', ascending=False).reset_index(drop=True)
    
    # Subsample for animation - take every 10th point
    subsample = sorted_driver.iloc[::10].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def update(frame):
        ax.clear()
        
        # Plot all points up to current frame
        # This shows time building up from oldest to newest
        x = subsample['age_ka'].iloc[:frame]
        y = subsample['d18O'].iloc[:frame]
        colors = subsample['color_hue'].iloc[:frame]
        
        ax.scatter(x, y, c=colors, cmap='hsv', s=5, alpha=0.6)
        ax.set_xlim(-5, 115)  # 0 to 110 ka (note: now showing ka BP, so smaller values = more recent)
        ax.set_ylim(-45, -32)
        ax.set_xlabel('Age (thousands of years before present)')
        ax.set_ylabel('δ¹⁸O (‰ SMOW)')
        ax.set_title('GISP2 Ice Core Through Time: Past → Present')
        
        # Add current point marker (the "frontier" of time)
        if frame > 0:
            current_x = subsample['age_ka'].iloc[frame - 1]  # -1 because we show points UP TO frame
            current_y = subsample['d18O'].iloc[frame - 1]
            ax.scatter([current_x], [current_y], c='black', s=100, zorder=5, marker='v')
            ax.axvline(x=current_x, color='gray', linestyle='--', alpha=0.5)
            # Add annotation showing how far back we're going
            time_diff = subsample['age_ka'].iloc[frame - 1]
            ax.text(0.02, 0.98, f'Exploring: {time_diff:.1f} ka ago', 
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        
        return []
    
    ani = FuncAnimation(fig, update, frames=len(subsample) + 1, interval=50, blit=False)
    
    print("Saving animation (this may take a moment)...")
    ani.save('datasets/spatio_temporal_100yr/icecore_animation.gif', 
             writer=PillowWriter(fps=20), dpi=100)
    print("Saved: datasets/spatio_temporal_100yr/icecore_animation.gif")
    
    plt.close()
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("MATPLOTLIB VISUALIZATION")
    print("Quick preview for pitch")
    print("="*60)
    
    print("\n[1/5] Creating 100-year mass change plot...")
    plot_100yr_mass_change()
    
    print("\n[2/5] Creating ice core with sensory mapping...")
    plot_icecore_with_sensory()
    
    print("\n[3/5] Creating scale comparison...")
    plot_scale_comparison()
    
    print("\n[4/5] Creating pitch summary...")
    plot_pitch_summary()
    
    print("\n[5/5] Creating animation (GIF)...")
    create_animation()
    
    print("\n" + "="*60)
    print("DONE! Generated files in datasets/spatio_temporal_100yr/")
    print("="*60)
    
    # Show the pitch summary
    img = plt.imread('datasets/spatio_temporal_100yr/pitch_summary.png')
    plt.figure(figsize=(16, 14))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()