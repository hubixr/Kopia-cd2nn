"""
Script to combine and visualize FWHM and Max Intensity data from multiple propagation_srcipt/main.py runs.

This script reads CSV files from the frequency_data/ folder and plots them together on the same graph,
allowing you to compare results from different configurations.

Usage:
    python combine_frequency_data.py
    
The script will:
1. Read all fwhm_data_*.csv files and plot them together
2. Read all max_intensity_data_*.csv files and plot them together
3. Create combined comparison plots in the combined_frequency_plots/ folder
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_frequency_data(data_type='fwhm'):
    """
    Load all CSV files of specified type from frequency_data/ folder.
    
    Args:
        data_type: 'fwhm' or 'max_intensity'
    
    Returns:
        Dictionary with timestamp as key and DataFrame as value
    """
    data_dir = Path('frequency_data')
    
    if not data_dir.exists():
        print(f"Error: {data_dir} folder not found!")
        print("Make sure you have run propagation_srcipt/main.py at least once.")
        return {}
    
    file_pattern = f'{data_type}_data_*.csv'
    csv_files = sorted(data_dir.glob(file_pattern))
    
    if not csv_files:
        print(f"No {file_pattern} files found in {data_dir}/")
        return {}
    
    data_dict = {}
    for csv_file in csv_files:
        timestamp = csv_file.stem.replace(f'{data_type}_data_', '')
        try:
            df = pd.read_csv(csv_file)
            data_dict[timestamp] = df
            print(f"Loaded: {csv_file.name}")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    return data_dict


def plot_combined_fwhm():
    """Plot FWHM data from all runs on the same graph."""
    fwhm_data = load_frequency_data('fwhm')
    
    if not fwhm_data:
        print("No FWHM data found. Skipping FWHM plot.")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fwhm_data)))
    
    for (timestamp, df), color in zip(fwhm_data.items(), colors):
        plt.plot(df['Frequency_GHz'], df['FWHM_mm'], 
                marker='o', linewidth=2, markersize=6,
                label=f'Run: {timestamp}', color=color)
    
    plt.title('FWHM vs Frequency - Combined Multiple Runs', fontsize=16)
    plt.xlabel('Frequency [GHz]', fontsize=12)
    plt.ylabel('FWHM [mm]', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path('combined_frequency_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'combined_fwhm_vs_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_max_intensity():
    """Plot Max Intensity data from all runs on the same graph."""
    max_int_data = load_frequency_data('max_intensity')
    
    if not max_int_data:
        print("No Max Intensity data found. Skipping Max Intensity plot.")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(max_int_data)))
    
    for (timestamp, df), color in zip(max_int_data.items(), colors):
        plt.plot(df['Frequency_GHz'], df['Max_Intensity'], 
                marker='s', linewidth=2, markersize=6,
                label=f'Run: {timestamp}', color=color)
    
    plt.title('Maximum Intensity vs Frequency - Combined Multiple Runs', fontsize=16)
    plt.xlabel('Frequency [GHz]', fontsize=12)
    plt.ylabel('Maximum Intensity', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path('combined_frequency_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'combined_max_intensity_vs_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_data_summary():
    """Print summary statistics for all loaded data."""
    fwhm_data = load_frequency_data('fwhm')
    max_int_data = load_frequency_data('max_intensity')
    
    print("\n" + "="*70)
    print("COMBINED FREQUENCY DATA SUMMARY")
    print("="*70)
    
    if fwhm_data:
        print(f"\nFWHM Data ({len(fwhm_data)} runs):")
        for timestamp, df in sorted(fwhm_data.items()):
            min_fwhm = df['FWHM_mm'].min()
            max_fwhm = df['FWHM_mm'].max()
            avg_fwhm = df['FWHM_mm'].mean()
            freq_min_fwhm = df.loc[df['FWHM_mm'].idxmin(), 'Frequency_GHz']
            print(f"  {timestamp}:")
            print(f"    Min FWHM: {min_fwhm:.6f} mm at {freq_min_fwhm:.1f} GHz")
            print(f"    Max FWHM: {max_fwhm:.6f} mm")
            print(f"    Avg FWHM: {avg_fwhm:.6f} mm")
    
    if max_int_data:
        print(f"\nMax Intensity Data ({len(max_int_data)} runs):")
        for timestamp, df in sorted(max_int_data.items()):
            min_intensity = df['Max_Intensity'].min()
            max_intensity = df['Max_Intensity'].max()
            avg_intensity = df['Max_Intensity'].mean()
            freq_max_intensity = df.loc[df['Max_Intensity'].idxmax(), 'Frequency_GHz']
            print(f"  {timestamp}:")
            print(f"    Min Intensity: {min_intensity:.6f}")
            print(f"    Max Intensity: {max_intensity:.6f} at {freq_max_intensity:.1f} GHz")
            print(f"    Avg Intensity: {avg_intensity:.6f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("FREQUENCY DATA COMBINATION SCRIPT")
    print("="*70)
    print("\nLoading FWHM data...")
    plot_combined_fwhm()
    
    print("\nLoading Max Intensity data...")
    plot_combined_max_intensity()
    
    print("\nGenerating summary...")
    print_data_summary()
    
    print("\nCombined plots saved in: combined_frequency_plots/")
    print("="*70)
