import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Default Parameters (will be overridden by .meta files) ===
default_pixel_size_mm = 1.5
default_z_step_mm = 1.0

def parse_meta_file(meta_path):
    """Parse .meta file to extract measurement parameters"""
    params = {
        'pixel_size_mm': default_pixel_size_mm,
        'z_step_mm': default_z_step_mm,
        'start_z_mm': 0.0
    }
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                for line in f:
                    if 'Pixel size:' in line:
                        params['pixel_size_mm'] = float(line.split(':')[1].strip().split()[0])
                    elif 'Step Z:' in line:
                        params['z_step_mm'] = float(line.split(':')[1].strip().split()[0])
                    elif 'Start Z:' in line:
                        params['start_z_mm'] = float(line.split(':')[1].strip().split()[0])
        except Exception as e:
            print(f"Warning: Could not parse {meta_path}: {e}")
    else:
        print(f"Warning: Meta file not found: {meta_path}")
    
    return params

# === Output folders ===
cross_section_dir = 'przekroje_boczne_all'
propagation_dir = 'propagation_outputs'
os.makedirs(cross_section_dir, exist_ok=True)
os.makedirs(propagation_dir, exist_ok=True)

def create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm):
    """Create cross-sections (Y-Z at middle X ± 2 pixels) in one image"""
    num_z, height, width = data.shape
    mid_x = width // 2
    
    # Define X positions: middle ± 2 pixels
    x_positions = [mid_x - 2, mid_x - 1, mid_x, mid_x + 1, mid_x + 2]
    # Filter out positions that are out of bounds
    x_positions = [x for x in x_positions if 0 <= x < width]
    
    # Create coordinate arrays
    z = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    y_extent = [0, height]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    
    # Create figure with subplots for each X position
    n_plots = len(x_positions)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
    fig.suptitle(f"Przekroje boczne pola (X = środek ± 2 piksele)\n{base_name}", fontsize=12, y=0.98)
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Plot each cross-section
    for idx, x_pos in enumerate(x_positions):
        profile = np.abs(data[:, :, x_pos])  # shape: (Z, Y)
        
        im = axes[idx].imshow(profile.T, cmap='hot', extent=[z[0], z[-1], y_extent[1], y_extent[0]], aspect='auto')
        axes[idx].set_title(f"X = {x_pos} (środek{x_pos-mid_x:+d})")
        axes[idx].set_xlabel("Z [mm]")
        axes[idx].set_ylabel("Y [pixels]")
        plt.colorbar(im, ax=axes[idx], label="Amplituda")
    
    plt.tight_layout()
    
    # Remove 'przekroj_boczny_' prefix if already present
    if base_name.startswith('przekroj_boczny_'):
        base_name = base_name[len('przekroj_boczny_'):]
    out_path = os.path.join(cross_section_dir, f"przekroj_boczny_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-sections: {out_path}")

def create_axis_scans(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm):
    """Create X-Y cross-sections at ALL Z positions showing complete beam propagation"""
    num_z, height, width = data.shape
    
    # Show ALL Z positions with real step (1mm)
    z_indices = range(0, num_z)  # Every single Z position
    
    # Calculate grid size for subplots - use more columns for many plots
    n_plots = len(z_indices)
    cols = min(10, n_plots)  # Max 10 columns for better layout
    rows = (n_plots + cols - 1) // cols  # Calculate needed rows
    
    # Create coordinate arrays - use integer pixel indices
    x_extent = [0, width]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    y_extent = [0, height]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    
    # Create figure - reasonable size for each subplot
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    fig.suptitle(f"Complete X-Y Propagation (every 1mm step) - {base_name}", fontsize=16)
    
    # Flatten axes for easier indexing
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each X-Y cross-section
    for idx, z_idx in enumerate(z_indices):
        if idx < len(axes):
            # Get X-Y cross-section at this Z position
            xy_slice = np.abs(data[z_idx, :, :])
            
            im = axes[idx].imshow(xy_slice, cmap='hot', extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]], aspect='equal')
            z_pos = start_z_mm + z_idx * z_step_mm
            axes[idx].set_title(f"Z={z_pos:.0f}mm", fontsize=8)
            axes[idx].set_xlabel("X [pixels]", fontsize=6)
            axes[idx].set_ylabel("Y [pixels]", fontsize=6)
            axes[idx].tick_params(labelsize=6)
            # Add smaller colorbar
            cbar = plt.colorbar(im, ax=axes[idx], shrink=0.8)
            cbar.ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(propagation_dir, f"complete_propagation_1mm_steps_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved complete propagation (all Z positions): {out_path}")


# === Process all .npy files in current directory subfolders ===
current_dir = os.getcwd()
print(f"Working in directory: {current_dir}")

for file_path in glob.glob('**/*.npy', recursive=True):
    # Skip files in output directories
    if cross_section_dir in file_path or propagation_dir in file_path:
        continue
    
    print(f"\nProcessing file: {file_path}")
    try:
        data = np.load(file_path)  # Expected shape: (Z, Y, X)
        print(f"Data shape: {data.shape}")
        
        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Look for corresponding .meta file
        meta_path = os.path.splitext(file_path)[0] + '.meta'
        params = parse_meta_file(meta_path)
        
        pixel_size_mm = params['pixel_size_mm']
        z_step_mm = params['z_step_mm']
        start_z_mm = params['start_z_mm']
        
        print(f"Using parameters: pixel_size={pixel_size_mm} mm, z_step={z_step_mm} mm, start_z={start_z_mm} mm")
        
        # Create both cross-section and axis scans
        create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm)
        create_axis_scans(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"\nProcessing complete!")
print(f"Cross-sections saved to: {cross_section_dir}/")
print(f"Axis scans saved to: {propagation_dir}/")
