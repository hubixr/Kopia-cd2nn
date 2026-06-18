import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ============================================================
# === Configuration — set these at the top                  ===
# ============================================================

# Target frequency [GHz] — only process .npy files matching this frequency
TARGET_FREQ_GHZ = 96

# If True, generates XY propagation images for all Z positions
GENERATE_PROPAGATION_SCANS = False

# === Default Parameters (will be overridden by .meta files) ===
default_pixel_size_mm = 1.5
default_z_step_mm = 1.0
default_z_offset_mm = 0


# ============================================================
# === Helper functions (identical to main_multifreq.py)     ===
# ============================================================

def parse_meta_file(meta_path):
    """Parse .meta file to extract measurement parameters"""
    params = {
        'pixel_size_mm': default_pixel_size_mm,
        'z_step_mm': default_z_step_mm,
        'start_z_mm': 0.0,
        'z_offset_mm': default_z_offset_mm
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
                    elif 'Z Offset:' in line:
                        params['z_offset_mm'] = float(line.split(':')[1].strip().split()[0])
        except Exception as e:
            print(f"Warning: Could not parse {meta_path}: {e}")
    else:
        print(f"Warning: Meta file not found: {meta_path}")
    
    return params


def scan_and_process(data_source_path, series_folder_path):
    """Scan directory for .npy files and process all of them."""
    for file_path in glob.glob(os.path.join(data_source_path, '*.npy')):
        process_single_file(file_path, "")


def process_single_file(file_path, output_dir):
    """Load and process a single .npy file, generating cross-section (and optionally propagation scans)."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"\nProcessing file: {file_path}")
    try:
        data = np.load(file_path)  # Expected shape: (Z, Y, X)
        print(f"  Data shape: {data.shape}")
        
        # Determine output directory
        if output_dir:
            current_file_output_dir = output_dir
        else:
            # Default: <input_file_dir>/result_data/<base_name>/
            current_file_output_dir = os.path.join(
                os.path.dirname(file_path), 'result_data', base_name
            )
        os.makedirs(current_file_output_dir, exist_ok=True)
        
        # Look for corresponding .meta file
        meta_path = os.path.splitext(file_path)[0] + '.meta'
        params = parse_meta_file(meta_path)
        
        pixel_size_mm = params['pixel_size_mm']
        z_step_mm = params['z_step_mm']
        start_z_mm = params['start_z_mm']
        z_offset_mm = params['z_offset_mm']
        
        print(f"  Parameters: pixel_size={pixel_size_mm} mm, z_step={z_step_mm} mm, start_z={start_z_mm} mm, z_offset={z_offset_mm} mm")
        
        # Create cross-section (always)
        create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir=current_file_output_dir)
        
        # Create per-frequency propagation scans (optional)
        if GENERATE_PROPAGATION_SCANS:
            create_axis_scans(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir=current_file_output_dir)
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")


def create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir):
    """Create cross-sections (Y-Z at middle X ± 2 pixels) in one image.
    
    Layout: subplots in up to 3 columns, with a shared colorbar placed in the
    bottom-right corner (reusing the empty cell when plots don't fill the grid).
    """
    num_z, height, width = data.shape
    mid_x = width // 2
    
    # Define X positions: middle ± 2 pixels
    x_positions = [mid_x - 2, mid_x - 1, mid_x, mid_x + 1, mid_x + 2]
    x_positions = [x for x in x_positions if 0 <= x < width]
    
    # Create coordinate arrays with Z = 300 - z + z_offset
    z_raw = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z = 300.0 - z_raw + z_offset_mm
    y_extent = [0, height]
    
    n_plots = len(x_positions)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle(f"Przekroje boczne pola {base_name}", fontsize=24, y=0.98)
    
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.2)
    axes = []
    
    plots_in_last_row = n_plots % cols if n_plots % cols != 0 else cols
    offset = (cols - plots_in_last_row) / 2.0 if n_plots % cols != 0 else 0
    
    for idx in range(n_plots):
        row = idx // cols
        col = idx % cols
        if row == rows - 1 and n_plots % cols != 0:
            col = col + offset
        ax = fig.add_subplot(gs[row, int(col):int(col)+1])
        ax.tick_params(labelsize=18)
        axes.append(ax)
    
    # Normalize all profiles to [0, 1] for consistent colorbar
    all_profiles = np.abs(np.stack([data[:, :, x] for x in x_positions]))
    global_max = all_profiles.max()
    if global_max == 0:
        global_max = 1
    
    # Plot each cross-section (no per-subplot colorbar, no per-subplot axis labels)
    im = None
    for idx, x_pos in enumerate(x_positions):
        profile = np.abs(data[:, :, x_pos]) / global_max
        im = axes[idx].imshow(profile.T, cmap='hot',
                              extent=[z[0], z[-1], y_extent[1], y_extent[0]],
                              aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f"X = {x_pos} (środek{x_pos-mid_x:+d})", fontsize=22)
    
    # Shared axis labels for the whole figure
    fig.supxlabel("Z [mm]", fontsize=22, y=0.04)
    fig.supylabel("Y [pixels]", fontsize=22, x=0.02)
    
    # Shared colorbar — reuse the empty bottom-right cell if the grid isn't full
    if n_plots < rows * cols:
        cax = fig.add_subplot(gs[rows - 1, cols - 1])
        cax.set_frame_on(False)
        cax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # Narrow the colorbar to ~25% of cell width, centered
        bbox = cax.get_position()
        cax.set_position([bbox.x0 + bbox.width * 0.45, bbox.y0,
                          bbox.width * 0.10, bbox.height])
        cbar = fig.colorbar(im, cax=cax, label="Amplituda")
    else:
        cbar = fig.colorbar(im, ax=axes, label="Amplituda", shrink=0.7)
    cbar.set_label("Amplituda", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    
    if base_name.startswith('przekroj_boczny_'):
        base_name = base_name[len('przekroj_boczny_'):]
    out_path = os.path.join(output_dir, f"przekroj_boczny_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-sections: {out_path}")


def create_axis_scans(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir):
    """Create X-Y cross-sections at ALL Z positions showing complete beam propagation"""
    num_z, height, width = data.shape
    
    # Show ALL Z positions with real step (1mm)
    z_indices = range(0, num_z)  # Every single Z position
    
    # Calculate Z positions with Z = 300 - z + z_offset
    z_raw = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z_positions = 300.0 - z_raw + z_offset_mm
    
    # Calculate grid size for subplots - max 3 columns (matching przekroje_boczne)
    n_plots = len(z_indices)
    cols = min(3, n_plots)  # Max 3 columns
    rows = (n_plots + cols - 1) // cols  # Calculate needed rows
    
    # Create coordinate arrays - use integer pixel indices
    x_extent = [0, width]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    y_extent = [0, height]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    
    # Create figure with identical sizing as przekroje_boczne
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle(f"Propagacja w osi Z z krokiem {z_step_mm} mm - {base_name}", fontsize=24, y=0.98)
    
    # Use GridSpec with identical spacing as przekroje_boczne
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.2)
    axes = []
    
    # Calculate positions for centering last row if needed
    plots_in_last_row = n_plots % cols if n_plots % cols != 0 else cols
    offset = (cols - plots_in_last_row) / 2.0 if n_plots % cols != 0 else 0
    
    for idx in range(n_plots):
        row = idx // cols
        col = idx % cols
        
        # For the last row with fewer elements, offset to center
        if row == rows - 1:
            col = col + offset
        
        ax = fig.add_subplot(gs[row, int(col):int(col)+1])
        ax.tick_params(labelsize=18)
        axes.append(ax)
    
    # Normalize all slices to [0, 1] for consistent colorbar
    all_slices = np.abs(data[z_indices, :, :])
    global_max = all_slices.max()
    if global_max == 0:
        global_max = 1
    
    # Plot each X-Y cross-section (no per-subplot colorbar, no per-subplot axis labels)
    im = None
    for idx, z_idx in enumerate(z_indices):
        xy_slice = np.abs(data[z_idx, :, :]) / global_max
        
        im = axes[idx].imshow(xy_slice, cmap='hot',
                              extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]],
                              aspect='equal', vmin=0, vmax=1)
        z_pos = z_positions[z_idx]
        axes[idx].set_title(f"Z={z_pos:.0f}mm", fontsize=22)
    
    # Shared axis labels for the whole figure
    fig.supxlabel("X [pixels]", fontsize=22, y=0.04)
    fig.supylabel("Y [pixels]", fontsize=22, x=0.02)
    
    # Shared colorbar — reuse the empty bottom-right cell if the grid isn't full
    if n_plots < rows * cols:
        cax = fig.add_subplot(gs[rows - 1, cols - 1])
        cax.set_frame_on(False)
        cax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # Narrow the colorbar to ~10% of cell width, centered
        bbox = cax.get_position()
        cax.set_position([bbox.x0 + bbox.width * 0.45, bbox.y0,
                          bbox.width * 0.10, bbox.height])
        cbar = fig.colorbar(im, cax=cax, label="Amplituda")
    else:
        cbar = fig.colorbar(im, ax=axes, label="Amplituda", shrink=0.7)
    cbar.set_label("Amplituda", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved complete propagation (all Z positions): {out_path}")


# ============================================================
# === Main processing (same logic as main_multifreq.py)     ===
# ============================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Single-frequency data processing — target: {TARGET_FREQ_GHZ} GHz")
print(f"Working directory: {current_script_dir}")

# Find all series folders inside data_processing/ (excluding 'old' and special dirs)
series_folders = [f.path for f in os.scandir(current_script_dir)
                  if f.is_dir() and not f.name.startswith('.')
                  and f.name not in ('old', 'propagation_outputs', 'przekroje_boczne_all')]

for series_folder_path in series_folders:
    series_name = os.path.basename(series_folder_path)
    
    # Construct path to 'dane z pomiarów' subfolder
    data_source_path = os.path.join(series_folder_path, 'dane z pomiarów')

    if os.path.isdir(data_source_path):
        # Structure: SeriaX/dane z pomiarów/Seriax/*.npy
        print(f"\nProcessing series: {series_name} -> dane z pomiarów/")
        for seria_sub in os.scandir(data_source_path):
            if seria_sub.is_dir() and not seria_sub.name.startswith('.'):
                sub_name = seria_sub.name
                sub_path = seria_sub.path
                print(f"  Sub-series: {sub_name}")
                scan_and_process(sub_path, series_folder_path)
    else:
        # Structure: SeriaX/*.npy (directly in series folder)
        print(f"\nProcessing series: {series_name} -> direct")
        scan_and_process(series_folder_path, series_folder_path)

print(f"\nProcessing complete for target frequency: {TARGET_FREQ_GHZ} GHz!")
