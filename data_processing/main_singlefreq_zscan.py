import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ============================================================
# === Configuration — set these at the top                  ===
# ============================================================

# --- Z-scan settings ---
# Target final Z positions [mm] (after Z = 300 - z_raw + z_offset transformation).
Z_TARGETS_MM = [150, 152, 154, 156, 158]  # list of Z positions to plot

# Alternatively, use a range with step (overrides Z_TARGETS_MM if not empty):
Z_RANGE_START_MM = 180   # inclusive
Z_RANGE_STOP_MM  = 220   # inclusive
Z_RANGE_STEP_MM  = 2     # step in mm
# Set USE_RANGE = True to use Z_RANGE_*, False to use Z_TARGETS_MM
USE_RANGE = True

# --- Cross-section settings ---
# If True, also generates the standard Y-Z cross-section plot
GENERATE_CROSS_SECTION = True

# === Default Parameters (will be overridden by .meta files) ===
default_pixel_size_mm = 1.5
default_z_step_mm = 1.0
default_z_offset_mm = 0


# ============================================================
# === Helper functions                                      ===
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


def z_raw_to_final(z_raw, z_offset_mm):
    """Convert raw Z (from scan) to final display Z"""
    return 300.0 - z_raw + z_offset_mm


def z_final_to_raw(z_final, z_offset_mm):
    """Convert final display Z back to raw Z"""
    return 300.0 + z_offset_mm - z_final


def create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir):
    """Create cross-sections (Y-Z at middle X ± 2 pixels) — same as main_singlefreq.py"""
    num_z, height, width = data.shape
    mid_x = width // 2
    
    x_positions = [mid_x - 2, mid_x - 1, mid_x, mid_x + 1, mid_x + 2]
    x_positions = [x for x in x_positions if 0 <= x < width]
    
    z_raw = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z = z_raw_to_final(z_raw, z_offset_mm)
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
    
    all_profiles = np.abs(np.stack([data[:, :, x] for x in x_positions]))
    global_max = all_profiles.max()
    if global_max == 0:
        global_max = 1
    
    im = None
    for idx, x_pos in enumerate(x_positions):
        profile = np.abs(data[:, :, x_pos]) / global_max
        im = axes[idx].imshow(profile.T, cmap='hot',
                              extent=[z[0], z[-1], y_extent[1], y_extent[0]],
                              aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f"X = {x_pos} (środek{x_pos-mid_x:+d})", fontsize=22)
    
    fig.supxlabel("Z [mm]", fontsize=22, y=0.04)
    fig.supylabel("Y [pixels]", fontsize=22, x=0.02)
    
    if n_plots < rows * cols:
        cax = fig.add_subplot(gs[rows - 1, cols - 1])
        cax.set_frame_on(False)
        cax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
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
    print(f"  Saved cross-sections: {out_path}")


def create_zscan_plots(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm,
                       output_dir, z_targets_mm):
    """Generate X-Y cross-section plots only at specified final Z positions."""
    num_z, height, width = data.shape
    
    # Build array of raw Z coordinates
    z_raw_arr = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z_final_arr = z_raw_to_final(z_raw_arr, z_offset_mm)
    
    # Find the closest raw index for each target Z
    selected_indices = []
    selected_z_final = []
    for z_target in z_targets_mm:
        idx = np.argmin(np.abs(z_final_arr - z_target))
        selected_indices.append(idx)
        selected_z_final.append(z_final_arr[idx])
    
    n_plots = len(selected_indices)
    if n_plots == 0:
        print(f"  No Z targets to plot!")
        return
    
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    x_extent = [0, width]
    y_extent = [0, height]
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle(f"Propagacja Z - {base_name}", fontsize=24, y=0.98)
    
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
    
    # Global normalization across all selected slices
    selected_data = np.abs(data[selected_indices, :, :])
    global_max = selected_data.max()
    if global_max == 0:
        global_max = 1
    
    im = None
    for idx, z_idx in enumerate(selected_indices):
        xy_slice = np.abs(data[z_idx, :, :]) / global_max
        im = axes[idx].imshow(xy_slice, cmap='hot',
                              extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]],
                              aspect='equal', vmin=0, vmax=1)
        axes[idx].set_title(f"Z={selected_z_final[idx]:.1f}mm", fontsize=22)
    
    fig.supxlabel("X [pixels]", fontsize=22, y=0.04)
    fig.supylabel("Y [pixels]", fontsize=22, x=0.02)
    
    if n_plots < rows * cols:
        cax = fig.add_subplot(gs[rows - 1, cols - 1])
        cax.set_frame_on(False)
        cax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        bbox = cax.get_position()
        cax.set_position([bbox.x0 + bbox.width * 0.45, bbox.y0,
                          bbox.width * 0.10, bbox.height])
        cbar = fig.colorbar(im, cax=cax, label="Amplituda")
    else:
        cbar = fig.colorbar(im, ax=axes, label="Amplituda", shrink=0.7)
    cbar.set_label("Amplituda", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"zscan_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Z-scan: {out_path}")
    for i, z_idx in enumerate(selected_indices):
        print(f"    Z={selected_z_final[i]:.1f}mm -> raw index {z_idx}")


def process_single_file(file_path):
    """Load a single .npy file and generate Z-scan (and optionally cross-section) plots."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"\nProcessing file: {file_path}")
    try:
        data = np.load(file_path)
        print(f"  Data shape: {data.shape}")
        
        # Determine output directory
        output_dir = os.path.join(os.path.dirname(file_path), 'result_data', base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load parameters from .meta
        meta_path = os.path.splitext(file_path)[0] + '.meta'
        params = parse_meta_file(meta_path)
        
        pixel_size_mm = params['pixel_size_mm']
        z_step_mm = params['z_step_mm']
        start_z_mm = params['start_z_mm']
        z_offset_mm = params['z_offset_mm']
        
        print(f"  Parameters: pixel_size={pixel_size_mm} mm, z_step={z_step_mm} mm, start_z={start_z_mm} mm, z_offset={z_offset_mm} mm")
        
        # Build Z targets list
        if USE_RANGE:
            z_targets = np.arange(Z_RANGE_START_MM, Z_RANGE_STOP_MM + Z_RANGE_STEP_MM, Z_RANGE_STEP_MM).tolist()
        else:
            z_targets = Z_TARGETS_MM
        
        print(f"  Target Z positions: {z_targets} mm")
        
        # Generate cross-section (optional)
        if GENERATE_CROSS_SECTION:
            print(f"  Generating cross-section...")
            create_cross_section(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm, output_dir)
        
        # Generate Z-scan plots
        print(f"  Generating Z-scan plots...")
        create_zscan_plots(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm,
                           output_dir, z_targets)
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")


def scan_and_process(data_source_path):
    """Scan directory for .npy files and process all of them."""
    for file_path in glob.glob(os.path.join(data_source_path, '*.npy')):
        process_single_file(file_path)


# ============================================================
# === Main processing (same logic as main_singlefreq.py)    ===
# ============================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Z-scan data processing")
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
                scan_and_process(sub_path)
    else:
        # Structure: SeriaX/*.npy (directly in series folder)
        print(f"\nProcessing series: {series_name} -> direct")
        scan_and_process(series_folder_path)

print(f"\nProcessing complete!")
