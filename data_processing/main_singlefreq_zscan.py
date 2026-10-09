import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Paths are relative to the script, independent of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'inputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'zscan')

# ============================================================
# === Configuration — set these at the top                  ===
# ============================================================

# --- Z-scan settings ---
# Option A: exact list of final Z positions [mm] to plot.
Z_TARGETS_MM = []  # e.g. [190, 192, 194, 196]

# Option B: range — auto-selects ALL unique layers whose Z_final falls within [start, stop].
# Only used when Z_TARGETS_MM is empty.
Z_RANGE_MM = [195, 205]  # [start, stop] inclusive

# === Default Parameters (will be overridden by .meta files) ===
default_pixel_size_mm = 1.5
default_z_step_mm = 1.0
default_z_offset_mm = 0


# ============================================================
# === Helper functions                                      ===
# ============================================================

def find_meta_file(file_path):
    """Find .meta file for given .npy path. Falls back to any .meta in same dir."""
    exact = os.path.splitext(file_path)[0] + '.meta'
    if os.path.exists(exact):
        return exact
    # Fallback: look for any .meta file in the same directory
    meta_files = glob.glob(os.path.join(os.path.dirname(file_path), '*.meta'))
    if meta_files:
        print(f"  Warning: {os.path.basename(exact)} not found, using {os.path.basename(meta_files[0])}")
        return meta_files[0]
    return exact  # will trigger "not found" message


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


def create_propagation_plots(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm,
                             output_dir, z_targets_mm):
    """Generate X-Y cross-section plots at specified final Z positions (propagation scan)."""
    num_z, height, width = data.shape
    
    # Build array of raw Z coordinates
    z_raw_arr = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z_final_arr = z_raw_to_final(z_raw_arr, z_offset_mm)
    
    # Find the closest raw index for each target Z (no duplicates)
    seen_indices = set()
    selected_indices = []
    selected_z_final = []
    for z_target in z_targets_mm:
        idx = np.argmin(np.abs(z_final_arr - z_target))
        if idx not in seen_indices:
            seen_indices.add(idx)
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
    fig.suptitle(f"Propagacja - {base_name}", fontsize=24, y=0.98)
    
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
    out_path = os.path.join(output_dir, f"propagacja_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved propagation: {out_path}")
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
        output_dir = os.path.join(OUTPUT_DIR, os.path.relpath(os.path.dirname(file_path), INPUT_DIR), base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load parameters from .meta
        meta_path = find_meta_file(file_path)
        params = parse_meta_file(meta_path)
        
        pixel_size_mm = params['pixel_size_mm']
        z_step_mm = params['z_step_mm']
        start_z_mm = params['start_z_mm']
        z_offset_mm = params['z_offset_mm']
        
        print(f"  Parameters: pixel_size={pixel_size_mm} mm, z_step={z_step_mm} mm, start_z={start_z_mm} mm, z_offset={z_offset_mm} mm")
        
        # Build Z targets list
        if Z_TARGETS_MM:
            z_targets = Z_TARGETS_MM
        else:
            # Auto-select all unique layers whose Z_final falls within Z_RANGE_MM
            num_z = data.shape[0]
            z_raw_arr = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
            z_final_arr = z_raw_to_final(z_raw_arr, z_offset_mm)
            mask = (z_final_arr >= Z_RANGE_MM[0]) & (z_final_arr <= Z_RANGE_MM[1])
            z_targets = sorted(z_final_arr[mask].tolist())
        
        print(f"  Target Z positions ({len(z_targets)} unique): {z_targets} mm")
        
        # Generate propagation plots (X-Y at each target Z)
        print(f"  Generating propagation plots...")
        create_propagation_plots(data, file_path, base_name, pixel_size_mm, z_step_mm, start_z_mm, z_offset_mm,
                                 output_dir, z_targets)
        
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")


def scan_and_process(data_source_path):
    """Scan directory for .npy files and process all of them."""
    for file_path in sorted(glob.glob(os.path.join(data_source_path, '*.npy'))):
        process_single_file(file_path)


# ============================================================
# === Main processing (same logic as main_singlefreq.py)    ===
# ============================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Z-scan data processing")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Find measurement series inside inputs/.
series_folders = [f.path for f in sorted(os.scandir(INPUT_DIR), key=lambda entry: entry.name)
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
