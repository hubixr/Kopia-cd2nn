import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Paths are relative to the script, independent of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'inputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'multifreq')

# === Default Parameters (will be overridden by .meta files) ===
default_pixel_size_mm = 1.5
default_z_step_mm = 1.0
default_z_offset_mm = 148

# === Controls ===
GENERATE_PROPAGATION_SCANS = False  # If True, generates per-frequency XY propagation images

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

def process_npy_files(data_source_path, series_folder_path, label):
    """Find .npy files in data_source_path and process them, saving to outputs/multifreq/<series>/<base_name>/"""
    for file_path in sorted(glob.glob(os.path.join(data_source_path, '*.npy'))):
        print(f"\nProcessing file: {file_path}")
        try:
            data = np.load(file_path)  # Expected shape: (Z, Y, X)
            print(f"  Data shape: {data.shape}")
            
            # Get base name (filename without extension) - this already includes the range/GHz value
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Output mirrors the source directory under outputs/multifreq/.
            result_data_base_dir = os.path.join(OUTPUT_DIR, os.path.relpath(data_source_path, INPUT_DIR))
            current_file_output_dir = os.path.join(result_data_base_dir, base_name)

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
    """Create cross-sections (Y-Z at middle X ± 2 pixels) in one image"""
    num_z, height, width = data.shape
    mid_x = width // 2
    
    # Define X positions: middle ± 2 pixels
    x_positions = [mid_x - 2, mid_x - 1, mid_x, mid_x + 1, mid_x + 2]
    # Filter out positions that are out of bounds
    x_positions = [x for x in x_positions if 0 <= x < width]
    
    # Create coordinate arrays with Z = 300 - z + z_offset
    z_raw = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
    z = 300.0 - z_raw + z_offset_mm
    y_extent = [0, height]  # Pixel indices: 0 to 32 (covers pixels 0-31)
    
    # Create figure with subplots for each X position - max 3 columns
    n_plots = len(x_positions)
    cols = min(3, n_plots)  # Max 3 columns
    rows = (n_plots + cols - 1) // cols  # Calculate needed rows
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    fig.suptitle(f"Przekroje boczne pola {base_name}", fontsize=24, y=0.98)
    
    # Use GridSpec for better control of subplot positioning
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.7)
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
        axes.append(ax)
    
    # Normalize all profiles to [0, 1] for consistent colorbar
    all_profiles = np.abs(np.stack([data[:, :, x] for x in x_positions]))
    global_max = all_profiles.max()
    if global_max == 0:
        global_max = 1
    
    # Plot each cross-section (no per-subplot colorbar)
    im = None
    for idx, x_pos in enumerate(x_positions):
        profile = np.abs(data[:, :, x_pos]) / global_max
        
        im = axes[idx].imshow(profile.T, cmap='hot',
                              extent=[z[0], z[-1], y_extent[1], y_extent[0]],
                              aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f"X = {x_pos} (środek{x_pos-mid_x:+d})", fontsize=20)
        axes[idx].set_xlabel("Z [mm]", fontsize=20)
        axes[idx].set_ylabel("Y [pixels]", fontsize=20)
    
    # Shared colorbar — reuse the empty bottom-right cell if the grid isn't full
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
    
    # Remove 'przekroj_boczny_' prefix if already present
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
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.7)
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
        axes.append(ax)
    
    # Normalize all slices to [0, 1] for consistent colorbar
    all_slices = np.abs(data[z_indices, :, :])
    global_max = all_slices.max()
    if global_max == 0:
        global_max = 1
    
    # Plot each X-Y cross-section (no per-subplot colorbar)
    im = None
    for idx, z_idx in enumerate(z_indices):
        xy_slice = np.abs(data[z_idx, :, :]) / global_max
        
        im = axes[idx].imshow(xy_slice, cmap='hot',
                              extent=[x_extent[0], x_extent[1], y_extent[1], y_extent[0]],
                              aspect='equal', vmin=0, vmax=1)
        z_pos = z_positions[z_idx]
        axes[idx].set_title(f"Z={z_pos:.0f}mm", fontsize=20)
        axes[idx].set_xlabel("X [pixels]", fontsize=20)
        axes[idx].set_ylabel("Y [pixels]", fontsize=20)
    
    # Shared colorbar — reuse the empty bottom-right cell if the grid isn't full
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
    out_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved complete propagation (all Z positions): {out_path}")


def extract_prefix(base_name):
    """Extract structure prefix from filename (e.g., 'wzorcowa180_200mm_140ghz' -> 'wzorcowa180_200mm')"""
    # Remove the last _XXXghz or _XX-XX part
    parts = base_name.split('_')
    if len(parts) >= 2:
        last = parts[-1]
        # Check if last part is a frequency (e.g., '140ghz', '95-105')
        if last.endswith('ghz') or ('-' in last and last.replace('-', '').isdigit()):
            return '_'.join(parts[:-1])
    return base_name


def extract_freq_value(base_name):
    """Extract numeric frequency value for sorting (e.g., '140ghz' -> 140, '95-105' -> 95)"""
    parts = base_name.split('_')
    last = parts[-1]
    if last.endswith('ghz'):
        return int(last.replace('ghz', ''))
    elif '-' in last:
        return int(last.split('-')[0])
    return 0


def create_combined_cross_sections(data_source_path, series_folder_path):
    """Create a combined figure with cross-sections for all frequencies of the same structure.
    
    Layout: rows = 5 X positions (środek-2..środek+2), columns = frequencies sorted ascending.
    """
    npy_files = sorted(glob.glob(os.path.join(data_source_path, '*.npy')))
    if not npy_files:
        return
    
    # Group files by prefix
    groups = {}
    for fp in npy_files:
        base_name = os.path.splitext(os.path.basename(fp))[0]
        prefix = extract_prefix(base_name)
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(fp)
    
    for prefix, file_list in groups.items():
        if len(file_list) < 2:
            continue  # Need at least 2 frequencies for a combined plot
        
        print(f"\nCreating combined cross-sections for: {prefix}")
        
        # Sort files by frequency
        file_list.sort(key=lambda fp: extract_freq_value(os.path.splitext(os.path.basename(fp))[0]))
        
        # Load all data
        all_data = []
        freq_labels = []
        meta_params = None
        for fp in file_list:
            base_name = os.path.splitext(os.path.basename(fp))[0]
            data = np.load(fp)
            all_data.append(data)
            
            # Get frequency label
            parts = base_name.split('_')
            freq_label = parts[-1]
            freq_labels.append(freq_label)
            
            # Parse meta from first file for parameters
            if meta_params is None:
                meta_path = os.path.splitext(fp)[0] + '.meta'
                meta_params = parse_meta_file(meta_path)
        
        num_z, height, width = all_data[0].shape
        mid_x = width // 2
        x_positions = [mid_x - 2, mid_x - 1, mid_x, mid_x + 1, mid_x + 2]
        x_positions = [x for x in x_positions if 0 <= x < width]
        n_x = len(x_positions)
        n_freq = len(all_data)
        
        # Calculate Z coordinates (use first file's params)
        z_step_mm = meta_params['z_step_mm']
        start_z_mm = meta_params['start_z_mm']
        z_offset_mm = meta_params['z_offset_mm']
        z_raw = np.arange(start_z_mm, start_z_mm + num_z * z_step_mm, z_step_mm)
        z = 300.0 - z_raw + z_offset_mm
        y_extent = [0, height]
        
        # Create figure: rows = X positions, columns = frequencies
        fig_width = 4 * n_freq
        fig_height = 4 * n_x
        fig, axes = plt.subplots(n_x, n_freq, figsize=(fig_width, fig_height))
        fig.suptitle(f"Przekroje boczne - {prefix}", fontsize=28, y=0.98)
        
        # Handle single-row or single-column edge case
        if n_x == 1:
            axes = [axes]
        if n_freq == 1:
            axes = [[a] for a in axes]
        
        for xi in range(n_x):
            x_pos = x_positions[xi]
            for fi in range(n_freq):
                profile = np.abs(all_data[fi][:, :, x_pos])
                ax = axes[xi][fi]
                im = ax.imshow(profile.T, cmap='hot', extent=[z[0], z[-1], y_extent[1], y_extent[0]], aspect='auto')
                
                if xi == 0:
                    ax.set_title(freq_labels[fi], fontsize=18)
                if fi == 0:
                    ax.set_ylabel(f"X={x_pos}\n(środek{x_pos-mid_x:+d})", fontsize=16)
                ax.set_xlabel("Z [mm]", fontsize=14)
                ax.tick_params(labelsize=12)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Amplituda", fontsize=12)
                cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        out_dir = os.path.join(OUTPUT_DIR, os.path.relpath(data_source_path, INPUT_DIR))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"przekroje_boczne_zbiorcze_{prefix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved combined cross-sections: {out_path}")


# === Main processing - run after all functions are defined ===
current_script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Find measurement series inside inputs/.
series_folders = [f.path for f in sorted(os.scandir(INPUT_DIR), key=lambda entry: entry.name) if f.is_dir() and not f.name.startswith('.') and f.name not in ('propagation_outputs', 'przekroje_boczne_all')]

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
                process_npy_files(sub_path, series_folder_path, sub_name)
                create_combined_cross_sections(sub_path, series_folder_path)
    elif series_name == 'old':
        # Structure: old/ref/*.npy, old/d2nn_krok1/*.npy, etc.
        print(f"\nProcessing old series folders...")
        for old_sub in os.scandir(series_folder_path):
            if old_sub.is_dir() and not old_sub.name.startswith('.'):
                sub_name = old_sub.name
                sub_path = old_sub.path
                print(f"  Old sub-series: {sub_name}")
                process_npy_files(sub_path, old_sub.path, sub_name)
                create_combined_cross_sections(sub_path, old_sub.path)
    else:
        # Structure: SeriaX/*.npy (directly in series folder)
        print(f"\nProcessing series: {series_name} -> direct")
        process_npy_files(series_folder_path, series_folder_path, series_name)
        create_combined_cross_sections(series_folder_path, series_folder_path)

print(f"\nProcessing complete for all series!")