from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import rfft2, irfft2, fft2, ifft2


#Config
PIXEL_SIZE = 9e-4  # [m]
FREQUENCY = 96 * 1e9  # [GHz]
C = 299792458  # [m/s]
WAVELENGTH = C / (FREQUENCY)  # [m]
DISTANCE_BETWEEN_DOE = 0.0201 #[m]
DISTANCE_TO_TARGET = 0.201 #[m]

# Distance range configuration
DISTANCE_MIN = 0.195  # [m] - minimum distance to target
DISTANCE_MAX = 0.205  # [m] - maximum distance to target
DISTANCE_STEP = 0.0001  # [m] - step size for distance variation
PADDING_MULTIPLIER = 4  # Padding multiplier for the propagation layer
H_BEFORE_PADDING, W_BEFORE_PADDING = 128, 128  # Size of the image in pixels before padding
H = 128 * (PADDING_MULTIPLIER + 1)
W = 128 * (PADDING_MULTIPLIER + 1)  # Rozmiar obrazu w pikselach

# Wczytaj wszystkie pliki BMP z folderu 'data/phase_masks'
phase_masks_dir = Path('data/phase_masks')
phase_mask_files = sorted(phase_masks_dir.glob('*.bmp'))
if not phase_mask_files:
    raise FileNotFoundError(f'No BMP files found in {phase_masks_dir}!')

phase_masks = []
for bmp_file in phase_mask_files:
    img = Image.open(bmp_file).convert('L')  # grayscale
    img_array = np.array(img, dtype=np.float32)
    img_norm = img_array / 255.0 * 2 * np.pi  # Normalize to [0, 2π]
    phase_masks.append(img_norm)
    print(f'Loaded {bmp_file}')
    print('Shape:', img_norm.shape)
    print('Min:', img_norm.min(), 'Max:', img_norm.max())

# Wczytaj pojedynczy plik BMP z folderu 'data/input_field'
input_field_dir = Path('data/input_field')
input_field_files = sorted(input_field_dir.glob('*.bmp'))
if not input_field_files:
    raise FileNotFoundError(f'No BMP files found in {input_field_dir}!')
input_img = Image.open(input_field_files[0]).convert('L')
input_array = np.array(input_img, dtype=np.float32)
input_norm = input_array / 255.0
print(f'Loaded input field: {input_field_files[0]}')
print('Shape:', input_norm.shape)
print('Min:', input_norm.min(), 'Max:', input_norm.max())

# Wizualizacja input field + wszystkich masek fazowych w jednej figurze
num_masks = len(phase_masks)
fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
# Input field na pozycji 0
axes[0].imshow(input_norm, cmap='viridis')
axes[0].set_title('Input field')
axes[0].axis('off')
plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
# Phase masks na kolejnych pozycjach
for i, mask in enumerate(phase_masks):
    axes[i + 1].imshow(mask, cmap='viridis')
    axes[i + 1].set_title(f'Phase mask {i+1}')
    axes[i + 1].axis('off')
    plt.colorbar(axes[i + 1].images[0], ax=axes[i + 1], fraction=0.046, pad=0.04)
plt.tight_layout()
# plt.show()  # Disabled - don't show individual plots

#input field as U
U = np.stack([input_norm, np.zeros_like(input_norm)], axis=-1)  # Two channels: input_norm and zeros

# Dodaj zero padding do U
pad_size = int(H / (PADDING_MULTIPLIER + 1) * PADDING_MULTIPLIER / 2)
U_padded = np.pad(U, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
print(f'U_padded shape: {U_padded.shape}')

# Generate array of distance values to test
distance_values = np.arange(DISTANCE_MIN, DISTANCE_MAX + DISTANCE_STEP, DISTANCE_STEP)
print(f"Testing distances: {distance_values}")

# Create results directory with subfolders
results_dir = Path('results_between')
results_dir.mkdir(exist_ok=True)

# Create subfolders for organization
final_outputs_dir = results_dir / 'final_outputs'
cross_sections_dir = results_dir / 'cross_sections'
all_masks_dir = results_dir / 'all_masks_overview'
comparisons_dir = results_dir / 'comparisons'
side_views_dir = results_dir / 'side_views'

final_outputs_dir.mkdir(exist_ok=True)
cross_sections_dir.mkdir(exist_ok=True)
all_masks_dir.mkdir(exist_ok=True)
comparisons_dir.mkdir(exist_ok=True)
side_views_dir.mkdir(exist_ok=True)

# Store all results for comparison
all_results = {}

# Store side view data for beam propagation visualization
side_view_data = {}

# Store 3D field data for cross-section analysis (like cross_view.py)
field_3d_data = {}

# Loop through each distance value
for dist_idx, target_distance in enumerate(distance_values):
    print(f"\n{'='*50}")
    print(f"Processing distance {dist_idx+1}/{len(distance_values)}: {target_distance:.3f} m")
    print(f"{'='*50}")
    
    # Reset U_padded for each distance iteration
    U_current = U_padded.copy()
    output_intensities = []
    
    # Store side view data for this distance
    side_view_profile = []
    propagation_distances = []
    cumulative_distance = 0.0
    
    # Store 3D field data for this distance
    field_3d_for_distance = []
    
    for i in range(num_masks):
        if i == num_masks-1:
            distance = target_distance  # Use the current target distance being tested
            print(f"Using target distance {target_distance:.3f} m for the last mask")
        else:
            distance = DISTANCE_BETWEEN_DOE  # Other masks use the initial distance
            print(f"Using distance between DOE {DISTANCE_BETWEEN_DOE:.3f} m for mask {i + 1}")
        
        print(f"Processing mask {i + 1}/{num_masks} with distance {distance:.3f} m")
        re_u = U_current[..., 0]
        im_u = U_current[..., 1]
        phase = np.pad(phase_masks[i], ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

        re_u = re_u * np.cos(phase) - im_u * np.sin(phase)
        im_u = re_u * np.sin(phase) + im_u * np.cos(phase)

        dx = PIXEL_SIZE
        W_pad = re_u.shape[1]
        H_pad = re_u.shape[0]
        fx = np.fft.fftfreq(W_pad, d=dx)  # oś pozioma
        fy = np.fft.fftfreq(H_pad, d=dx)  # oś pionowa
        FX, FY = np.meshgrid(fx, fy)
        r2 = FX**2 + FY**2
        k = 2 * np.pi / WAVELENGTH
        arg = -1j*np.pi*distance * WAVELENGTH * r2
        h = np.exp(1j*k*distance)*np.exp(arg)

        h_real = np.real(h).astype(np.complex64)
        h_imag = np.imag(h).astype(np.complex64)

        N = re_u.shape[0]  # Assuming square input for simplicity
        print(f"Convolution with phase mask {i + 1}")
        # Use complex FFT for correct shape and propagation
        re_re = np.real(ifft2(fft2(re_u) * h_real)) / np.sqrt(N**3)
        im_im = np.real(ifft2(fft2(im_u) * h_imag)) / np.sqrt(N**3)
        re_im = np.real(ifft2(fft2(re_u) * h_real)) / np.sqrt(N**3)
        im_re = np.real(ifft2(fft2(im_u) * h_imag)) / np.sqrt(N**3)

        out_real = re_re - im_im
        out_imag = re_im + im_re

        # Crop output to original (unpadded) size
        crop_h_start = pad_size
        crop_h_end = pad_size + H_BEFORE_PADDING
        crop_w_start = pad_size
        crop_w_end = pad_size + W_BEFORE_PADDING
        out_real_cropped = out_real[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        out_imag_cropped = out_imag[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        # Calculate and print power before and after cropping
        power_before = np.sum(out_real**2 + out_imag**2)
        power_after = np.sum(out_real_cropped**2 + out_imag_cropped**2)
        percent_lost = 100 * (power_before - power_after) / power_before if power_before > 0 else 0
        print(f"Power before cropping: {power_before:.6f}")
        print(f"Power after cropping: {power_after:.6f}")
        print(f"Percentage of power lost: {percent_lost:.2f}%")

        # For next propagation step, pad back to padded size (except after last mask)
        if i < num_masks - 1:
            out_real = np.pad(out_real_cropped, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
            out_imag = np.pad(out_imag_cropped, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
            U_current = np.stack([out_real, out_imag], axis=-1)
        else:
            U_current = np.stack([out_real_cropped, out_imag_cropped], axis=-1)

        print(f"Output field after mask {i + 1} shape: {U_current.shape}")
        
        # Calculate output intensity
        output_intensity = out_real_cropped**2 + out_imag_cropped**2
        output_intensities.append(output_intensity)
        
        # Store side view profile data (middle column cross-section)
        center_x = output_intensity.shape[1] // 2
        center_profile = output_intensity[:, center_x]  # Middle column (y-axis profile)
        side_view_profile.append(center_profile)
        cumulative_distance += distance
        propagation_distances.append(cumulative_distance)
        
        # Store 3D field data for cross-section analysis
        # Create complex field for storage
        complex_field = out_real_cropped + 1j * out_imag_cropped
        field_3d_for_distance.append(complex_field)
    
    # Store final result for this distance
    final_intensity = output_intensities[-1]
    all_results[target_distance] = {
        'final_intensity': final_intensity,
        'all_intensities': output_intensities.copy(),
        'max_intensity': np.max(final_intensity),
        'total_power': np.sum(final_intensity)
    }
    
    # Store side view data
    side_view_data[target_distance] = {
        'profiles': side_view_profile,
        'distances': propagation_distances.copy()
    }
    
    # Store 3D field data
    field_3d_data[target_distance] = field_3d_for_distance.copy()
    
    # Save results for this specific distance (final outputs only)
    distance_str = f"{target_distance:.3f}m".replace('.', 'p')
    
    # Save the final output field (intensity) as a colormap image
    output_colormap = plt.get_cmap('hot')(final_intensity / final_intensity.max() if final_intensity.max() > 0 else final_intensity)
    output_colormap_img = (output_colormap[..., :3] * 255).astype(np.uint8)  # Drop alpha, scale to [0,255]
    output_img = Image.fromarray(output_colormap_img)
    output_img.save(final_outputs_dir / f'final_output_dist_{distance_str}.png')
    
    # Plot final intensity cross-section through the center
    center_y = final_intensity.shape[0] // 2
    plt.figure(figsize=(8, 4))
    plt.plot(final_intensity[center_y, :], label=f'Distance {target_distance:.3f}m (y={center_y})', linewidth=2)
    plt.title(f'Final intensity cross-section (y={center_y}) - Distance {target_distance:.3f}m')
    plt.xlabel('x position')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cross_sections_dir / f'final_cross_section_dist_{distance_str}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot all mask outputs for this distance (optional - remove if not needed)
    fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))
    if num_masks == 1:
        axes = [axes]
    for j, inten in enumerate(output_intensities):
        axes[j].imshow(inten, cmap='hot')
        axes[j].set_title(f'Distance {target_distance:.3f}m\nMask {j+1} ({inten.shape[1]}x{inten.shape[0]} px)')
        axes[j].axis('off')
        plt.colorbar(axes[j].images[0], ax=axes[j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(all_masks_dir / f'all_masks_dist_{distance_str}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate side view beam propagation plot
    if len(side_view_profile) > 0:
        # Create 2D array for side view visualization (stack middle columns)
        side_view_2d = np.column_stack(side_view_profile)  # Each column is a propagation step
        
        # Check if we have enough data for contour plot (need at least 2x2)
        if side_view_2d.shape[1] >= 2:
            # Create coordinates for plotting
            y_positions = np.arange(side_view_2d.shape[0]) * PIXEL_SIZE * 1000  # Transverse position in mm
            x_positions = np.array(propagation_distances) * 1000  # Propagation distance in mm
            X, Y = np.meshgrid(x_positions, y_positions)
            
            # Plot side view
            plt.figure(figsize=(12, 6))
            plt.contourf(X, Y, side_view_2d, levels=50, cmap='hot')
            plt.colorbar(label='Intensity')
            plt.xlabel('Propagation distance (mm)')
            plt.ylabel('Transverse position (mm)')
            plt.title(f'Beam propagation side view - Distance {target_distance:.3f}m\n(Middle column cross-section)')
            
            # Add vertical lines to show mask positions
            for i, x_pos in enumerate(propagation_distances):
                plt.axvline(x=x_pos*1000, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
                plt.text(x_pos*1000, y_positions[-1]*0.95, f'M{i+1}', 
                        verticalalignment='top', horizontalalignment='center', 
                        color='white', fontsize=8, alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(side_views_dir / f'side_view_dist_{distance_str}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print(f"Skipping contour plot for distance {target_distance:.3f}m - insufficient data points (need at least 2 masks)")
        
        # Alternative visualization: line plots at different propagation distances
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(side_view_profile)))
        y_positions = np.arange(side_view_2d.shape[0]) * PIXEL_SIZE * 1000  # Transverse position in mm
        for i, (profile, dist) in enumerate(zip(side_view_profile, propagation_distances)):
            plt.plot(y_positions, profile, color=colors[i], 
                    label=f'z = {dist*1000:.1f}mm (Mask {i+1})', linewidth=2)
        
        plt.xlabel('Transverse position (mm)')
        plt.ylabel('Intensity')
        plt.title(f'Beam profiles at different propagation distances - Distance {target_distance:.3f}m')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(side_views_dir / f'beam_profiles_dist_{distance_str}.png', dpi=150, bbox_inches='tight')
        plt.close()

# Generate comparison plots for all distances
print(f"\n{'='*50}")
print("Generating comparison plots...")
print(f"{'='*50}")

# Plot final intensities for all distances
fig, axes = plt.subplots(2, int(np.ceil(len(distance_values)/2)), figsize=(6 * int(np.ceil(len(distance_values)/2)), 12))
axes = axes.flatten() if len(distance_values) > 1 else [axes]

for i, (dist, result) in enumerate(all_results.items()):
    if i < len(axes):
        axes[i].imshow(result['final_intensity'], cmap='hot')
        axes[i].set_title(f'Distance: {dist:.3f}m\nMax: {result["max_intensity"]:.2e}\nPower: {result["total_power"]:.2e}')
        axes[i].axis('off')
        plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046, pad=0.04)

# Hide unused subplots
for i in range(len(all_results), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(comparisons_dir / 'comparison_all_distances.png', dpi=150, bbox_inches='tight')
# plt.show()  # Disabled - don't show individual plots

# Plot cross-sections comparison
center_y = H_BEFORE_PADDING // 2
plt.figure(figsize=(12, 6))
for dist, result in all_results.items():
    intensity = result['final_intensity']
    plt.plot(intensity[center_y, :], label=f'{dist:.3f}m (max: {np.max(intensity):.2e})', linewidth=2)

plt.title(f'Intensity cross-sections comparison (y={center_y})')
plt.xlabel('x position')
plt.ylabel('Intensity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(comparisons_dir / 'cross_sections_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()  # Disabled - don't show individual plots

# Generate comprehensive side view comparison
print("Generating comprehensive side view for full range...")

# Create one comprehensive side view showing all distances
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Create a combined dataset showing all distances and their beam evolution
all_profiles = []
all_distance_labels = []
step_positions = []
current_step = 0

for target_dist in sorted(side_view_data.keys()):
    side_data = side_view_data[target_dist]
    profiles = side_data['profiles']
    
    # Add each mask step for this target distance
    for i, profile in enumerate(profiles):
        all_profiles.append(profile)
        all_distance_labels.append(f"{target_dist:.3f}m_M{i+1}")
        step_positions.append(current_step)
        current_step += 1
    
    # Add a small gap between different target distances
    current_step += 0.5

# Create the combined side view array
if all_profiles and len(all_profiles) >= 2:
    combined_side_view = np.column_stack(all_profiles)
    
    # Create coordinates
    y_positions = np.arange(combined_side_view.shape[0]) * PIXEL_SIZE * 1000  # Transverse position in mm
    x_positions = np.array(step_positions)
    X, Y = np.meshgrid(x_positions, y_positions)
    
    # Plot the comprehensive side view
    im = ax.contourf(X, Y, combined_side_view, levels=100, cmap='hot')
    ax.set_xlabel('Propagation Steps (Target Distance + Mask)')
    ax.set_ylabel('Transverse Position (mm)')
    ax.set_title('Comprehensive Beam Propagation Side View - Full Distance Range')
    
    # Add vertical lines and labels for each step
    step_idx = 0
    colors_dist = plt.cm.tab10(np.linspace(0, 1, len(side_view_data)))
    
    for dist_idx, target_dist in enumerate(sorted(side_view_data.keys())):
        side_data = side_view_data[target_dist]
        num_masks = len(side_data['profiles'])
        
        # Add vertical lines for each mask in this distance
        for mask_idx in range(num_masks):
            x_pos = step_positions[step_idx]
            ax.axvline(x=x_pos, color='white', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add labels at the top
            label = f"{target_dist:.3f}m\nM{mask_idx+1}"
            ax.text(x_pos, y_positions[-1]*0.98, label, 
                   verticalalignment='top', horizontalalignment='center', 
                   color='white', fontsize=8, alpha=0.9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_dist[dist_idx], alpha=0.3))
            
            step_idx += 1
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Customize x-axis
    ax.set_xticks(step_positions[::2])  # Show every other tick to avoid crowding
    ax.set_xticklabels([all_distance_labels[i] for i in range(0, len(all_distance_labels), 2)], 
                       rotation=45, ha='right', fontsize=8)
else:
    # If insufficient data for contour plot, create a simple message
    ax.text(0.5, 0.5, 'Insufficient data for comprehensive side view\n(Need at least 2 data points)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Comprehensive Beam Propagation Side View - Insufficient Data')

plt.tight_layout()
plt.savefig(side_views_dir / 'comprehensive_side_view_full_range.png', dpi=150, bbox_inches='tight')
# plt.show()  # Disabled - don't show individual plots

# Also create a simpler version showing only final results for each distance
plt.figure(figsize=(12, 8))
final_profiles_only = []
final_distances = []

for target_dist in sorted(side_view_data.keys()):
    final_profile = side_view_data[target_dist]['profiles'][-1]  # Only final profile
    final_profiles_only.append(final_profile)
    final_distances.append(target_dist)

if final_profiles_only:
    final_side_view = np.column_stack(final_profiles_only)
    y_positions = np.arange(final_side_view.shape[0]) * PIXEL_SIZE * 1000
    x_positions = np.array(final_distances) * 1000  # Convert to mm
    X, Y = np.meshgrid(x_positions, y_positions)
    
    plt.contourf(X, Y, final_side_view, levels=50, cmap='hot')
    plt.colorbar(label='Final Intensity')
    plt.xlabel('Target Distance (mm)')
    plt.ylabel('Transverse Position (mm)')
    plt.title('Final Beam Profiles vs Target Distance')
    
    # Add vertical lines for each distance
    for dist in final_distances:
        plt.axvline(x=dist*1000, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
        plt.text(dist*1000, y_positions[-1]*0.95, f'{dist:.3f}m', 
                verticalalignment='top', horizontalalignment='center', 
                color='white', fontsize=8, alpha=0.8)

plt.tight_layout()
plt.savefig(side_views_dir / 'final_profiles_vs_distance.png', dpi=150, bbox_inches='tight')
# plt.show()  # Disabled - don't show individual plots

# Generate beam evolution comparison (final profiles only)
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(distance_values)))

for i, (target_dist, side_data) in enumerate(side_view_data.items()):
    final_profile = side_data['profiles'][-1]  # Get final profile (middle column)
    y_positions = np.arange(len(final_profile)) * PIXEL_SIZE * 1000
    plt.plot(y_positions, final_profile, color=colors[i], 
             label=f'Target: {target_dist:.3f}m (max: {np.max(final_profile):.2e})', 
             linewidth=2)

plt.xlabel('Transverse position (mm)')
plt.ylabel('Final intensity')
plt.title('Final beam profiles comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(side_views_dir / 'final_beam_profiles_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()

# Generate cross-section analysis similar to cross_view.py
print("Generating cross-section analysis (similar to cross_view.py)...")

# Create 3D data arrays and save as .npy files, then generate side views
for target_dist, field_data in field_3d_data.items():
    if len(field_data) > 0:
        # Stack the 2D fields into a 3D array (Z, Y, X)
        field_3d_array = np.stack(field_data, axis=0)  # Shape: (num_masks, height, width)
        
        # Save as .npy file
        distance_str = f"{target_dist:.4f}m".replace('.', 'p')
        npy_filename = side_views_dir / f'field_3d_dist_{distance_str}.npy'
        np.save(npy_filename, field_3d_array)
        print(f"Saved 3D field data: {npy_filename}")
        
        # Generate side view like cross_view.py
        num_z, height, width = field_3d_array.shape
        mid_x = width // 2
        
        # Get amplitude profile from middle X column
        profile = np.abs(field_3d_array[:, :, mid_x])  # Shape: (Z, Y)
        
        # Create coordinate arrays
        pixel_size_mm = PIXEL_SIZE * 1000  # Convert to mm
        z_step_mm = 1.0  # Step between masks (arbitrary units for visualization)
        
        z = np.arange(0, num_z * z_step_mm, z_step_mm)
        y = np.arange(0, height * pixel_size_mm, pixel_size_mm)
        
        # Create the side view plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(profile.T, cmap='hot', 
                       extent=[z[0], z[-1], y[-1], y[0]], 
                       aspect='auto')
        plt.title(f"Side View - Target Distance {target_dist:.4f}m\n(Center X = {mid_x})")
        plt.xlabel("Propagation Step")
        plt.ylabel("Transverse Position (mm)")
        plt.colorbar(im, label="Field Amplitude")
        
        # Add labels for mask positions
        for i, z_pos in enumerate(z):
            plt.axvline(x=z_pos, color='white', linestyle='--', alpha=0.7, linewidth=1)
            plt.text(z_pos, y[-1]*0.95, f'M{i+1}', 
                    verticalalignment='top', horizontalalignment='center', 
                    color='white', fontsize=8, alpha=0.9)
        
        plt.tight_layout()
        
        # Save the plot
        side_view_filename = side_views_dir / f'side_view_cross_{distance_str}.png'
        plt.savefig(side_view_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved side view: {side_view_filename}")

# Create comprehensive side view combining all distances
print("Creating comprehensive side view for all distances...")

all_profiles_combined = []
all_z_labels = []
current_z = 0

for target_dist in sorted(field_3d_data.keys()):
    field_data = field_3d_data[target_dist]
    if len(field_data) > 0:
        field_3d_array = np.stack(field_data, axis=0)
        num_z, height, width = field_3d_array.shape
        mid_x = width // 2
        profile = np.abs(field_3d_array[:, :, mid_x])  # Shape: (Z, Y)
        
        # Add each mask step
        for i in range(num_z):
            all_profiles_combined.append(profile[i, :])
            all_z_labels.append(f"{target_dist:.4f}m_M{i+1}")
            current_z += 1
        
        # Add gap between distances
        current_z += 0.5

if all_profiles_combined:
    # Create the comprehensive side view
    combined_profile = np.column_stack(all_profiles_combined)  # Shape: (Y, total_steps)
    
    pixel_size_mm = PIXEL_SIZE * 1000
    y = np.arange(0, combined_profile.shape[0] * pixel_size_mm, pixel_size_mm)
    z_steps = np.arange(len(all_profiles_combined))
    
    plt.figure(figsize=(16, 8))
    im = plt.imshow(combined_profile, cmap='hot', 
                   extent=[z_steps[0], z_steps[-1], y[-1], y[0]], 
                   aspect='auto')
    plt.title("Comprehensive Side View - All Target Distances")
    plt.xlabel("Propagation Steps (Distance + Mask)")
    plt.ylabel("Transverse Position (mm)")
    plt.colorbar(im, label="Field Amplitude")
    
    # Add vertical lines and labels
    step_idx = 0
    colors_dist = plt.cm.tab10(np.linspace(0, 1, len(field_3d_data)))
    
    for dist_idx, target_dist in enumerate(sorted(field_3d_data.keys())):
        field_data = field_3d_data[target_dist]
        num_masks = len(field_data)
        
        for mask_idx in range(num_masks):
            plt.axvline(x=step_idx, color='white', linestyle='--', alpha=0.7, linewidth=1)
            
            label = f"{target_dist:.4f}m\nM{mask_idx+1}"
            plt.text(step_idx, y[-1]*0.98, label, 
                   verticalalignment='top', horizontalalignment='center', 
                   color='white', fontsize=7, alpha=0.9,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors_dist[dist_idx], alpha=0.3))
            
            step_idx += 1
        
        step_idx += 0.5  # Gap between distances
    
    plt.tight_layout()
    plt.savefig(side_views_dir / 'comprehensive_side_view_cross_all.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Disabled - don't show individual plots
    plt.close()
    print("Saved comprehensive side view: comprehensive_side_view_cross_all.png")

# Generate summary statistics
print("\nSummary Statistics:")
print("-" * 50)
summary_text = "Summary Statistics:\n" + "-" * 50 + "\n"
for dist, result in all_results.items():
    line = f"Distance {dist:.3f}m: Max intensity = {result['max_intensity']:.2e}, Total power = {result['total_power']:.2e}"
    print(line)
    summary_text += line + "\n"

# Find optimal distance (highest max intensity)
optimal_dist = max(all_results.keys(), key=lambda d: all_results[d]['max_intensity'])
optimal_line = f"\nOptimal distance (highest max intensity): {optimal_dist:.3f}m"
optimal_max = f"Max intensity: {all_results[optimal_dist]['max_intensity']:.2e}"
optimal_power = f"Total power: {all_results[optimal_dist]['total_power']:.2e}"

print(optimal_line)
print(optimal_max)
print(optimal_power)

summary_text += optimal_line + "\n" + optimal_max + "\n" + optimal_power + "\n"

# Save summary to file
with open(comparisons_dir / 'summary_statistics.txt', 'w') as f:
    f.write(summary_text)

print(f"\nAll results saved to: {results_dir.absolute()}")
print(f"├── final_outputs/        - Final intensity images")
print(f"├── cross_sections/       - Cross-section plots")  
print(f"├── all_masks_overview/   - All masks overview")
print(f"├── side_views/           - Beam propagation side views")
print(f"└── comparisons/          - Comparison plots and statistics")


