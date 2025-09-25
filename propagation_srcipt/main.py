from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import rfft2, irfft2, fft2, ifft2


#Config
PIXEL_SIZE = 9e-4  # [m]
C = 299792458  # [m/s]

# Frequency range configuration
FREQUENCY_MIN = 160 * 1e9  # [Hz] - minimum frequency
FREQUENCY_MAX = 200 * 1e9  # [Hz] - maximum frequency
FREQUENCY_STEP = 0.5 * 1e9   # [Hz] - step size for frequency variation

DISTANCE_BETWEEN_DOE = 0.05 #[m]
DISTANCE_TO_TARGET = 0.201 #[m]
PADDING_MULTIPLIER = 10  # Padding multiplier for the propagation layer
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
plt.show()

# Generate array of frequency values to test
frequency_values = np.arange(FREQUENCY_MIN, FREQUENCY_MAX + FREQUENCY_STEP, FREQUENCY_STEP)
print(f"Testing frequencies: {frequency_values / 1e9} GHz")

# Create results directory with subfolders
results_dir = Path('results_frequency_range')
results_dir.mkdir(exist_ok=True)

# Create subfolders for organization
final_outputs_dir = results_dir / 'final_outputs'
cross_sections_dir = results_dir / 'cross_sections'
all_masks_dir = results_dir / 'all_masks_overview'
comparisons_dir = results_dir / 'comparisons'

final_outputs_dir.mkdir(exist_ok=True)
cross_sections_dir.mkdir(exist_ok=True)
all_masks_dir.mkdir(exist_ok=True)
comparisons_dir.mkdir(exist_ok=True)

# Store all results for comparison
all_results = {}

#input field as U
U = np.stack([input_norm, np.zeros_like(input_norm)], axis=-1)  # Two channels: input_norm and zeros

# Dodaj zero padding do U
pad_size = int(H / (PADDING_MULTIPLIER + 1) * PADDING_MULTIPLIER / 2)
U_padded = np.pad(U, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
print(f'U_padded shape: {U_padded.shape}')


# Loop through each frequency value
for freq_idx, frequency in enumerate(frequency_values):
    wavelength = C / frequency  # Calculate wavelength for current frequency
    print(f"\n{'='*60}")
    print(f"Processing frequency {freq_idx+1}/{len(frequency_values)}: {frequency/1e9:.1f} GHz")
    print(f"Wavelength: {wavelength*1e3:.3f} mm")
    print(f"{'='*60}")
    
    # Reset U_padded for each frequency iteration
    U_current = U_padded.copy()
    output_intensities = []
    
    # Calculate initial power for power loss calculation
    initial_power = np.sum(U_current[..., 0]**2 + U_current[..., 1]**2)
    
    for i in range(num_masks):
        if i == num_masks-1 :
            distance = DISTANCE_TO_TARGET  # Last mask uses distance to target
            print("Using distance to target for the last mask")
        else:
            distance = DISTANCE_BETWEEN_DOE  # Other masks use the initial distance
            print(f"Using distance between DOE for mask {i + 1}")
        print(f"Processing mask {i + 1}/{num_masks} with distance {distance:.3f} m")
        re_u = U_current[..., 0]
        im_u = U_current[..., 1]
        phase = np.pad(phase_masks[i], ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')

        # Apply phase mask correctly (save original values)
        re_u_new = re_u * np.cos(phase) - im_u * np.sin(phase)
        im_u_new = re_u * np.sin(phase) + im_u * np.cos(phase)
        re_u = re_u_new
        im_u = im_u_new

        dx = PIXEL_SIZE
        W_pad = re_u.shape[1]
        H_pad = re_u.shape[0]
        fx = np.fft.fftfreq(W_pad, d=dx)  # oś pozioma
        fy = np.fft.fftfreq(H_pad, d=dx)  # oś pionowa
        FX, FY = np.meshgrid(fx, fy)
        r2 = FX**2 + FY**2
        k = 2 * np.pi / wavelength
        arg = -1j*np.pi*distance * wavelength * r2
        h = np.exp(1j*k*distance)*np.exp(arg)

        h_real = np.real(h).astype(np.complex64)
        h_imag = np.imag(h).astype(np.complex64)

        print("re_u shape:", re_u.shape)
        print("im_u shape:", im_u.shape)

        print("h_real shape:", h_real.shape)
        print("h_imag shape:", h_imag.shape)
        N = re_u.shape[0]  # Assuming square input for simplicity
        print("Convolution with phase mask", i + 1)
        # Use complex FFT for correct shape and propagation
        re_re = np.real(ifft2(fft2(re_u) * h_real))
        im_im = np.real(ifft2(fft2(im_u) * h_imag))
        re_im = np.real(ifft2(fft2(re_u) * h_imag))
        im_re = np.real(ifft2(fft2(im_u) * h_real))

        out_real = re_re - im_im
        out_imag = re_im + im_re

        # Crop output to original (unpadded) size
        crop_h_start = pad_size
        crop_h_end = pad_size + H_BEFORE_PADDING
        crop_w_start = pad_size
        crop_w_end = pad_size + W_BEFORE_PADDING
        out_real_cropped = out_real[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        out_imag_cropped = out_imag[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        # Calculate intensity and add to list
        output_intensity = out_real_cropped**2 + out_imag_cropped**2
        output_intensities.append(output_intensity)

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
    
    # Store final result for this frequency
    final_intensity = output_intensities[-1]
    all_results[frequency] = {
        'final_intensity': final_intensity,
        'all_intensities': output_intensities.copy(),
        'max_intensity': np.max(final_intensity),
        'total_power': np.sum(final_intensity),
        'initial_power': initial_power,
        'power_loss_percent': 100 * (initial_power - np.sum(final_intensity)) / initial_power,
        'wavelength': wavelength
    }
    
    # Save results for this specific frequency
    frequency_str = f"{frequency/1e9:.1f}GHz".replace('.', 'p')
    
    # Create frequency-specific results directory
    freq_results_dir = Path(f'results_between/frequency_{frequency_str}')
    freq_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final output field (intensity) as a colormap image
    output_colormap = plt.get_cmap('hot')(final_intensity / final_intensity.max() if final_intensity.max() > 0 else final_intensity)
    output_colormap_img = (output_colormap[..., :3] * 255).astype(np.uint8)
    output_img = Image.fromarray(output_colormap_img)
    output_img.save(freq_results_dir / f'cmap_final_output_freq_{frequency_str}.png')
    
    # Plot and save intensity cross-sections
    center_y = final_intensity.shape[0] // 2
    plt.figure(figsize=(8, 4))
    plt.plot(final_intensity[center_y, :], label=f'Final Output at {frequency/1e9:.1f} GHz (y={center_y})')
    plt.title(f'Intensity cross-section (y={center_y}) at frequency {frequency/1e9:.1f} GHz')
    plt.xlabel('x [pixels]')
    plt.ylabel('Normalized intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(freq_results_dir / f'cross_section_y{center_y}_freq_{frequency_str}.png')
    plt.close()
    
    # Save individual mask outputs for this frequency
    for mask_idx, intensity in enumerate(output_intensities):
        mask_colormap = plt.get_cmap('hot')(intensity / intensity.max() if intensity.max() > 0 else intensity)
        mask_colormap_img = (mask_colormap[..., :3] * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_colormap_img)
        mask_img.save(freq_results_dir / f'cmap_after_mask_{mask_idx+1}_freq_{frequency_str}.png')
    
    print(f"Completed processing for frequency {frequency/1e9:.1f} GHz")
    print(f"  Max intensity: {np.max(final_intensity):.6f}")
    print(f"  Total power: {np.sum(final_intensity):.6f}")

# After processing all frequencies, create comparison plots
print("\nGenerating frequency comparison plots...")

# Create comparison directory
comparison_dir = Path('results_between/frequency_comparisons')
comparison_dir.mkdir(parents=True, exist_ok=True)

# Extract data for plotting
freq_list = sorted(all_results.keys())
max_intensities = [all_results[f]['max_intensity'] for f in freq_list]
total_powers = [all_results[f]['total_power'] for f in freq_list]
power_losses = [all_results[f]['power_loss_percent'] for f in freq_list]
freq_ghz = [f/1e9 for f in freq_list]

# Plot max intensity vs frequency
plt.figure(figsize=(10, 6))
plt.plot(freq_ghz, max_intensities, 'b-o', linewidth=2, markersize=6)
plt.title('Maximum Intensity vs Frequency')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Maximum Intensity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(comparison_dir / 'max_intensity_vs_frequency.png', dpi=300)
plt.close()

# Plot total power vs frequency
plt.figure(figsize=(10, 6))
plt.plot(freq_ghz, total_powers, 'r-o', linewidth=2, markersize=6)
plt.title('Total Power vs Frequency')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Total Power')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(comparison_dir / 'total_power_vs_frequency.png', dpi=300)
plt.close()

# Plot power loss vs frequency
plt.figure(figsize=(10, 6))
plt.plot(freq_ghz, power_losses, 'g-o', linewidth=2, markersize=6)
plt.title('Total Power Loss vs Frequency')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Power Loss [%]')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(comparison_dir / 'power_loss_vs_frequency.png', dpi=300)
plt.close()

# Create side-by-side comparison of outputs at different frequencies
num_freq = len(freq_list)
fig, axes = plt.subplots(2, (num_freq + 1) // 2, figsize=(5 * ((num_freq + 1) // 2), 10))
if num_freq == 1:
    axes = axes.reshape(2, 1)
elif num_freq <= 2:
    axes = axes.reshape(2, 1) if num_freq == 1 else axes
axes = axes.flatten() if num_freq > 2 else axes.flatten()

for i, freq in enumerate(freq_list):
    if i < len(axes):
        intensity = all_results[freq]['final_intensity']
        im = axes[i].imshow(intensity, cmap='hot')
        axes[i].set_title(f'{freq/1e9:.1f} GHz\nMax: {np.max(intensity):.4f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(comparison_dir / 'all_frequencies_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate cross-section comparison
plt.figure(figsize=(12, 8))
center_y = all_results[freq_list[0]]['final_intensity'].shape[0] // 2

for freq in freq_list:
    intensity = all_results[freq]['final_intensity']
    plt.plot(intensity[center_y, :], label=f'{freq/1e9:.1f} GHz', linewidth=2)

plt.title(f'Intensity Cross-sections Comparison (y={center_y})')
plt.xlabel('x [pixels]')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(comparison_dir / 'cross_sections_comparison.png', dpi=300)
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("FREQUENCY RANGE ANALYSIS SUMMARY")
print("="*60)
print(f"Frequency range: {min(freq_ghz):.1f} - {max(freq_ghz):.1f} GHz")
print(f"Frequency step: {FREQUENCY_STEP/1e9:.1f} GHz")
print(f"Number of frequencies processed: {len(freq_list)}")
print()

for freq in freq_list:
    result = all_results[freq]
    print(f"Frequency {freq/1e9:.1f} GHz:")
    print(f"  Wavelength: {result['wavelength']*1e3:.3f} mm")
    print(f"  Max intensity: {result['max_intensity']:.6f}")
    print(f"  Total power: {result['total_power']:.6f}")
    print(f"  Power loss: {result['power_loss_percent']:.2f}%")
    print()

# Find optimal frequency
best_freq = max(freq_list, key=lambda f: all_results[f]['max_intensity'])
print(f"Highest intensity at: {best_freq/1e9:.1f} GHz (Max intensity: {all_results[best_freq]['max_intensity']:.6f})")

best_power_freq = max(freq_list, key=lambda f: all_results[f]['total_power'])
print(f"Highest total power at: {best_power_freq/1e9:.1f} GHz (Total power: {all_results[best_power_freq]['total_power']:.6f})")

# Find best and worst power loss frequencies
best_efficiency_freq = min(freq_list, key=lambda f: all_results[f]['power_loss_percent'])
worst_efficiency_freq = max(freq_list, key=lambda f: all_results[f]['power_loss_percent'])
print(f"Lowest power loss at: {best_efficiency_freq/1e9:.1f} GHz (Power loss: {all_results[best_efficiency_freq]['power_loss_percent']:.2f}%)")
print(f"Highest power loss at: {worst_efficiency_freq/1e9:.1f} GHz (Power loss: {all_results[worst_efficiency_freq]['power_loss_percent']:.2f}%)")

print("\nResults saved in:")
print("- Individual frequency results: results_between/frequency_[freq]GHz/")
print("- Comparison plots: results_between/frequency_comparisons/")
print("  - max_intensity_vs_frequency.png")
print("  - total_power_vs_frequency.png")
print("  - power_loss_vs_frequency.png")
print("  - all_frequencies_comparison.png")
print("  - cross_sections_comparison.png")
print("="*60)


