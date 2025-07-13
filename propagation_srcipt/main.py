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
DISTANCE = 0.11 #[m]
DISTANCE_TO_TARGET = 0.21 #[m]
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
plt.show()

#input field as U
U = np.stack([input_norm, np.zeros_like(input_norm)], axis=-1)  # Two channels: input_norm and zeros

# Dodaj zero padding do U
pad_size = int(H / (PADDING_MULTIPLIER + 1) * PADDING_MULTIPLIER / 2)
U_padded = np.pad(U, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
print(f'U_padded shape: {U_padded.shape}')


for i in range(num_masks):
    if i == num_masks - 1:
        distance = DISTANCE_TO_TARGET  # Last mask uses distance to target
    else:
        distance = DISTANCE  # Other masks use the initial distance
    print(f"Processing mask {i + 1}/{num_masks} with distance {distance:.3f} m")
    re_u = U_padded[..., 0]
    im_u = U_padded[..., 1]
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

    print("re_u shape:", re_u.shape)
    print("im_u shape:", im_u.shape)

    print("h_real shape:", h_real.shape)
    print("h_imag shape:", h_imag.shape)
    N = re_u.shape[0]  # Assuming square input for simplicity
    print("Convolution with phase mask", i + 1)
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
        U_padded = np.stack([out_real, out_imag], axis=-1)
    else:
        U_padded = np.stack([out_real_cropped, out_imag_cropped], axis=-1)

    print(f"Output field after mask {i + 1} shape: {U_padded.shape}")
    # Collect output intensities for each mask
    if i == 0:
        output_intensities = []
    output_intensity = out_real_cropped**2 + out_imag_cropped**2
    # output_intensity = output_intensity / output_intensity.max()  # Normalize to [0,1] (disabled)
    output_intensities.append(output_intensity)

    # Save the output field (intensity) as a colormap image (hot colormap)
    output_colormap = plt.get_cmap('hot')(output_intensity / output_intensity.max() if output_intensity.max() > 0 else output_intensity)
    output_colormap_img = (output_colormap[..., :3] * 255).astype(np.uint8)  # Drop alpha, scale to [0,255]

    results_dir = Path('results_between')
    results_dir.mkdir(exist_ok=True)
    output_img = Image.fromarray(output_colormap_img)
    output_img.save(results_dir / f'cmap_after_mask_{i+1}.png')

    # Plot intensity cross-section through the center (y=64)
    center_y = output_intensity.shape[0] // 2
    plt.figure(figsize=(8, 4))
    plt.plot(output_intensity[center_y, :], label=f'Output after mask {i+1} (y={center_y})')
    plt.title(f'Intensity cross-section (y={center_y}) after mask {i+1}')
    plt.xlabel('x')
    plt.ylabel('Normalized intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f'cross_section_y{center_y}_after_mask_{i+1}.png')
    plt.close()

    # After all iterations, plot all outputs in one plot
    if i == num_masks - 1:
        fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 5))
        if num_masks == 1:
            axes = [axes]
        for j, inten in enumerate(output_intensities):
            axes[j].imshow(inten, cmap='hot')
            axes[j].set_title(f'Output after mask {j+1}\n{inten.shape[1]}x{inten.shape[0]} px')
            axes[j].axis('off')
            plt.colorbar(axes[j].images[0], ax=axes[j], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


