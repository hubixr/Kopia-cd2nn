import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pathlib import Path
import os as os
from PIL import Image

# Parametry przestrzeni
H, W = 128, 128
px_size_mm = 0.9
x = np.linspace(-(W//2)*px_size_mm, (W//2)*px_size_mm, W)
y = np.linspace(-(H//2)*px_size_mm, (H//2)*px_size_mm, H)
X, Y = np.meshgrid(x, y)

# --- 1. Generowanie jednego ogniska Gaussa ---
def generate_gaussian_targets(filename):
    center = [0, 0]  # Center of the Gaussian spot (in mm)
    sigma = 3        # Standard deviation (in mm)

    cov = [[sigma**2, 0], [0, sigma**2]]

    rv = multivariate_normal(mean=center, cov=cov)

    gaussian_map = rv.pdf(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(H, W)

    gaussian_map /= gaussian_map.max()  # Normalize to [0, 1]

    # Save the Gaussian map as a NPY file
    npy_filename = filename.with_suffix('.npy')
    np.save(npy_filename, gaussian_map)
    print(f"Saved Gaussian map as NPY to {npy_filename}")

    plt.imshow(gaussian_map, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spot")
    plt.colorbar()
    plt.savefig(filename.with_suffix('.png'))
    plt.close()

# --- 2. Generowanie x pól THz z artykułu "The collimated THz beam" ---
def generate_thz_inputs(folder, num_samples=2000):
    folder.mkdir(parents=True, exist_ok=True)

    # Wavelength parameters
    FREQUENCY_MIN = 160 * 1e9
    FREQUENCY_MAX = 200 * 1e9
    FREQUENCY_STEP = 10 * 1e9 
    C = 299792458  # [m/s]
    wavelength_min = C / (FREQUENCY_MAX)
    wavelength_max = C / (FREQUENCY_MIN)
    wavelength_step = C / (FREQUENCY_STEP)
    wavelengths = np.arange(wavelength_min, wavelength_max + wavelength_step, wavelength_step)

    idx = 0
    for wl in wavelengths:
        for i in range(num_samples):
            diameter_px = np.random.randint(50, 128)
            radius_px = diameter_px // 2
            aperture_mask = np.zeros((H, W), dtype=np.float32)
            center_x, center_y = W // 2, H // 2
            y_indices, x_indices = np.ogrid[:H, :W]
            distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
            aperture_mask[distance_from_center <= radius_px] = 1
            field = aperture_mask
            # Add wavelength as third channel
            field_with_wl = np.zeros((H, W, 3), dtype=np.float32)
            field_with_wl[..., 0] = field  # real part
            field_with_wl[..., 1] = 0      # imaginary part (if needed)
            field_with_wl[..., 2] = wl     # wavelength
            npy_filename = folder / f"field_{idx:05d}.npy"
            np.save(npy_filename, field_with_wl)
            print(f"Saved THz input field {idx} with wavelength {wl} as NPY to {npy_filename}")
            idx += 1

    print("umin=", field.min())
if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    # generate_gaussian_targets(output_folder / "target_field")  # Disabled target field generation
    generate_thz_inputs(output_folder / "input_fields")

    # Visualize the first 10 inputs on one graph
    input_folder = output_folder / "input_fields"
    plt.figure(figsize=(15, 6))
    for i in range(10):
        npy_input_file = input_folder / f"field_{i:05d}.npy"
        input_field = np.load(npy_input_file)
        # Show only the real part
        plt.subplot(2, 5, i + 1)
        plt.imshow(input_field[..., 0], cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
        wl = input_field[0, 0, 2]
        plt.title(f"Input Field {i}\nwl={wl:.3f}")
        plt.colorbar()
    plt.tight_layout()
    plt.show()
