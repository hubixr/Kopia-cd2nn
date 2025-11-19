# --- 4. Generowanie x pól Gaussa jako wejścia z kanałem długości fali i zapis do npy ---
def generate_gaussian_inputs_npy(folder, num_samples=15):
    # Wavelength parameters from frequency config
    FREQUENCY_MIN = 130 * 1e9
    FREQUENCY_MAX = 200 * 1e9
    FREQUENCY_STEP = 0.5 * 1e9
    freq_step_parameter = (FREQUENCY_MAX - FREQUENCY_MIN) / FREQUENCY_STEP

    C = 299792458  # [m/s]
    wavelength_min = C / (FREQUENCY_MAX)
    wavelength_max = C / (FREQUENCY_MIN)
    wavelength_step = (wavelength_max - wavelength_min) / freq_step_parameter
    print("wavelength min:", wavelength_min)
    print("wavelength max:", wavelength_max)
    print("wavelength step:", wavelength_step)
    wavelengths = np.arange(wavelength_min, wavelength_max + wavelength_step, wavelength_step)
        # Debug: Print wavelengths and their count
    print(wavelengths)
    print(len(wavelengths))
    folder.mkdir(parents=True, exist_ok=True)
    idx = 0
    for wl in wavelengths:
        for i in range(num_samples):
            radius_px = np.random.randint(32, 65)
            sigma = radius_px / 2 * px_size_mm
            center = [0, 0]
            cov = [[sigma**2, 0], [0, sigma**2]]
            rv = multivariate_normal(mean=center, cov=cov)
            gaussian_map = rv.pdf(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(H, W)
            gaussian_map /= gaussian_map.max()
            arr = np.zeros((H, W, 3), dtype=np.float32)
            arr[..., 0] = gaussian_map
            arr[..., 1] = 0.0
            arr[..., 2] = wl
            npy_filename = folder / f"field_{idx:05d}.npy"
            np.save(npy_filename, arr)
            print(f"Saved Gaussian input field {idx} (radius ~{radius_px}px, sigma={sigma:.2f}mm, wl={wl}) as npy to {npy_filename}")
            idx += 1

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


if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    generate_gaussian_inputs_npy(output_folder / "input_fields")

    # Visualize the first 10 inputs on one graph (from npy)
    input_folder = output_folder / "input_fields"
    plt.figure(figsize=(15, 6))
    for i in range(10):
        npy_input_file = input_folder / f"field_{i:05d}.npy"
        input_field = np.load(npy_input_file)
        plt.subplot(2, 5, i + 1)
        plt.imshow(input_field[..., 0], cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
        plt.title(f"Input Field {i} (wl={input_field[0,0,2]:.3f})")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


