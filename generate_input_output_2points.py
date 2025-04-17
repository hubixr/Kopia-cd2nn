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

# --- 1. Generowanie dwóch ognisk Gaussa ---
def generate_gaussian_targets(filename):
    center1 = [-20, 0]  # mm
    center2 = [20, 0]   # mm
    sigma = 6           # mm  # Increased sigma to make the points 2 times bigger

    cov = [[sigma**2, 0], [0, sigma**2]]

    rv1 = multivariate_normal(mean=center1, cov=cov)
    rv2 = multivariate_normal(mean=center2, cov=cov)

    gaussian_map = rv1.pdf(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(H, W)
    gaussian_map += rv2.pdf(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(H, W)

    gaussian_map /= gaussian_map.max()

    np.save(filename, gaussian_map)
    plt.imshow(gaussian_map, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spots")
    plt.colorbar()
    plt.savefig(filename.with_suffix('.png'))
    plt.close()

    # Save the Gaussian map as a grayscale BMP file
    bmp_filename = filename.with_suffix('.bmp')
    scaled_gaussian_map = (gaussian_map * 255).astype(np.uint8)  # Scale to 0-255
    gray_image = Image.fromarray(scaled_gaussian_map, mode='L')
    gray_image.save(bmp_filename)
    print(f"Target saved as grayscale BMP: {bmp_filename}")

# --- 2. Generowanie x pól THz z artykułu "The collimated THz beam" ---
def generate_thz_inputs(folder, num_samples=200):
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Kolimowana wiązka THz modelowana jako Gauss z lekkim odchyleniem
        waist = np.random.uniform(25, 35)  # mm
        x0 = np.random.uniform(-10, 10)    # mm
        y0 = np.random.uniform(-10, 10)
        theta = np.random.uniform(-0.05, 0.05)  # nachylenie fazy w rad/mm

        amp = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * waist**2))
        phase = theta * X
        field = amp * np.exp(1j * phase)

        # # Resize the field to fit within H, W
        # field_resized = np.zeros((H, W), dtype=np.complex64)
        # start_x = (H - field.shape[0]) // 2
        # start_y = (W - field.shape[1]) // 2
        # field_resized[start_x:start_x + field.shape[0], start_y:start_y + field.shape[1]] = field
        # field = field_resized

        U = np.stack([np.real(field), np.imag(field)], dtype=np.float32, axis=-1)
        U[U < 0] = 0
        
        # Removed saving as .npy file
        # np.save(folder / f"field_{i:04d}.npy", U)

        # Save the THz input field as a grayscale BMP file
        bmp_filename = folder / f"field_{i:04d}.bmp"
        plt.imsave(bmp_filename, np.abs(field), cmap='gray')
        print(f"Saved THz input field {i} as grayscale BMP to {bmp_filename}")

if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    generate_gaussian_targets(output_folder / "target_field.npy")
    generate_thz_inputs(output_folder / "input_fields")
