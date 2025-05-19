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

    # Save the Gaussian map as a grayscale BMP file
    bmp_filename = filename.with_suffix('.bmp')
    plt.imsave(bmp_filename, gaussian_map, cmap='gray')
    print(f"Saved Gaussian map as grayscale BMP to {bmp_filename}")

    plt.imshow(gaussian_map, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spot")
    plt.colorbar()
    plt.savefig(filename.with_suffix('.png'))
    plt.close()

# --- 2. Generowanie x pól THz z artykułu "The collimated THz beam" ---
def generate_thz_inputs(folder, num_samples=500):
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Plane wave with apertures of random diameters between 40 and 128px
        diameter_px = np.random.randint(50, 128)  # Random diameter in pixels
        radius_px = diameter_px // 2

        # Create a circular aperture mask
        aperture_mask = np.zeros((H, W), dtype=np.float32)
        center_x, center_y = W // 2, H // 2
        y_indices, x_indices = np.ogrid[:H, :W]
        distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        aperture_mask[distance_from_center <= radius_px] = 1

        # Plane wave field
        field = aperture_mask

        # Save the THz input field as a grayscale BMP file
        bmp_filename = folder / f"field_{i:04d}.bmp"
        plt.imsave(bmp_filename, field, cmap='gray')
        print(f"Saved THz input field {i} as grayscale BMP to {bmp_filename}")

    print("umin=", field.min())
if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    generate_gaussian_targets(output_folder / "target_field")
    generate_thz_inputs(output_folder / "input_fields")

    # Visualize the first 10 inputs and targets on one graph
    input_folder = output_folder / "input_fields"
    target_file = output_folder / "target_field.bmp"

    # Update to load the target field from the BMP file instead of .npy
    bmp_target_file = output_folder / "target_field.bmp"
    target_field = plt.imread(bmp_target_file)  # Load BMP file as an array

    # Normalize the target field to [0, 1] if needed
    target_field = target_field / 255.0 if target_field.max() > 1 else target_field

    # # Plot the target field
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(target_field, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spot")
    plt.colorbar()

    # Plot the first 10 input fields
    for i in range(10):
        bmp_input_file = input_folder / f"field_{i:04d}.bmp"
        input_field = Image.open(bmp_input_file).convert('L')  # Load BMP file as grayscale
        input_field = np.array(input_field, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Plot the input field
        plt.subplot(3, 4, i + 2)
        plt.imshow(input_field, cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
        plt.title(f"Input Field {i}")
        plt.colorbar()

    plt.tight_layout()
    plt.show()
