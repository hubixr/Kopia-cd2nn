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
        # Kolimowana wiązka THz modelowana jako Gauss z lekkim odchyleniem
        waist = np.random.uniform(15, 25)  # mm
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

    print("umin=", U.min())
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
