import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pathlib import Path
import os as os

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


    np.save(filename, gaussian_map)
    plt.imshow(gaussian_map, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spot")
    plt.colorbar()
    plt.savefig(filename.with_suffix('.png'))
    plt.close()

# --- 2. Generowanie x pól THz z artykułu "The collimated THz beam" ---
def generate_thz_inputs(folder, num_samples=200):
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Kolimowana wiązka THz modelowana jako Gauss z lekkim odchyleniem
        waist = np.random.uniform(20, 30)  # mm
        x0 = np.random.uniform(-10, 10)    # mm
        y0 = np.random.uniform(-10, 10)
        theta = np.random.uniform(-0.05, 0.05)  # nachylenie fazy w rad/mm

        amp = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * waist**2))
        phase = theta * X
        field = amp * np.exp(1j * phase)

        U = np.stack([np.real(field), np.imag(field)], dtype=np.float32, axis=-1)
        U[U < 0] = 0
        
        np.save(folder / f"field_{i:04d}.npy", U)
    print("umin=", U.min())
if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    generate_gaussian_targets(output_folder / "target_field.npy")
    generate_thz_inputs(output_folder / "input_fields")

    # Visualize the first 10 inputs and targets on one graph
    input_folder = output_folder / "input_fields"
    target_file = output_folder / "target_field.npy"

    # Load the target field
    target_field = np.load(target_file)

    # # Plot the target field
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(target_field, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Gaussian Spot")
    plt.colorbar()

    # Plot the first 10 input fields
    for i in range(10):
        input_field = np.load(input_folder / f"field_{i:04d}.npy")
        real_part = input_field[..., 0]
        imag_part = input_field[..., 1]

        # Real part
        plt.subplot(3, 4, i + 2)
        plt.imshow(real_part, cmap='viridis', extent=[x[0], x[-1], y[0], y[-1]])
        plt.title(f"Input Field {i} - Real Part")
        plt.colorbar()

    plt.tight_layout()
    plt.show()
