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

# --- 1. Generowanie litery H jako celu ---
def generate_letter_target(filename):
    letter_size = (50, 50)  # Size of the letter H in pixels
    letter_image = np.zeros((H, W), dtype=np.float32)

    # Define the letter H in the center of the image
    start_x, start_y = (H - letter_size[0]) // 2, (W - letter_size[1]) // 2
    end_x, end_y = start_x + letter_size[0], start_y + letter_size[1]

    # Draw the vertical bars of H
    letter_image[start_x:end_x, start_y:start_y + 10] = 1.0  # Left bar
    letter_image[start_x:end_x, end_y - 10:end_y] = 1.0  # Right bar

    # Draw the horizontal bar of H
    letter_image[start_x + letter_size[0] // 2 - 5:start_x + letter_size[0] // 2 + 5, start_y:end_y] = 1.0

    # Normalize and save the letter image
    letter_image /= letter_image.max()
    np.save(filename, letter_image)
    plt.imshow(letter_image, cmap='hot', extent=[x[0], x[-1], y[0], y[-1]])
    plt.title("Target Letter H")
    plt.colorbar()
    plt.savefig(filename.with_suffix('.png'))
    plt.close()

    # Save the letter image as a grayscale BMP file
    bmp_filename = filename.with_suffix('.bmp')
    scaled_letter_image = (letter_image * 255).astype(np.uint8)  # Scale to 0-255
    gray_image = Image.fromarray(scaled_letter_image, mode='L')
    gray_image.save(bmp_filename)
    print(f"Target saved as grayscale BMP: {bmp_filename}")

# --- 2. Generowanie x pól THz z artykułu "The collimated THz beam" ---
def generate_thz_inputs(folder, num_samples=1000):
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

if __name__ == "__main__":
    os.makedirs("./cdnn_data", exist_ok=True)
    output_folder = Path("./cdnn_data")
    generate_letter_target(output_folder / "target_field.npy")
    generate_thz_inputs(output_folder / "input_fields")
