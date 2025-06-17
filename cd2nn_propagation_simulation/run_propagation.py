import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from cd2nn_model import CDNNModel
import time
# from tensorflow.keras import mixed_precision
from PIL import Image
import argparse

# mixed_precision.set_global_policy('float32')

# ================================
# PARAMETRY UKLADU
# ================================
DOE_SHAPE = (128, 128)
PIXEL_SIZE = 9e-4  # [m]
FREQUENCY = 96 * 1e9  # [GHz]
C = 299792458  # [m/s]
WAVELENGTH = C / (FREQUENCY)  # [m]
print("Wavelength:", WAVELENGTH)
PROPAGATION_DISTANCE_BEETWEEN_DOE = 0.1  # [m]
PROPAGATION_DISTANCE_TO_TARGET = 0.2 # [m]
NUM_LAYERS = 1
EPOCHS = 2000

DATA_DIR = Path("./cdnn_data")
INPUT_DIR = DATA_DIR / "input_fields"
TARGET_FILE = DATA_DIR / "target_field.npy"

# ================================
# Parse command-line arguments
# ================================
parser = argparse.ArgumentParser()
parser.add_argument('--mask_path', type=str, default="validation_data_lenses/phase_mask/best_trained_doe_phase_1_PSNR_48.69_freq_96.000GHz_batch_1_layers_2_epochs_3_lr_0.030_dist_doe_0.100_dist_target_0.200_doe_shape_128x128.bmp", help='Path to the phase mask image (.bmp)')
parser.add_argument('--output_path', type=str, default="./outputs", help='Path to save the output .npy file')
# ... add other arguments as needed ...
args = parser.parse_args()
mask_path = args.mask_path
output_path = args.output_path

# ================================
# FUNKCJE POMOCNICZE
# ================================
# Function to load .bmp file and preprocess it for the model
def load_bmp_as_input(file_path, target_shape):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    image = image.resize(target_shape, Image.Resampling.LANCZOS)  # Resize to target shape using LANCZOS
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize to 0-1
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

# ================================
# BUDOWA MODELU
# ================================
print("Budowanie modelu CDNN...")
model = CDNNModel(
    num_layers=NUM_LAYERS,
    phase_mask = mask_path,
    shape=DOE_SHAPE,
    wavelength=WAVELENGTH,
    distance_to_plane=PROPAGATION_DISTANCE_TO_TARGET,
    distance_between_layers=PROPAGATION_DISTANCE_BEETWEEN_DOE,
    pixel_size=PIXEL_SIZE
)

loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.003, clipnorm=1.0)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
distance_in_mm = PROPAGATION_DISTANCE_TO_TARGET * 1000
frequency_in_GHz = FREQUENCY / 1e9

# Load and preprocess the input plane and phase mask from .bmp files
input_plane_path = "validation_data_lenses/input/Input_px_0.9mm_size_128_frequency96GHz_f_200mm.bmp"  # Replace with the actual path to your input plane .bmp file

# Load input plane
input_plane = load_bmp_as_input(input_plane_path, DOE_SHAPE)  # Preprocess to match model input shape
print("input_plane shape:", input_plane.shape)


# Create the input tensor with real and imaginary parts
real_part = input_plane[..., 0]  # Real part from the input plane
print("real_part shape:", real_part.shape)
print("real_part min:", np.min(real_part))
print("real_part max:", np.max(real_part))
imaginary_part = np.zeros_like(real_part)  # Imaginary part initialized to 0
print("imaginary_part shape:", imaginary_part.shape)
print("imaginary_part min:", np.min(imaginary_part))
print("imaginary_part max:", np.max(imaginary_part))
input_tensor = np.stack([real_part, imaginary_part], axis=-1)  # Shape: [H, W, 2]
input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension, Shape: [1, H, W, 2]
print("input_tensor shape:", input_tensor.shape)
# Pass the input tensor through the model
output = model(input_tensor).numpy()
# model.summary()

# Visualize the output and save to file
plt.imshow(output[0], cmap='hot')
plt.title("Output Intensity")
plt.colorbar()
output_file = "output_intensity.png"
plt.savefig(output_file)
plt.close()
print(f"Output intensity visualization saved to {output_file}")

# Function to calculate optical power of an image
# Optical power is proportional to the sum of squared pixel intensities
def calculate_optical_power(image_array):
    return np.sum(image_array**2)

# Calculate and compare power loss between input and output during evaluation
def calculate_power_loss(input_data, output_data):
    input_power = calculate_optical_power(input_data)
    output_power = calculate_optical_power(output_data)
    power_loss = input_power - output_power
    power_loss_ratio = (power_loss / input_power) * 100  # Percentage loss
    return input_power, output_power, power_loss, power_loss_ratio

# ================================
# Calculate power before normalization
# ================================
print("Calculating power before normalization...")
input_power = calculate_optical_power(real_part)
print(f"Input Power = {input_power:.2f}")

# Denormalize the output before calculating power
output_denormalized = output[0] * 255.0  # Assuming the output was normalized to 0-1

# Calculate power after denormalization
output_power = calculate_optical_power(output_denormalized)
print(f"Output Power (Denormalized) = {output_power:.2f}")

# Compare power loss
power_loss = input_power - output_power
power_loss_ratio = (power_loss / input_power) * 100
print(f"Power Loss = {power_loss:.2f}, Power Loss Ratio = {power_loss_ratio:.2f}%")

# ================================
# Save intensity and amplitude to output_path
# ================================
intensity = np.abs(output[0])
amplitude = np.sqrt(intensity)
np.save(output_path, {'intensity': intensity, 'amplitude': amplitude})
print(f"Propagation results saved to {output_path}")

# ================================
# Save heatmaps as images
# ================================
import matplotlib.pyplot as plt
import os

def save_heatmap(array, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(array, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Save intensity heatmap
intensity_img_path = os.path.splitext(output_path)[0] + '_intensity_heatmap.png'
save_heatmap(intensity, intensity_img_path, 'Output Intensity Heatmap')
print(f"Saved intensity heatmap to {intensity_img_path}")

# Save amplitude heatmap
amplitude_img_path = os.path.splitext(output_path)[0] + '_amplitude_heatmap.png'
save_heatmap(amplitude, amplitude_img_path, 'Output Amplitude Heatmap')
print(f"Saved amplitude heatmap to {amplitude_img_path}")
