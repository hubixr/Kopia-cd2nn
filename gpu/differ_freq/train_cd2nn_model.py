import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from cd2nn_model import CDNNModel
from DiffractiveMaskLayer import DiffractiveMaskLayer
import time
# from tensorflow.keras import mixed_precision
from PIL import Image
from PIL import ImageOps

# Clear VRAM and reset TensorFlow session
tf.keras.backend.clear_session()

# mixed_precision.set_global_policy('float32')


# ================================
# PARAMETRY UKLADU
# ================================
DOE_SHAPE = (128, 128)  # [px]
PIXEL_SIZE = 9e-4  # [m]
# FREQUENCY = 96 * 1e9  # [GHz]
C = 299792458  # [m/s]
# WAVELENGTH = C / (FREQUENCY)  # [m]

PROPAGATION_DISTANCE_BEETWEEN_DOE = 0.05  # [m]
PROPAGATION_DISTANCE_TO_TARGET = 0.2  # [m]
NUM_LAYERS = 2
EPOCHS = 1000
# ================================
# Wavelength from range
FREQUENCY_MIN = 160 * 1e9
FREQUENCY_MAX = 200 * 1e9
FREQUENCY_STEP = 0.5 * 1e9
WAVELENGTH_MIN = C / (FREQUENCY_MAX)
WAVELENGTH_MAX = C / (FREQUENCY_MIN)
WAVELENGTH_STEP = (WAVELENGTH_MAX - WAVELENGTH_MIN) / 21
print("Wavelength:", WAVELENGTH_MIN)
print("Wavelength:", WAVELENGTH_MAX)
print("Wavelength:", WAVELENGTH_STEP)
# ================================
LEARNING_RATE = 0.1                     # ↑ Faster convergence but less stable | ↓ Slower but more stable training
BATCH_SIZE = 32                       # ↑ Smoother gradients, more memory | ↓ Noisier gradients, less memory
CALLBACK_PATIENCE = 10                 # ↑ Train longer before early stop | ↓ Stop training sooner if no improvement
CALLBACK_MIN_DELTA = 5e-2             # ↑ Require larger improvement to continue | ↓ Continue with smaller improvements (default 1e-4)
SMOOTHNESS_WEIGHT = 1e-5              # ↑ Smoother phase patterns | ↓ Allow more dramatic phase variations
POWER_LOSS_WEIGHT = 1.2                 # ↑ Prioritize power efficiency | ↓ Allow more power loss for better focusing (default 1)
FOCAL_INTENSITY_WEIGHT = 0.2          # ↑ Stronger focus at center | ↓ Less emphasis on central focusing
USE_ALL_LAYERS_POWER_LOSS = True      # True: Consider all layer losses | False: Only final layer power loss
# ================================
# SMOOTHNESS FUNCTION WEIGHTS - MODIFIED FOR KINOFORM-LIKE PATTERNS
# ================================
SMOOTHNESS_TRADITIONAL_WEIGHT = 0.1   # ↑ Penalize neighbor phase differences more | ↓ Allow sharper phase transitions
SMOOTHNESS_VARIATION_WEIGHT = 0.5     # ↑ Enforce uniform local phase variation | ↓ Allow varied local phase patterns
SMOOTHNESS_BINARY_WEIGHT = 0.1       # ↑ Discourage 0/2π phase values more | ↓ Allow more binary-like phase patterns
SMOOTHNESS_TARGET_STD_PERCENT = 0.1   # ↑ Encourage larger local variations | ↓ Prefer smaller local phase variations (10% of 2π)
# ================================
DATA_DIR = Path("./cdnn_data")
INPUT_DIR = DATA_DIR / "input_fields"
TARGET_FILE = DATA_DIR / "target_field.bmp"

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Set a manual memory limit (in MB) for each GPU
        memory_limit_mb = 40960  # 40GB limit
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
        print(f"Memory limit of {memory_limit_mb} MB set for GPUs.")
    except RuntimeError as e:
        print("Error setting memory limit:", e)
else:
    print("No GPUs found.")
# ================================
# FUNKCJE POMOCNICZE
# ================================

# Function to load .bmp file and preprocess it for the model
def load_bmp_fields(file_path, target_shape):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    # image = image.resize(target_shape, Image.Resampling.LANCZOS)  # Resize to target shape using LANCZOS
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize to 0-1
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

def load_bmp_target_field(target_file, shape):
    image = Image.open(target_file).convert('L')  # Convert to grayscale
    image = image.resize(shape, Image.Resampling.LANCZOS)  # Resize to target shape
    target_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to 0-1
    target_array = np.expand_dims(target_array, axis=0)  # Add batch dimension
    # Ensure the target array has a channel dimension
    target_array = np.expand_dims(target_array, axis=-1)  # Add channel dimension
    # print("target_array shape:", target_array.shape)
    return target_array

# Modify the input data to have two channels: one with the BMP values and the second filled with zeros
def add_zero_channel(input_data):
    zero_channel = np.zeros(input_data.shape + (1,), dtype=input_data.dtype)  # Create a channel of zeros
    input_data = np.expand_dims(input_data, axis=-1)  # Add a channel dimension to the BMP values
    return np.concatenate((input_data, zero_channel), axis=-1)  # Concatenate along the last axis

# Ensure input data is resized or cropped to (128, 128)
def crop_or_resize_input(input_data, target_shape):
    cropped_data = input_data[:, :target_shape[0], :target_shape[1]]  # Crop to target shape
    return cropped_data

# def load_npy_fields(input_dir):
#     files = sorted(input_dir.glob("*.npy"))
#     inputs = []
#     for f in files:
#         field = np.load(f)  # Load .npy file directly
#         inputs.append(field)
#     if not inputs:
#         raise ValueError("No valid input fields found in the directory.")
#     return np.stack(inputs, axis=0)

# ================================
# GLOWNA CZESC
# ================================
print("Laduję dane wejściowe...")
input_files = sorted(INPUT_DIR.glob("*.npy"))
inputs = []
for f in input_files:
    arr = np.load(f)
    # Ensure shape is (H, W, 3)
    if arr.shape[-1] != 3:
        raise ValueError(f"Input file {f} does not have 3 channels.")
    # Optionally crop/resize if needed
    arr = crop_or_resize_input(arr, DOE_SHAPE)
    inputs.append(arr)
if not inputs:
    raise ValueError("No valid input fields found in the directory.")
input_data = np.stack(inputs, axis=0).astype(np.float32)
print(f"Input data shape: {input_data.shape}")
# After loading input_data
print("Example input wavelength channel (first sample):")
print(input_data[0, :, :, 2])

print(f"Liczba próbek: {input_data.shape[0]}")
print("Laduję target...")
target_data = load_bmp_target_field(TARGET_FILE, DOE_SHAPE).astype(np.float32)
num_samples = input_data.shape[0]
targets = np.repeat(target_data, num_samples, axis=0)
print("targets shape", targets.shape)

# ================================
# PODZIAŁ NA ZBIORY
# ================================
indices = np.arange(num_samples)
np.random.shuffle(indices)

train_end = int(0.7 * num_samples)
val_end = int(0.85 * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# Use the centered data for training
x_train, y_train = input_data[train_idx], targets[train_idx]
x_val, y_val = input_data[val_idx], targets[val_idx]
x_test, y_test = input_data[test_idx], targets[test_idx]
# Normalize data to 0-1
x_train = ((x_train / np.max(np.abs(x_train))))
x_val = ((x_val / np.max(np.abs(x_val))))
x_test = ((x_test / np.max(np.abs(x_test))))

# print("y_test min:", y_test.min())
# print("y_test max:", y_test.max())
# print("y_train mean:", y_train.mean())

# Debugging: Check input data for NaN or large values
# print("Input data stats:")
# print(f"x_train min: {np.min(x_train)}, max: {np.max(x_train)}, mean: {np.mean(x_train)}")
# print(f"x_val min: {np.min(x_val)}, max: {np.max(x_val)}, mean: {np.mean(x_val)}")
# print(f"x_test min: {np.min(x_test)}, max: {np.max(x_test)}, mean: {np.mean(x_test)}")

# Debugging: Check target data for NaN or large values
# print("Target data stats:")
# print(f"y_train min: {np.min(y_train)}, max: {np.max(y_train)}, mean: {np.mean(y_train)}")
# print(f"y_val min: {np.min(y_val)}, max: {np.max(y_val)}, mean: {np.mean(y_val)}")
# print(f"y_test min: {np.min(y_test)}, max: {np.max(y_test)}, mean: {np.mean(y_test)}")

# ================================
# BUDOWA MODELU
# ================================
print("Budowanie modelu CDNN...")
model = CDNNModel(
    num_layers=NUM_LAYERS,
    shape=DOE_SHAPE,
    wavelength_min=WAVELENGTH_MIN,
    wavelength_max=WAVELENGTH_MAX,
    wavelength_step=WAVELENGTH_STEP,
    distance_to_plane=PROPAGATION_DISTANCE_TO_TARGET,
    distance_between_layers=PROPAGATION_DISTANCE_BEETWEEN_DOE,
    pixel_size=PIXEL_SIZE
)
# Print all layers with their distance to the next layer
print(f"Number of layers created: {len(model.prop_layers)}")
for i, prop_layer in enumerate(model.prop_layers):
    print(f"Layer {i+1}: Distance to next layer = {prop_layer.distance} m")


print(f"Power loss mode: {'All layers' if USE_ALL_LAYERS_POWER_LOSS else 'Final layer only'}")
print(f"Power loss weight: {POWER_LOSS_WEIGHT}")

loss_fn = tf.keras.losses.MeanSquaredError()  # Absolute Mean Error
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0) #clipnorm for gradient clipping - better stability


def psnr_metric(y_true, y_pred):
    # Ensure both tensors have the same shape
    if len(y_true.shape) == 4 and len(y_pred.shape) == 3:
        # y_true has batch dimension, y_pred doesn't - add batch dimension to y_pred
        y_pred = tf.expand_dims(y_pred, axis=0)
    elif len(y_true.shape) == 3 and len(y_pred.shape) == 4:
        # y_pred has batch dimension, y_true doesn't - squeeze y_pred
        y_pred = tf.squeeze(y_pred, axis=0)
    
    # Ensure both have channel dimension
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def calculate_power(y):
    return tf.reduce_sum(tf.square(y), axis=[1, 2])

lambda_smooth = SMOOTHNESS_WEIGHT  # Weight for smoothness regularization def 1e-6
lambda_power = POWER_LOSS_WEIGHT
def smoothness_regularization(phase):
    """
    Simplified smoothness regularization that encourages intermediate phase values
    and penalizes both no variation and extreme jumps.
    
    Args:
        phase: Tensor of shape (H, W), the phase mask in range [0, 2π].
    Returns:
        Smoothness regularization term (scalar).
    """
    pi2 = tf.constant(2 * np.pi, dtype=phase.dtype)
    
    # Ensure phase is in [0, 2π] range
    phase = tf.math.floormod(phase, pi2)
    
    # Part 1: Traditional smoothness (8-neighbor differences with periodic wrapping)
    phase_padded = tf.pad(phase, [[1, 1], [1, 1]], mode='REFLECT')
    
    # All 8 neighbors
    neighbors = [
        phase_padded[0:-2, 1:-1],  # top
        phase_padded[2:  , 1:-1],  # bottom
        phase_padded[1:-1, 0:-2],  # left
        phase_padded[1:-1, 2:  ],  # right
        phase_padded[0:-2, 0:-2],  # top-left
        phase_padded[0:-2, 2:  ],  # top-right
        phase_padded[2:  , 0:-2],  # bottom-left
        phase_padded[2:  , 2:  ],  # bottom-right
    ]
    
    smoothness_term = 0
    for n in neighbors:
        diff = phase - n
        # Wrap phase differences to [-π, π] for proper periodic boundary handling
        diff = tf.math.atan2(tf.sin(diff), tf.cos(diff))
        smoothness_term += tf.reduce_mean(tf.square(diff))
    
    # Part 2: Variation encouragement (penalize flat regions)
    phase_padded_3x3 = tf.pad(phase, [[1, 1], [1, 1]], mode='REFLECT')
    
    # Extract 3x3 patches using tf.image.extract_patches
    patches = tf.image.extract_patches(
        tf.expand_dims(tf.expand_dims(phase_padded_3x3, 0), -1),  # Add batch and channel dims
        sizes=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # Shape: [1, H, W, 9]
    
    patches = tf.squeeze(patches, axis=0)  # Remove batch dim: [H, W, 9]
    
    # Calculate local standard deviation
    local_mean = tf.reduce_mean(patches, axis=-1)  # [H, W]
    local_variance = tf.reduce_mean(tf.square(patches - tf.expand_dims(local_mean, -1)), axis=-1)  # [H, W]
    local_std = tf.sqrt(local_variance + 1e-8)
    
    # Target standard deviation (10% of 2π range)
    target_std = SMOOTHNESS_TARGET_STD_PERCENT * pi2
    variation_penalty = tf.reduce_mean(tf.square(local_std - target_std))
    
    # Part 3: Binary pattern penalty (discourage 0 and 2π values)
    normalized_phase = phase / pi2
    # sin(2π * normalized_phase) is maximum at 0 and 2π, minimum at π
    binary_penalty = tf.reduce_mean(tf.square(tf.sin(pi2 * normalized_phase)))
    
    # Combine all terms
    total_penalty = (
        SMOOTHNESS_TRADITIONAL_WEIGHT * smoothness_term + 
        SMOOTHNESS_VARIATION_WEIGHT * variation_penalty + 
        SMOOTHNESS_BINARY_WEIGHT * binary_penalty
    )
    
    return total_penalty

def custom_loss_with_model(model):
    def custom_loss(y_true, y_pred):
        # Compute the standard loss (e.g., Mean Squared Error)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Add smoothness regularization for each phase mask
        smoothness_loss = 0
        for i, layer in enumerate(model.doe_layers):
            layer_smoothness = smoothness_regularization(layer.phase)
            smoothness_loss += layer_smoothness
            # Debug: Print smoothness loss for each layer during training
            # if i == 0:  # Only print for debugging, can remove later
                # tf.print(f"Layer {i+1} smoothness loss:", layer_smoothness)
        
        # Power loss calculation - choose between all layers or only final layer
        if USE_ALL_LAYERS_POWER_LOSS:
            # Power loss from all propagation layers - calculate cumulatively
            remaining_power = 1.0  # Start with 100% power
            for i, power_loss in enumerate(model.all_power_losses):
                layer_power_loss = tf.reduce_mean(power_loss)
                remaining_power = remaining_power * (1.0 - layer_power_loss)  # Apply sequential loss
                # Optional: print power loss for each layer during training
                # tf.print(f"Layer {i+1} power loss:", layer_power_loss * 100, "%")
            
            total_power_loss = 1.0 - remaining_power  # Total cumulative power loss
            # tf.print("Total power loss:", total_power_loss * 100, "%")
            power_loss_term = total_power_loss
        else:
            # Power loss from final layer only (original behavior)
            power_loss_term = tf.reduce_mean(model.last_power_loss)
            # tf.print("Final layer power loss:", power_loss_term * 100, "%")

        # Focal intensity: mean value in a 10x10 px window at the center of the field
        shape = tf.shape(y_pred)
        center_y = shape[1] // 2
        center_x = shape[2] // 2
        window_size = 10
        half_window = window_size // 2
        # Slicing: [center_y-half_window:center_y+half_window, center_x-half_window:center_x+half_window]
        focal_patch = y_pred[:, 
                             center_y-half_window:center_y+half_window, 
                             center_x-half_window:center_x+half_window]
        focal_intensity = tf.reduce_mean(focal_patch) 

        # Combine the losses (add a negative sign to maximize focal intensity)
        total_loss = (
            mse_loss
            + lambda_smooth * smoothness_loss
            + lambda_power * power_loss_term  # Now can use power loss from all layers or just final
            - FOCAL_INTENSITY_WEIGHT * focal_intensity  # Adjust weight as needed
        )
        
        # Debug: Print loss components occasionally
        tf.print("MSE:", mse_loss, "Smooth:", lambda_smooth * smoothness_loss, "Power:", lambda_power * power_loss_term, "Focal:", FOCAL_INTENSITY_WEIGHT * focal_intensity)
        
        return total_loss

    return custom_loss


model.compile(optimizer=opt, loss=custom_loss_with_model(model), metrics=[psnr_metric])


print("Tworzenie datasetów...")
start_time = time.time()
# Reduce batch size for better accuracy
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(500).batch(BATCH_SIZE).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))
end_time = time.time()
print(f"Data loading time: {end_time - start_time:.2f} seconds")
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))

from pathlib import Path

# Create temporary directory structure for callbacks - will be updated after evaluation
temp_phase_histograms_dir = Path("temp_phase_histograms")
temp_phase_histograms_dir.mkdir(exist_ok=True)

class PhaseHistogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        num_layers = len(self.model.doe_layers)
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 3))
        if num_layers == 1:
            axes = [axes]
        for i, layer in enumerate(self.model.doe_layers):
            phase_vals = layer.phase.numpy().flatten()
            axes[i].hist(phase_vals, bins=100)
            axes[i].set_title(f'Histogram fazy DOE {i+1} – epoka {epoch}')
            axes[i].set_xlabel('Faza [rad]')
            axes[i].set_ylabel('Liczność')
            axes[i].grid(True)
        plt.tight_layout()
        out_path = self.save_dir / f'phase_hist_epoch_{epoch:04d}.png'
        plt.savefig(out_path)
        plt.close(fig)
        # print(f"Saved phase histogram(s) for epoch {epoch} to {out_path}")

print("Trenowanie modelu...")
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=CALLBACK_MIN_DELTA,
    patience= CALLBACK_PATIENCE,
    restore_best_weights=True,
)

start_time = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[callback, PhaseHistogramCallback(temp_phase_histograms_dir)],
    verbose=1
)
model.summary()

keras.utils.plot_model(model, show_shapes=True, to_file='model_plot.png')
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")

print("Ocena modelu na zbiorze testowym:")
evaluation_results = model.evaluate(test_dataset)
print(evaluation_results)

# Update file naming to include model parameters
psnr_value = evaluation_results[1]
power_loss_mode = "all_layers" if USE_ALL_LAYERS_POWER_LOSS else "final_only"
file_suffix = f"PSNR_{psnr_value:.2f}_b_{BATCH_SIZE}_l_{NUM_LAYERS}_ep_{EPOCHS}_lr_{LEARNING_RATE:.3f}_dist_doe_{PROPAGATION_DISTANCE_BEETWEEN_DOE:.3f}_dist_target_{PROPAGATION_DISTANCE_TO_TARGET:.3f}_shape_{DOE_SHAPE[0]}x{DOE_SHAPE[1]}"

def periodic_phase_optimization(phase):
    """
    Post-process a phase mask to reduce sharp discontinuities by adjusting each pixel by -2π, 0, or +2π
    to minimize the sum of squared phase differences with its 4-connected neighbors.
    Args:
        phase: tf.Tensor of shape [H, W], dtype float32, values in [0, 2π)
    Returns:
        optimized_phase: tf.Tensor of shape [H, W], dtype float32
    """
    pi2 = tf.constant(2 * np.pi, dtype=phase.dtype)
    H = tf.shape(phase)[0]
    W = tf.shape(phase)[1]

    # Create 3 candidate phase masks: phi-2pi, phi, phi+2pi
    candidates = tf.stack([
        phase - pi2,  # k = -1
        phase,       # k = 0
        phase + pi2  # k = +1
    ], axis=-1)  # shape: [H, W, 3]

    # Pad for neighbor computation (REFLECT to avoid border artifacts)
    pad = [[1, 1], [1, 1], [0, 0]]
    candidates_padded = tf.pad(candidates, pad, mode='REFLECT')  # shape: [H+2, W+2, 3]

    # For each direction, get neighbor values for all candidates
    up    = candidates_padded[0:-2, 1:-1, :]  # [H, W, 3]
    down  = candidates_padded[2:  , 1:-1, :]
    left  = candidates_padded[1:-1, 0:-2, :]
    right = candidates_padded[1:-1, 2:  , :]

    # For each candidate, compute cost as sum of squared differences to 4 neighbors
    cost = (
        tf.square(candidates - up) +
        tf.square(candidates - down) +
        tf.square(candidates - left) +
        tf.square(candidates - right)
    )  # shape: [H, W, 3]

    # Find the k (index) with minimum cost for each pixel
    best_k = tf.argmin(cost, axis=-1, output_type=tf.int32)  # shape: [H, W], values in {0,1,2}

    # Gather the optimal phase for each pixel
    # Prepare indices for tf.gather_nd
    H_idx = tf.range(H, dtype=tf.int32)
    W_idx = tf.range(W, dtype=tf.int32)
    H_grid, W_grid = tf.meshgrid(H_idx, W_idx, indexing='ij')
    gather_idx = tf.stack([H_grid, W_grid, best_k], axis=-1)  # shape: [H, W, 3]
    optimized_phase = tf.gather_nd(candidates, gather_idx)  # shape: [H, W]

    # Optionally wrap back to [0, 2pi)
    optimized_phase = tf.math.floormod(optimized_phase, pi2)
    return optimized_phase

# Create organized output directory using file_suffix
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

organized_output_dir = results_dir / f"results_{file_suffix}"
organized_output_dir.mkdir(exist_ok=True)

# Create subdirectories
phase_comparison_dir = organized_output_dir / "phase_comparison"
sample_outputs_dir = organized_output_dir / "sample_outputs"
inputs_outputs_dir = organized_output_dir / "inputs_outputs"
doe_masks_dir = organized_output_dir / "b_doe_masks"
history_dir = organized_output_dir / "saved_histories"
models_dir = organized_output_dir / "models"
phase_histograms_dir = organized_output_dir / "phase_histograms"

phase_comparison_dir.mkdir(exist_ok=True)
sample_outputs_dir.mkdir(exist_ok=True)
inputs_outputs_dir.mkdir(exist_ok=True)
doe_masks_dir.mkdir(exist_ok=True)
history_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)
phase_histograms_dir.mkdir(exist_ok=True)

# Move phase histogram files from temp directory to organized directory
import shutil
if temp_phase_histograms_dir.exists():
    for file in temp_phase_histograms_dir.glob("*.png"):
        shutil.move(str(file), str(phase_histograms_dir / file.name))
    temp_phase_histograms_dir.rmdir()

# Calculate and save power loss information
power_loss_info = []
power_loss_info.append("=== MODEL PARAMETERS AND POWER LOSS ANALYSIS ===\n\n")

# Add training parameters
power_loss_info.append("Training Parameters:\n")
power_loss_info.append(f"  - Learning Rate: {LEARNING_RATE}\n")
power_loss_info.append(f"  - Batch Size: {BATCH_SIZE}\n")
power_loss_info.append(f"  - Epochs: {EPOCHS}\n")
power_loss_info.append(f"  - Callback Patience: {CALLBACK_PATIENCE}\n")
power_loss_info.append(f"  - Callback Min Delta: {CALLBACK_MIN_DELTA}\n")
power_loss_info.append(f"  - Smoothness Weight: {SMOOTHNESS_WEIGHT}\n")
power_loss_info.append(f"  - Smoothness Traditional Weight: {SMOOTHNESS_TRADITIONAL_WEIGHT}\n")
power_loss_info.append(f"  - Smoothness Variation Weight: {SMOOTHNESS_VARIATION_WEIGHT}\n")
power_loss_info.append(f"  - Smoothness Binary Weight: {SMOOTHNESS_BINARY_WEIGHT}\n")
power_loss_info.append(f"  - Smoothness Target Std Percent: {SMOOTHNESS_TARGET_STD_PERCENT}\n")
power_loss_info.append(f"  - Power Loss Weight: {POWER_LOSS_WEIGHT}\n")
power_loss_info.append(f"  - Focal Intensity Weight: {FOCAL_INTENSITY_WEIGHT}\n")
power_loss_info.append(f"  - Use All Layers Power Loss: {USE_ALL_LAYERS_POWER_LOSS}\n\n")

power_loss_info.append("Model Configuration:\n")
power_loss_info.append(f"  - Number of DOE layers: {NUM_LAYERS}\n")
power_loss_info.append(f"  - DOE shape: {DOE_SHAPE}\n")
power_loss_info.append(f"  - Wavelength_min: {WAVELENGTH_MIN:.2e} m\n")
power_loss_info.append(f"  - Wavelength_max: {WAVELENGTH_MAX:.2e} m\n")
power_loss_info.append(f"  - Wavelength_step: {WAVELENGTH_STEP:.2e} m\n")
power_loss_info.append(f"  - Distance between DOEs: {PROPAGATION_DISTANCE_BEETWEEN_DOE} m\n")
power_loss_info.append(f"  - Distance to target: {PROPAGATION_DISTANCE_TO_TARGET} m\n")
power_loss_info.append(f"  - Pixel size: {PIXEL_SIZE:.2e} m\n\n")

# Calculate power losses using test data
test_sample = x_test[:1]  # Use first test sample
test_output = model(test_sample)

power_loss_info.append("Power Loss per Layer:\n")
if hasattr(model, 'all_power_losses') and len(model.all_power_losses) > 0:
    remaining_power = 1.0  # Start with 100% power
    cumulative_power_losses = []
    
    for i, power_loss in enumerate(model.all_power_losses):
        layer_power_loss = float(tf.reduce_mean(power_loss).numpy())
        cumulative_power_losses.append(layer_power_loss)
        power_before_layer = remaining_power
        remaining_power = remaining_power * (1.0 - layer_power_loss)  # Apply sequential loss
        power_loss_info.append(f"  Layer {i+1}: {layer_power_loss:.6f} ({layer_power_loss*100:.4f}% of incident power)\n")
        power_loss_info.append(f"    Power before layer {i+1}: {power_before_layer:.6f} ({power_before_layer*100:.4f}%)\n")
        power_loss_info.append(f"    Power after layer {i+1}: {remaining_power:.6f} ({remaining_power*100:.4f}%)\n\n")
    
    total_power_loss = 1.0 - remaining_power  # Total cumulative power loss
    power_loss_info.append(f"Total Cumulative Power Loss: {total_power_loss:.6f} ({total_power_loss*100:.4f}%)\n")
    power_loss_info.append(f"Final Power Transmission: {remaining_power:.6f} ({remaining_power*100:.4f}%)\n")
    power_loss_info.append(f"Power Efficiency: {remaining_power*100:.4f}%\n")
else:
    power_loss_info.append("  No power loss information available\n")

# Add final layer power loss if available
if hasattr(model, 'last_power_loss'):
    final_power_loss = float(tf.reduce_mean(model.last_power_loss).numpy())
    power_loss_info.append(f"\nFinal Layer Power Loss: {final_power_loss:.6f} ({final_power_loss*100:.4f}%)\n")

# Save power loss information to file
parameters_file = organized_output_dir / "parameters.txt"
with open(parameters_file, 'w') as f:
    f.writelines(power_loss_info)
print(f"Model parameters and power loss analysis saved to {parameters_file}")

# Save the best trained phase mask to a folder as BMP
for i, layer in enumerate(model.doe_layers):
    phase = layer.phase.numpy()
    
    # Ensure phase is properly wrapped to [0, 2π] range before processing
    phase_wrapped = np.mod(phase, 2*np.pi)
    print(f"DOE Layer {i+1} - Phase range before wrapping: [{np.min(phase):.3f}, {np.max(phase):.3f}] radians")
    print(f"DOE Layer {i+1} - Phase range after wrapping: [{np.min(phase_wrapped):.3f}, {np.max(phase_wrapped):.3f}] radians")
    
    # Convert wrapped phase to tf.Tensor for optimization
    phase_tensor = tf.convert_to_tensor(phase_wrapped, dtype=tf.float32)
    # Optimize phase mask to reduce discontinuities
    optimized_phase = periodic_phase_optimization(phase_tensor).numpy()
    
    # Verify optimized phase range
    print(f"DOE Layer {i+1} - Optimized phase range: [{np.min(optimized_phase):.3f}, {np.max(optimized_phase):.3f}] radians")

    # Normalize optimized phase to range 0-255
    phase_normalized = (optimized_phase/(2*np.pi)*255).astype(np.uint8)
    print(f"DOE Layer {i+1} - Final BMP values range: [{np.min(phase_normalized)}, {np.max(phase_normalized)}]")

    # Save as BMP file
    output_file_bmp = doe_masks_dir / f'b_doe_{i + 1}_{file_suffix}.bmp'
    Image.fromarray(phase_normalized).save(output_file_bmp)
    print(f"Saved best trained (optimized) phase mask for DOE Layer {i + 1} as BMP to {output_file_bmp}")

# Save a side-by-side PNG comparison of the phase mask before and after periodic phase optimization for each DOE layer
for i, layer in enumerate(model.doe_layers):
    phase = layer.phase.numpy()
    
    # Ensure phase is properly wrapped to [0, 2π] range before processing
    phase_wrapped = np.mod(phase, 2*np.pi)
    
    # Unoptimized phase mask
    phase_unoptimized_normalized = (phase_wrapped/(2*np.pi)*255).astype(np.uint8)
    # Optimize phase mask
    phase_tensor = tf.convert_to_tensor(phase_wrapped, dtype=tf.float32)
    optimized_phase = periodic_phase_optimization(phase_tensor).numpy()
    phase_optimized_normalized = (optimized_phase/(2*np.pi)*255).astype(np.uint8)

    # Create a side-by-side comparison image with matplotlib and titles
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(phase_unoptimized_normalized, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Before Optimization')
    axes[0].axis('off')
    axes[1].imshow(phase_optimized_normalized, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('After Optimization')
    axes[1].axis('off')
    plt.tight_layout()
    output_file_png = phase_comparison_dir / f'phase_comparison_{i + 1}_{file_suffix}.png'
    plt.savefig(output_file_png)
    plt.close(fig)
    print(f"Saved phase mask comparison (before/after optimization) for DOE Layer {i + 1} as PNG to {output_file_png}")

# print("LICZBA WARSTW:", len(model.doe_layers))
# Ensure the `saved_histories` directory exists
history_dir = Path("saved_histories")
history_dir.mkdir(exist_ok=True)

# Save training history to a file with the current date
history_file = history_dir / f"history_{file_suffix}_{time.strftime('%Y-%m-%d')}.npy"
np.save(history_file, history.history)
print(f"Training history saved to {history_file}")

# Save training history as a graph with loss and PSNR over epochs
plt.figure(figsize=(12, 6))

# Plot loss with a logarithmic y-axis
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.title('Loss over Epochs (Log Scale)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot PSNR metric
plt.subplot(1, 2, 2)
plt.plot(history.history['psnr_metric'], label='Training PSNR')
if 'val_psnr_metric' in history.history:
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
plt.title('PSNR over Epochs')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()

# Save the graph to a file
history_graph_file = history_dir / f"history_graph_{file_suffix}_{time.strftime('%Y-%m-%d')}.png"
plt.tight_layout()
plt.savefig(history_graph_file)
plt.close()
print(f"Training history graph saved to {history_graph_file}")

# Also save a detailed training parameters loss graph to the organized results directory
plt.figure(figsize=(15, 10))

# Create a 2x2 subplot layout for detailed analysis
# Plot 1: Training and Validation Loss (Linear Scale)
plt.subplot(2, 2, 1)
epochs_range = range(1, len(history.history['loss']) + 1)
plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
plt.title('Training Parameters: Loss per Epoch (Linear Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss (Log Scale)
plt.subplot(2, 2, 2)
plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
if 'val_loss' in history.history:
    plt.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
plt.yscale('log')
plt.title('Training Parameters: Loss per Epoch (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: PSNR Metric
plt.subplot(2, 2, 3)
plt.plot(epochs_range, history.history['psnr_metric'], 'g-', label='Training PSNR', linewidth=2)
if 'val_psnr_metric' in history.history:
    plt.plot(epochs_range, history.history['val_psnr_metric'], 'orange', label='Validation PSNR', linewidth=2)
plt.title('Training Parameters: PSNR per Epoch')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Learning Rate (if available) or Loss Improvement
plt.subplot(2, 2, 4)
if len(history.history['loss']) > 1:
    loss_improvement = [0] + [history.history['loss'][i-1] - history.history['loss'][i] 
                              for i in range(1, len(history.history['loss']))]
    plt.plot(epochs_range, loss_improvement, 'm-', label='Loss Improvement', linewidth=2)
    plt.title('Training Parameters: Loss Improvement per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Add training parameters as text
plt.figtext(0.02, 0.02, 
           f'Parameters: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Layers={NUM_LAYERS}, '
           f'Smoothness={SMOOTHNESS_WEIGHT}, Power={POWER_LOSS_WEIGHT}', 
           fontsize=10, ha='left')

# Save the detailed training parameters graph to results directory
training_params_graph = organized_output_dir / f"training_parameters_loss_{file_suffix}.png"
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Make room for parameter text
plt.savefig(training_params_graph, dpi=300, bbox_inches='tight')
plt.close()
print(f"Detailed training parameters loss graph saved to {training_params_graph}")

# ================================
# WIZUALIZACJA WYNIKÓW + FAZY DOE
# ================================
print("Wizualizacja wyników i eksport masek fazowych...")
sample_inputs = x_test[:5]
# print("Sample inputs shape:", sample_inputs.shape)
output_amplitude = model(sample_inputs).numpy()
# output_amplitude = (output_amplitude - output_amplitude.min()) / (output_amplitude.max() - output_amplitude.min());
# print("Output amplitude shape:", output_amplitude.shape)
# print("Sample inputs shape:", sample_inputs.shape)
# print("Sample inputs range:", sample_inputs.min(), sample_inputs.max())
# print("Training history:", history.history)
print("Number of propagation layers:", len(model.prop_layers))

# for i, layer in enumerate(model.doe_layers):
#     print(f"DOE Layer {i+1} phase shape:", layer.phase.numpy().shape)

fig, axes = plt.subplots(4, 5, figsize=(18, 12))

# Update the loop to iterate over the range of len(model.doe_layers)
for i in range(len(model.doe_layers)):
    im0 = axes[0, i].imshow(sample_inputs[i, :, :, 0], cmap='gray')
    axes[0, i].set_title(f'Input {i}')
    axes[0, i].axis('off')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

    im1 = axes[1, i].imshow(output_amplitude[i], cmap='hot')
    axes[1, i].set_title(f'Output {i}')
    axes[1, i].axis('off')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

    phase = model.doe_layers[i].phase.numpy()
    phase_tensor = tf.convert_to_tensor(phase, dtype=tf.float32)
    optimized_phase = periodic_phase_optimization(phase_tensor).numpy()
    phase = (optimized_phase/(2*np.pi)*255).astype(np.uint8)
    # print("phsae min:", phase.min())
    # print("phase max:", phase.max())
    im2 = axes[2, i].imshow(phase, cmap='gray', vmin=0, vmax=255)
    axes[2, i].set_title(f'DOE Phase {i + 1}')
    axes[2, i].axis('off')
    plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    im3 = axes[3, i].imshow(y_test[i], cmap='hot')
    axes[3, i].set_title(f'Target {i}')
    axes[3, i].axis('off')
    plt.colorbar(im3, ax=axes[3, i], fraction=0.046, pad=0.04)

# Update sample output file naming to include model parameters
sample_output_file = sample_outputs_dir / f'output_{file_suffix}.png'
plt.tight_layout()
plt.savefig(sample_output_file)
plt.close()
print(f"Sample output saved to {sample_output_file}")

# Plot 5 inputs and outputs even if there is only one DOE
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i in range(5):
    # Plot input
    im0 = axes[0, i].imshow(sample_inputs[i, :, :, 0], cmap='gray', extent=[0, DOE_SHAPE[1], 0, DOE_SHAPE[0]])
    axes[0, i].set_title(f'Input {i + 1}')
    axes[0, i].set_xlabel('X (pixels)')
    axes[0, i].set_ylabel('Y (pixels)')
    axes[0, i].axis('on')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

    # Plot output
    im1 = axes[1, i].imshow(output_amplitude[i], cmap='hot', extent=[0, DOE_SHAPE[1], 0, DOE_SHAPE[0]])
    axes[1, i].set_title(f'Output {i + 1}')
    axes[1, i].set_xlabel('X (pixels)')
    axes[1, i].set_ylabel('Y (pixels)')
    axes[1, i].axis('on')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(inputs_outputs_dir / f"inputs_outputs_plot_{file_suffix}.png")
plt.close()
print("Plotted 5 inputs and outputs.")

# ================================
# ZAPIS MODELU
# ================================
print("Zapisuję model...")
model.save(models_dir / f'cd2nn_model_{file_suffix}.keras')
print("Model zapisany jako cdnn_model_v2.keras")

# Update sample output file naming to include model parameters
output_dir = Path("sample_outputs")
output_dir.mkdir(exist_ok=True)
sample_output_file = output_dir / f'output_{file_suffix}.png'
plt.tight_layout()
plt.savefig(sample_output_file)
plt.close()
print(f"Sample output saved to {sample_output_file}")

# Plot 5 inputs and outputs even if there is only one DOE
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i in range(5):
    # Plot input
    im0 = axes[0, i].imshow(sample_inputs[i, :, :, 0], cmap='gray', extent=[0, DOE_SHAPE[1], 0, DOE_SHAPE[0]])
    axes[0, i].set_title(f'Input {i + 1}')
    axes[0, i].set_xlabel('X (pixels)')
    axes[0, i].set_ylabel('Y (pixels)')
    axes[0, i].axis('on')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

    # Plot output
    im1 = axes[1, i].imshow(output_amplitude[i], cmap='hot', extent=[0, DOE_SHAPE[1], 0, DOE_SHAPE[0]])
    axes[1, i].set_title(f'Output {i + 1}')
    axes[1, i].set_xlabel('X (pixels)')
    axes[1, i].set_ylabel('Y (pixels)')
    axes[1, i].axis('on')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"sample_outputs/inputs_outputs_plot_{file_suffix}.png")
plt.close()
print("Plotted 5 inputs and outputs.")

# ================================
# ZAPIS MODELU
# ================================
print("Zapisuję model...")
model.save(f'models/cd2nn_model_{file_suffix}.keras')
print("Model zapisany jako cdnn_model_v2.keras")






