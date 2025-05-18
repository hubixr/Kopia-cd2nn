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

# mixed_precision.set_global_policy('float32')


# ================================
# PARAMETRY UKLADU
# ================================
DOE_SHAPE = (128, 128)  # [px]
PIXEL_SIZE = 9e-4  # [m]
FREQUENCY = 96 * 1e9  # [GHz]
C = 299792458  # [m/s]
WAVELENGTH = C / (FREQUENCY)  # [m]
print("Wavelength:", WAVELENGTH)
PROPAGATION_DISTANCE_BEETWEEN_DOE = 0.1  # [m]
PROPAGATION_DISTANCE_TO_TARGET = 0.2  # [m]
NUM_LAYERS = 1
EPOCHS = 50
LEARNING_RATE = 0.003
BATCH_SIZE = 1
CALLBACK_PATIENCE = 1
CALLBACK_MIN_DELTA = 1e-4 #deflaut 1e-5
# ================================
DATA_DIR = Path("./cdnn_data")
INPUT_DIR = DATA_DIR / "input_fields"
TARGET_FILE = DATA_DIR / "target_field.bmp"

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Set a manual memory limit (in MB) for each GPU
        memory_limit_mb = 8192  # Example: 4GB limit
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
    print("target_array shape:", target_array.shape)
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
input_files = sorted(INPUT_DIR.glob("*.bmp"))
inputs = []
for f in input_files:
    image = Image.open(f).convert('L')  # Convert to grayscale
    image = image.resize(DOE_SHAPE, Image.Resampling.LANCZOS)  # Resize to match DOE_SHAPE
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to 0-1
    inputs.append(image_array)
if not inputs:
    raise ValueError("No valid input fields found in the directory.")
input_data = np.stack(inputs, axis=0).astype(np.float32)  # Load .bmp files directly

# Debugging: Print the shape of input_data
print(f"Input data shape: {input_data.shape}")

# Update the input data processing to include the zero channel
input_data = add_zero_channel(input_data)

# Debugging: Print the shape of input_data before reshaping
print(f"Input data shape before reshaping: {input_data.shape}")

# Apply cropping or resizing to input_data
input_data = crop_or_resize_input(input_data, DOE_SHAPE)

# Debugging: Print the shape of input_data after cropping or resizing
print(f"Input data shape after cropping or resizing: {input_data.shape}")


# Debugging: Print the shape of input_data after adding the second channel
print(f"Input data shape after adding second channel: {input_data.shape}")



# Debugging: Print input data shape
print(f"Input data shape after reshaping: {input_data.shape}")
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

print("y_test min:", y_test.min())
print("y_test max:", y_test.max())
print("y_train mean:", y_train.mean())

# Debugging: Check input data for NaN or large values
print("Input data stats:")
print(f"x_train min: {np.min(x_train)}, max: {np.max(x_train)}, mean: {np.mean(x_train)}")
print(f"x_val min: {np.min(x_val)}, max: {np.max(x_val)}, mean: {np.mean(x_val)}")
print(f"x_test min: {np.min(x_test)}, max: {np.max(x_test)}, mean: {np.mean(x_test)}")

# Debugging: Check target data for NaN or large values
print("Target data stats:")
print(f"y_train min: {np.min(y_train)}, max: {np.max(y_train)}, mean: {np.mean(y_train)}")
print(f"y_val min: {np.min(y_val)}, max: {np.max(y_val)}, mean: {np.mean(y_val)}")
print(f"y_test min: {np.min(y_test)}, max: {np.max(y_test)}, mean: {np.mean(y_test)}")

# ================================
# BUDOWA MODELU
# ================================
print("Budowanie modelu CDNN...")
model = CDNNModel(
    num_layers=NUM_LAYERS,
    shape=DOE_SHAPE,
    wavelength=WAVELENGTH,
    distance_to_plane=PROPAGATION_DISTANCE_TO_TARGET,
    distance_between_layers=PROPAGATION_DISTANCE_BEETWEEN_DOE,
    pixel_size=PIXEL_SIZE
)

loss_fn = tf.keras.losses.MeanSquaredError()  # Absolute Mean Error
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0) #clipnorm for gradient clipping - better stability


def psnr_metric(y_true, y_pred):
    # Ensure y_pred has the same shape as y_true by adding a channel dimension if missing
    if len(y_pred.shape) == 3:  # Check if y_pred is missing the channel dimension
        y_pred = tf.expand_dims(y_pred, axis=-1)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def calculate_power(y):
    return tf.reduce_sum(tf.square(y), axis=[1, 2])

lambda_smooth = 3e-6  # Weight for smoothness regularization
def smoothness_regularization(phase):
    """
    Compute the smoothness regularization term for the phase mask using L2 difference with 8 neighbors.
    Args:
        phase: Tensor of shape (H, W), the phase mask.
    Returns:
        Smoothness regularization term (scalar).
    """
    # Pad the phase mask to handle borders
    phase_padded = tf.pad(phase, [[1, 1], [1, 1]], mode='REFLECT')
    neighbors = [
        phase_padded[0:-2, 0:-2],  # top-left
        phase_padded[0:-2, 1:-1],  # top
        phase_padded[0:-2, 2:  ],  # top-right
        phase_padded[1:-1, 0:-2],  # left
        phase_padded[1:-1, 2:  ],  # right
        phase_padded[2:  , 0:-2],  # bottom-left
        phase_padded[2:  , 1:-1],  # bottom
        phase_padded[2:  , 2:  ]   # bottom-right
    ]
    center = phase
    smoothness_term = 0
    for n in neighbors:
        smoothness_term += tf.reduce_sum(tf.square(center - n))
    return smoothness_term
def custom_loss_with_model(model):
    def custom_loss(y_true, y_pred):
        
        # Custom loss function with smoothness regularization.
        # Args:
        #     y_true: Ground truth target.
        #     y_pred: Predicted output.
        # Returns:
        #     Total loss (scalar).
        
        # Compute the standard loss (e.g., Mean Squared Error)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Add smoothness regularization for each phase mask
        smoothness_loss = 0
        for layer in model.doe_layers:
            smoothness_loss += smoothness_regularization(layer.phase)

        # Combine the losses
        total_loss = mse_loss + lambda_smooth * smoothness_loss
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
print("Data parameters:")
print("x_train range:", x_train.min(), x_train.max())
print("y_train range:", y_train.min(), y_train.max())
print("x_test range:", x_test.min(), x_test.max())
print("y_test range:", y_test.min(), y_test.max())

print("Trenowanie modelu...")
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor the loss
    min_delta=CALLBACK_MIN_DELTA,  # Minimum change in loss to qualify as an improvement
    patience= CALLBACK_PATIENCE,      # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restore the weights of the best epoch
)
start_time = time.time()
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[callback], verbose=1)
model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file='model_plot.png')
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")

print("Ocena modelu na zbiorze testowym:")
evaluation_results = model.evaluate(test_dataset)
print(evaluation_results)

# Update file naming to include model parameters
psnr_value = evaluation_results[1]  # Assuming PSNR is the second metric in evaluation_results
file_suffix = f"PSNR_{psnr_value:.2f}_freq_{FREQUENCY/1e9:.3f}GHz_batch_{BATCH_SIZE}_layers_{NUM_LAYERS}_epochs_{EPOCHS}_lr_{LEARNING_RATE:.3f}_dist_doe_{PROPAGATION_DISTANCE_BEETWEEN_DOE:.3f}_dist_target_{PROPAGATION_DISTANCE_TO_TARGET:.3f}_doe_shape_{DOE_SHAPE[0]}x{DOE_SHAPE[1]}"

# Save the best trained phase mask to a folder as BMP
output_dir = Path("best_doe_masks")
output_dir.mkdir(exist_ok=True)
for i, layer in enumerate(model.doe_layers):
    phase = layer.phase.numpy()

    # Normalize phase to range 0-255
    phase_normalized = (phase/(2*np.pi)*255).astype(np.uint8)

    # Save as BMP file
    output_file_bmp = output_dir / f'best_trained_doe_phase_{i + 1}_{file_suffix}.bmp'
    Image.fromarray(phase_normalized).save(output_file_bmp)
    print(f"Saved best trained phase mask for DOE Layer {i + 1} as BMP to {output_file_bmp}")

print("LICZBA WARSTW:", len(model.doe_layers))
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
plt.yscale('log')  # Set y-axis to logarithmic scale
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
history_graph_file = f"saved_histories/history_graph_{file_suffix}_{time.strftime('%Y-%m-%d')}.png"
plt.tight_layout()
plt.savefig(history_graph_file)
plt.close()  # Close the plot to avoid displaying it
print(f"Training history graph saved to {history_graph_file}")

# ================================
# WIZUALIZACJA WYNIKÓW + FAZY DOE
# ================================
print("Wizualizacja wyników i eksport masek fazowych...")
sample_inputs = x_test[:5]
print("Sample inputs shape:", sample_inputs.shape)
output_amplitude = model(sample_inputs).numpy()
# output_amplitude = (output_amplitude - output_amplitude.min()) / (output_amplitude.max() - output_amplitude.min())
print("Output amplitude shape:", output_amplitude.shape)
print("Sample inputs shape:", sample_inputs.shape)
print("Sample inputs range:", sample_inputs.min(), sample_inputs.max())
# print("Training history:", history.history)
print("Number of propagation layers:", len(model.prop_layers))

for i, layer in enumerate(model.doe_layers):
    print(f"DOE Layer {i+1} phase shape:", layer.phase.numpy().shape)

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
    phase = (phase/(2*np.pi)*255).astype(np.uint8)
    print("phsae min:", phase.min())
    print("phase max:", phase.max())
    im2 = axes[2, i].imshow(phase, cmap='gray', vmin=0, vmax=255)
    axes[2, i].set_title(f'DOE Phase {i + 1}')
    axes[2, i].axis('off')
    plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    im3 = axes[3, i].imshow(y_test[i], cmap='hot')
    axes[3, i].set_title(f'Target {i}')
    axes[3, i].axis('off')
    plt.colorbar(im3, ax=axes[3, i], fraction=0.046, pad=0.04)

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

# Validate input data for NaN or Inf values before training
if np.any(np.isnan(x_train)) or np.any(np.isinf(x_train)):
    raise ValueError("NaN or Inf detected in x_train")
if np.any(np.isnan(x_val)) or np.any(np.isinf(x_val)):
    raise ValueError("NaN or Inf detected in x_val")
if np.any(np.isnan(x_test)) or np.any(np.isinf(x_test)):
    raise ValueError("NaN or Inf detected in x_test")

print("Input data validation passed: No NaN or Inf values detected.")



