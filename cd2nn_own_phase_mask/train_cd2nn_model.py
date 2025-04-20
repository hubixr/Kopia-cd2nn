import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from cd2nn_model import CDNNModel
import time
from tensorflow.keras import mixed_precision
from PIL import Image

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

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# ================================
# FUNKCJE POMOCNICZE
# ================================

"""def load_input_fields(input_dir, shape):
    files = sorted(input_dir.glob("*.npy"))
    # print(f"Found files: {files}")  # Debugging line
    inputs = []
    for f in files:
        field = np.load(f)
        if field.shape != shape + (2,):
            raise ValueError(f"Plik {f} ma kształt {field.shape}, oczekiwano {shape + (2,)}")
        field = np.round(field, 9)
        inputs.append(field)
    if not inputs:
        raise ValueError("No valid input fields found in the directory.")
    return np.stack(inputs, axis=0)

def load_target_field(target_file, shape):
    target = np.load(target_file)
    if target.shape != shape:
        raise ValueError(f"Target field ma kształt {target.shape}, oczekiwano {shape}")
    target = np.expand_dims(target, axis=0)
    target = np.round(target, 9)
    return target"""



# Function to load .bmp file and preprocess it for the model
def load_bmp_as_input(file_path, target_shape):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    image = image.resize(target_shape, Image.Resampling.LANCZOS)  # Resize to target shape using LANCZOS
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize to 0-1
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

# ================================
# GLOWNA CZESC
# ================================
"""print("Laduję dane wejściowe...")
# Revert to original input data loading logic
input_data = load_input_fields(INPUT_DIR, DOE_SHAPE).astype(np.float16)
print(f"Liczba próbek: {input_data.shape[0]}")
print("Laduję target...")
target_data = load_target_field(TARGET_FILE, DOE_SHAPE).astype(np.float16)
num_samples = input_data.shape[0]
targets = np.repeat(target_data, num_samples, axis=0)"""

# ================================
# PODZIAŁ NA ZBIORY
# ================================
"""indices = np.arange(num_samples)
np.random.shuffle(indices)

train_end = int(0.7 * num_samples)
val_end = int(0.85 * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# Use the centered data for training
x_train, y_train = input_data[train_idx], targets[train_idx]
x_val, y_val = input_data[val_idx], targets[val_idx]
x_test, y_test = input_data[test_idx], targets[test_idx]""
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
print(f"y_test min: {np.min(y_test)}, max: {np.max(y_test)}, mean: {np.mean(y_test)}")"""

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

loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.003, clipnorm=1.0)
# opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
# mixed_precision.set_global_policy('mixed_float16')

"""print("Tworzenie datasetów...")
start_time = time.time()
# Reduce batch size for better accuracy
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(500).batch(8).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))
end_time = time.time()
print(f"Data loading time: {end_time - start_time:.2f} seconds")
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(8).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8).map(lambda x, y: (tf.cast(x, tf.float16), tf.cast(y, tf.float16)))
print("Data parameters:")
print("x_train range:", x_train.min(), x_train.max())
print("y_train range:", y_train.min(), y_train.max())
print("x_test range:", x_test.min(), x_test.max())
print("y_test range:", y_test.min(), y_test.max())

print("Trenowanie modelu...")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
start_time = time.time()
# history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[callback], verbose=1)
end_time = time.time()
print(f"Model training time: {end_time - start_time:.2f} seconds")
"""

# Save the best trained phase mask to a folder
"""output_dir = Path("best_doe_masks")
output_dir.mkdir(exist_ok=True)
for i, layer in enumerate(model.doe_layers):
    phase = layer.phase.numpy()
    output_file = output_dir / f'best_trained_doe_phase_{i + 1}.npy'
    np.save(output_file, phase)
    print(f"Saved best trained phase mask for DOE Layer {i + 1} to {output_file}")"""

"""# Ensure the `saved_histories` directory exists
history_dir = Path("saved_histories")
history_dir.mkdir(exist_ok=True)

# Save training history to a file with the current date
history_file = history_dir / f"history_{time.strftime('%Y-%m-%d')}.npy"
np.save(history_file, history.history)
print(f"Training history saved to {history_file}")

# Save training history as a graph with accuracy and loss over epochs
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the graph
history_graph_file = f"saved_histories/history_graph_{time.strftime('%Y-%m-%d')}.png"
plt.tight_layout()
plt.savefig(history_graph_file)
plt.close()
print(f"Training history graph saved to {history_graph_file}")"""

# ================================
# WIZUALIZACJA WYNIKÓW + FAZY DOE
# ================================
"""print("Wizualizacja wyników i eksport masek fazowych...")
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
for i in range(NUM_LAYERS):
    im0 = axes[0, i].imshow(sample_inputs[i, :, :, 0], cmap='gray')
    axes[0, i].set_title(f'Input {i}')
    axes[0, i].axis('off')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

    im1 = axes[1, i].imshow(output_amplitude[i], cmap='hot')
    axes[1, i].set_title(f'Output {i}')
    axes[1, i].axis('off')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

    # Use phase directly from the model instead of loading from a file
    phase = model.doe_layers[i].phase.numpy()
    im2 = axes[2, i].imshow(phase, cmap='gray', vmin=0, vmax=2 * np.pi)
    axes[2, i].set_title(f'DOE Phase {i + 1}')
    axes[2, i].axis('off')
    plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    im3 = axes[3, i].imshow(y_test[i], cmap='hot')
    axes[3, i].set_title(f'Target {i}')
    axes[3, i].axis('off')
    plt.colorbar(im3, ax=axes[3, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('cdnn_sample_outputs_v2_with_phase_and_target.png')
plt.close()"""
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
# ================================
# ZAPIS MODELU
# ================================
"""print("Zapisuję model...")
model.save('cdnn_model_v2.keras')
print("Model zapisany jako cdnn_model_v2.keras")
print("Ocena modelu na zbiorze testowym:")
print(model.evaluate(test_dataset))
print("długość train dataset:", len(train_dataset))
print("długość val dataset:", len(val_dataset))
print("długość test dataset:", len(test_dataset))"""
