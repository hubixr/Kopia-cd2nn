import os
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image

# --- User parameters ---
TRAIN_SCRIPT = "cd2nn_training/train_cd2nn_model.py"
PROP_SCRIPT = "cd2nn_propagation_simulation/run_propagation.py"
MASK_SAVE_DIR = "masks"
PROP_OUTPUT_DIR = "propagation_outputs"
RESULTS_CSV = "results.csv"

param_name = "learning_rate"

# Sweep ranges for EPOCHS and SMOOTHNESS_WEIGHT
EPOCHS_RANGE = [10,25]  # Example: [1, 3, 5] or use range(start, stop, step)
SMOOTHNESS_WEIGHT_RANGE = [3e-9]
LR_VALUES = [0.03]  # Example: [0.01, 0.03, 0.1]
PROPAGATION_DISTANCE_BEETWEEN_DOE = 0.1  # [m]
PROPAGATION_DISTANCE_TO_TARGET = 0.2  # [m]
NUM_LAYERS = 1
BATCH_SIZE = 1
CALLBACK_PATIENCE = 1
CALLBACK_MIN_DELTA = 1e-5 #deflaut 1e-5

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=None)
# ... add other arguments as needed ...
args, unknown = parser.parse_known_args()

if args.input_dir is not None:
    INPUT_DIR = Path(args.input_dir)
    DATA_DIR = INPUT_DIR.parent
else:
    DATA_DIR = Path("./cdnn_data")
    INPUT_DIR = DATA_DIR / "input_fields"

# --- Helper: Optimize phase mask ---
def periodic_phase_optimization(phase):
    pi2 = 2 * np.pi
    phase = tf.convert_to_tensor(phase, dtype=tf.float32)
    H, W = phase.shape
    shifts = tf.constant([-pi2, 0.0, pi2], dtype=tf.float32)
    phase_shifted = phase[None, :, :] + shifts[:, None, None]  # [3, H, W]
    # Pad for neighbors
    pad = [[0,0],[1,1],[1,1]]
    phase_padded = tf.pad(phase_shifted, pad, mode='REFLECT')
    neighbors = [
        phase_padded[:, :-2, 1:-1],  # up
        phase_padded[:, 2:, 1:-1],   # down
        phase_padded[:, 1:-1, :-2],  # left
        phase_padded[:, 1:-1, 2:],   # right
    ]
    diffs = [tf.square(phase_shifted - n) for n in neighbors]
    cost = tf.add_n(diffs)  # [3, H, W]
    best_k = tf.argmin(cost, axis=0)  # [H, W]
    best_phase = tf.gather(phase_shifted, best_k, axis=0, batch_dims=0)
    return tf.math.floormod(best_phase, pi2)

# --- Main sweep ---
os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(PROP_OUTPUT_DIR, exist_ok=True)

with open(RESULTS_CSV, "a") as results_file:
    for epochs in EPOCHS_RANGE:
        for smoothness_weight in SMOOTHNESS_WEIGHT_RANGE:
            for val in LR_VALUES:
                # 1. Run training
                mask_unopt_path = os.path.join(MASK_SAVE_DIR, f"mask_unopt_{val:.4f}_ep{epochs}_sm{smoothness_weight:.0e}.bmp")
                mask_opt_path = os.path.join(MASK_SAVE_DIR, f"mask_opt_{val:.4f}_ep{epochs}_sm{smoothness_weight:.0e}.bmp")
                subprocess.run([
                    "python", os.path.basename(TRAIN_SCRIPT),
                    f"--{param_name}", str(val),
                    "--save_mask_unopt", os.path.abspath(mask_unopt_path),
                    "--input_dir", str(INPUT_DIR.resolve()),
                    "--epochs", str(epochs),
                    "--propagation_distance_between_doe", str(PROPAGATION_DISTANCE_BEETWEEN_DOE),
                    "--propagation_distance_to_target", str(PROPAGATION_DISTANCE_TO_TARGET),
                    "--num_layers", str(NUM_LAYERS),
                    "--batch_size", str(BATCH_SIZE),
                    "--callback_patience", str(CALLBACK_PATIENCE),
                    "--callback_min_delta", str(CALLBACK_MIN_DELTA),
                    "--smoothness_weight", str(smoothness_weight)
                ], check=True, cwd=os.path.dirname(TRAIN_SCRIPT))

                # 2. Optimize mask
                phase_unopt = np.array(Image.open(mask_unopt_path))
                phase_opt = periodic_phase_optimization(phase_unopt).numpy()
                # np.save(mask_opt_path, phase_opt)  # Removed to prevent saving large .npy files in masks folder

                for mask_path, mask_label in [(mask_unopt_path, "unopt"), (mask_opt_path, "opt")]:
                    # 4. Run propagation simulation
                    output_path = os.path.join(PROP_OUTPUT_DIR, f"output_{mask_label}_{val:.4f}_ep{epochs}_sm{smoothness_weight:.0e}.npy")
                    try:
                        result = subprocess.run([
                            "python", os.path.basename(PROP_SCRIPT),
                            "--mask_path", os.path.abspath(mask_path),
                            "--output_path", os.path.abspath(output_path)
                        ], check=True, cwd=os.path.dirname(PROP_SCRIPT), capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"Propagation script failed for {mask_path}. STDERR:\n{result.stderr}")
                            continue
                    except subprocess.CalledProcessError as e:
                        print(f"Propagation script failed for {mask_path}. STDERR:\n{e.stderr}")
                        continue
                    if not os.path.exists(output_path):
                        print(f"Propagation output file not found: {output_path}")
                        continue
                    # 5. Load output and log max values
                    try:
                        output = np.load(output_path, allow_pickle=True).item()
                        max_intensity = np.max(output["intensity"])
                        max_amplitude = np.max(output["amplitude"])
                        # Delete the .npy file after reading
                        try:
                            os.remove(output_path)
                        except Exception as e:
                            print(f"Warning: Could not delete {output_path}: {e}")
                        results_file.write(f"{val},{epochs},{smoothness_weight},{mask_label},{max_intensity},{max_amplitude}\n")
                        results_file.flush()
                    except Exception as e:
                        print(f"Failed to load or process output file {output_path}: {e}")
                        continue