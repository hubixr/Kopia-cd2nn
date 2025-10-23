# CD2NN for THz - Frequency-Dependent Diffractive Deep Neural Network

This directory contains the implementation of a Cascaded Diffractive Deep Neural Network (CD2NN) designed for terahertz (THz) frequency applications with support for multiple wavelengths.

## Overview

The CD2NN model simulates optical wave propagation through diffractive optical elements (DOEs) using the Fresnel diffraction approximation. This implementation supports:

- **Multi-wavelength training**: Train on a range of THz frequencies (160-200 GHz by default)
- **Complex field propagation**: Full complex amplitude and phase propagation
- **Multiple DOE layers**: Configurable number of cascaded diffractive layers
- **GPU acceleration**: Optimized for NVIDIA GPUs with TensorFlow

## Project Structure

```
gpu/differ_freq/
├── train_cd2nn_model.py       # Main training script
├── cd2nn_model.py              # CDNN model architecture
├── DiffractiveMaskLayer.py     # Phase modulation layer
├── PropagationLayer.py         # Fresnel propagation layer
├── run_n_times.py              # Batch training utility
├── multiple_iteration.py       # Parameter sweep utility
├── cdnn_data/                  # Training data directory
│   ├── input_fields/           # Input field files (.npy format)
│   └── target_field.bmp        # Target output pattern
└── results/                    # Training results and outputs
```

## Installation

### Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended: 30GB+ VRAM)
- TensorFlow 2.13.0 or higher

### Setup

1. Install dependencies:
```bash
pip install -r ../../requirements.txt
```

2. Verify GPU availability:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Data Preparation

### Input Fields

Input fields should be stored as NumPy arrays (`.npy` files) in `cdnn_data/input_fields/` with shape `(128, 128, 3)`:
- **Channel 0**: Real part of the field
- **Channel 1**: Imaginary part of the field  
- **Channel 2**: Wavelength value (constant across spatial dimensions)

Generate input fields using:
generate_input_gauss_diff_freq.py

### Target Field

The target output pattern should be a grayscale BMP image (128x128 pixels) stored as `cdnn_data/target_field.bmp`.

## Training

### Basic Training

Run the training script with default parameters:
```bash
python train_cd2nn_model.py
```

### Key Training Parameters

Edit `train_cd2nn_model.py` to adjust:

#### Physical Parameters
```python
DOE_SHAPE = (128, 128)                          # DOE resolution [pixels]
PIXEL_SIZE = 9e-4                                # Pixel size [m] (0.9 mm)
PROPAGATION_DISTANCE_BEETWEEN_DOE = 0.1         # DOE spacing [m]
PROPAGATION_DISTANCE_TO_TARGET = 0.2            # Last DOE to target [m]
NUM_LAYERS = 2                                   # Number of DOE layers
```

#### Frequency Range
```python
FREQUENCY_MIN = 160 * 1e9    # 160 GHz
FREQUENCY_MAX = 200 * 1e9    # 200 GHz
FREQUENCY_STEP = 0.5 * 1e9   # 0.5 GHz steps
```

#### Training Hyperparameters
```python
EPOCHS = 150
BATCH_SIZE = 32

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[10, 15, 25, 40, 100],
    values=[0.8, 0.4, 0.2, 0.05, 0.03, 0.01]
)

CALLBACK_PATIENCE = 15        # Early stopping patience
CALLBACK_MIN_DELTA = 1e-5     # Minimum improvement threshold
```

#### Loss Function Weights
```python
SMOOTHNESS_WEIGHT = 0                  # Phase smoothness regularization
POWER_LOSS_WEIGHT = 1.2                # Power efficiency penalty
FOCAL_INTENSITY_WEIGHT = 0             # Central focusing bonus
USE_ALL_LAYERS_POWER_LOSS = True       # Cumulative power loss
```

### Output Organization

Results are automatically organized in timestamped directories:
```
results/results_PSNR_XX.XX_b_32_l_2.../
├── b_doe_masks/              # Trained DOE phase masks (.bmp)
├── phase_comparison/         # Before/after optimization
├── phase_histograms/         # Phase distribution per epoch
├── sample_outputs/           # Sample predictions
├── inputs_outputs/           # Visualizations
├── saved_histories/          # Training curves
├── models/                   # Saved Keras models
└── parameters.txt            # Full parameter log
```

## Model Architecture

### DiffractiveMaskLayer
Applies phase modulation to the complex field:
```
U_out = U_in * exp(i * φ)
```
- **Input**: `[B, H, W, 3]` - real, imaginary, wavelength
- **Output**: `[B, H, W, 3]` - modulated field
- **Trainable**: Phase mask φ (0 to 2π)

### PropagationLayer
Implements Fresnel diffraction using FFT:
```
U(z) = IFFT{ FFT{U(0)} * H(fx, fy) }

H(fx, fy) = exp(i*k*z) * exp(-i*π*λ*z*(fx² + fy²))
```
- **Multi-wavelength**: Pre-computed transfer functions
- **Power tracking**: Monitors energy conservation
- **Zero padding**: Prevents edge artifacts

## Advanced Features

### Periodic Phase Optimization
Reduces 2π discontinuities after training for better manufacturability.

### Power Loss Analysis
Tracks power loss at each layer with cumulative and per-layer metrics saved to `parameters.txt`.

### Multi-Run Training
```bash
python run_n_times.py
```

### Parameter Sweep
```bash
python multiple_iteration.py
```

## Monitoring

### Loss Components
- **MSE**: Mean squared error vs target
- **Smoothness**: Phase continuity
- **Power loss**: Energy efficiency
- **Focal intensity**: Central focusing

### Phase Evolution
Phase histograms saved every epoch show DOE optimization progress.

## GPU Configuration

Automatic GPU memory management:
```python
memory_limit_mb = 30720  # 30GB limit
```

## Troubleshooting

### NaN Values
- Reduce learning rate
- Check input data for NaN/Inf
- Increase epsilon in normalization

### Out of Memory
- Reduce `BATCH_SIZE`
- Decrease `padding_multiplier`
- Lower `memory_limit_mb`

### Poor Convergence
- Adjust loss weights
- Modify learning rate schedule
- Increase epochs
- Verify data quality


## References

- TensorFlow Custom Layers: https://www.tensorflow.org/tutorials/customization/custom_layers
- Fresnel Diffraction: Goodman, J. W. "Introduction to Fourier Optics"

## License

Part of the CD2NN-for-THz framework for designing diffractive optical elements using neural networks.
