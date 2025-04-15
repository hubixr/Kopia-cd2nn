# CDNN Layers

This directory contains two custom TensorFlow layers designed for computational diffractive neural networks (CDNNs). These layers simulate optical phenomena such as phase modulation and wave propagation.

## Files

### 1. `FirstDiffractiveMaskLayer.py`

This file defines the `DiffractiveMaskLayer` class, which represents a diffractive optical element (DOE) that introduces a phase delay to an input optical field. 

- **Input**: A tensor of shape `[B, H, W, 2]`, where the last dimension represents the real and imaginary parts of the optical field.
- **Output**: A tensor of the same shape `[B, H, W, 2]`, with the field modulated by the phase mask.
- **Trainable Parameter**: The phase map of the DOE, initialized randomly and optimized during training.

### 2. `PropagationLayer.py`

This file defines the `PropagationLayer` class, which simulates the propagation of an optical field over a specified distance.

- **Input**: A tensor of shape `[B, H, W, 2]`, where the last dimension represents the real and imaginary parts of the optical field.
- **Output**: A tensor of the same shape `[B, H, W, 2]`, representing the field after propagation.
- **Parameters**:
  - `wavelength`: Wavelength of light (in meters).
  - `distance`: Propagation distance (in meters).
  - `pixel_size`: Size of a pixel (in meters).
  - `shape`: Shape of the optical field (height and width).

## Usage

These layers can be used in TensorFlow models to simulate optical systems. The `DiffractiveMaskLayer` applies phase modulation, while the `PropagationLayer` models the propagation of light through free space or other media.

## References

- TensorFlow Custom Layers: [https://www.tensorflow.org/tutorials/customization/custom_layers](https://www.tensorflow.org/tutorials/customization/custom_layers)
