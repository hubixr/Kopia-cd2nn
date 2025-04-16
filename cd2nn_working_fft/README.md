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

# CD2NN Working FFT

This directory contains the implementation of the CD2NN model using the FFT (Fast Fourier Transform) approach for diffractive optical element (DOE) design and training. The FFT-based approach is used to simulate light propagation efficiently.

## Directory Structure

- **cd2nn_model.py**: Contains the implementation of the CD2NN model architecture.
- **cdnn_model_v2.keras**: The saved trained model in Keras format.
- **conv2d_manual.py**: Implements manual 2D convolution operations.
- **DiffractiveMaskLayer.py**: Defines the diffractive mask layer used in the model.
- **FirstDiffractiveMaskLayer.py**: Implements the first diffractive mask layer.
- **learning_rate_vs_loss.png**: A graph showing the relationship between learning rate and loss during training.
- **logs/**: Directory containing training logs.
- **saved_histories/**: Directory containing saved training histories.
- **best_doe_masks/**: Directory containing the best-trained DOE phase masks.
- **cdnn_data/**: Directory containing input and target data for training and testing.

## Key Features

- **FFT-Based Propagation**: Efficient simulation of light propagation using FFT.
- **Customizable DOE Design**: Supports multiple diffractive mask layers for advanced optical designs.
- **Training and Testing**: Scripts for training the model and evaluating its performance.

## Usage

1. **Training**:
   Use the `train_cd2nn_model.py` script to train the model. Ensure the input and target data are prepared in the `cdnn_data/` directory.

2. **Testing**:
   Use the `test.ipynb` notebook to evaluate the model's performance and visualize the results.

3. **Visualization**:
   - Use `learning_rate_vs_loss.png` to analyze the effect of learning rate on training.
   - Visualize output intensity using `output_intensity.png`.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Notes

- Ensure the input data is normalized and matches the expected dimensions.
- Modify the model parameters in `cd2nn_model.py` as needed for your specific use case.

## Acknowledgments

This project is part of the CD2NN-for-THz framework for designing diffractive optical elements using neural networks.
