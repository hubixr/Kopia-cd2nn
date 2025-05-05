import tensorflow as tf
from DiffractiveMaskLayer import DiffractiveMaskLayer
from PropagationLayer import PropagationLayer

"""
CDNNModel: A TensorFlow-based model for designing computational diffractive neural networks (CDNNs).

This model simulates the propagation of light through a series of diffractive optical elements (DOEs) and free-space propagation layers.

Key Features:
- Supports multiple DOEs and propagation layers.
- Uses FFT for efficient simulation of light propagation.
- Outputs the normalized amplitude of the optical field after passing through the structure.

Attributes:
- `doe_layers`: List of trainable diffractive optical element layers.
- `prop_layers`: List of propagation layers for simulating light propagation.

Methods:
- `call(inputs)`: Simulates the optical system and returns the normalized amplitude of the output field.

Inputs:
- Tensor of shape `[B, H, W, 2]` representing the complex input field (real and imaginary parts).

Outputs:
- Tensor of shape `[B, H, W]` representing the normalized amplitude of the output field.
"""

class CDNNModel(tf.keras.Model):

    def __init__(self, num_layers, shape, wavelength, distance_between_layers, distance_to_plane, pixel_size, name=None):
        super(CDNNModel, self).__init__(name=name)
        self.shape_ = shape
        self.doe_layers = []
        self.prop_layers = []

        for i in range(num_layers - 1):
            self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{i + 1}"))
            self.prop_layers.append(PropagationLayer(
                wavelength, distance_between_layers, pixel_size, shape, name=f"prop_{i + 1}"
            ))
            print(f"Layer {i + 1}: DOE + Propagation z={distance_between_layers} m")

        self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{num_layers}"))
        self.prop_layers.append(PropagationLayer(
            wavelength, distance_to_plane, pixel_size, shape, name=f"prop_{num_layers}"
        ))
        print(f"Final Layer: DOE + Propagation z={distance_to_plane} m")


    def call(self, inputs):
        field = inputs

        for i, (doe, prop) in enumerate(zip(self.doe_layers, self.prop_layers)):
            field = doe(field)
            field = prop(field)

        U_real = field[..., 0]
        U_imag = field[..., 1]
        U_complex = tf.complex(U_real, U_imag)
        intensity = tf.square(U_complex)  # intensity = |U|^2
        amplitude = tf.cast(tf.sqrt(intensity),dtype=tf.float32) #/ tf.reduce_max(tf.sqrt(intensity))  # Amplitude normalization
        # print("AAmplitude shape",amplitude.shape)
        amplitude = amplitude / tf.reduce_max(amplitude)  # Normalize amplitude
        print(
            "Amplitude min:", tf.reduce_min(amplitude),
            "max:", tf.reduce_max(amplitude),
            "mean:", tf.reduce_mean(amplitude)
        )

        return amplitude  