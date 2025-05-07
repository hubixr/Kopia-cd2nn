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
        # Generate plane wave with apertures of random diameters
        aperture_diameter = tf.random.uniform([], minval=40, maxval=128, dtype=tf.int32)
        center = (self.shape_[0] // 2, self.shape_[1] // 2)
        y, x = tf.meshgrid(tf.range(self.shape_[0]), tf.range(self.shape_[1]), indexing='ij')
        mask = tf.square(x - center[1]) + tf.square(y - center[0]) <= tf.square(aperture_diameter // 2)
        mask = tf.cast(mask, tf.float32)

        # Apply the aperture mask to the input field
        inputs = inputs * mask[..., tf.newaxis]

        field = inputs

        for i, (doe, prop) in enumerate(zip(self.doe_layers, self.prop_layers)):
            field = doe(field)
            field = prop(field)

        U_real = field[..., 0]
        U_imag = field[..., 1]
        intensity = tf.square(U_real)+tf.square(U_imag)  # intensity = |U|^2
        print(
            "Intensity min:", tf.reduce_min(intensity),
            "max:", tf.reduce_max(intensity),
            "mean:", tf.reduce_mean(intensity)
        )
        # print("Intensity shape:", intensity.shape)
        amplitude = tf.sqrt(intensity)
        # print("Amplitude shape:", amplitude.shape)
        print(
            "Amplitude min:", tf.reduce_min(amplitude),
            "max:", tf.reduce_max(amplitude),
            "mean:", tf.reduce_mean(amplitude)
        )

        return intensity