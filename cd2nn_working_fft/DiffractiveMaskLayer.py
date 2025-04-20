import tensorflow as tf
import numpy as np

"""
DiffractiveMaskLayer: A TensorFlow custom layer representing a diffractive optical element (DOE).

This layer introduces a phase delay to the input optical field, simulating the effect of a DOE. The phase map is a trainable parameter, allowing the network to optimize the DOE for specific tasks.

Key Features:
- Trainable phase map initialized randomly (can be replaced with a constant value).
- Handles complex input fields represented as two channels (Re, Im).
- Ensures numerical stability by handling NaN and Inf values in the input and output fields.

Attributes:
- `phase`: Trainable weight representing the phase map of the DOE.

Methods:
- `call(inputs)`: Applies the phase modulation to the input field and returns the modulated field.

Inputs:
- Tensor of shape `[B, H, W, 2]` representing the complex input field (real and imaginary parts).

Outputs:
- Tensor of shape `[B, H, W, 2]` representing the modulated complex field (real and imaginary parts).
"""

class DiffractiveMaskLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(DiffractiveMaskLayer, self).__init__(name=name)
        self.shape_ = shape

    def build(self, input_shape):
        # Initialize phase as a trainable weight
        self.phase = self.add_weight(
            name="phase",
            shape=self.shape_,
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2 * np.pi),
            trainable=True
        )
        super(DiffractiveMaskLayer, self).build(input_shape)
    
    # Add explicit casting to float16 for phase and inputs
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)  # Cast inputs to float16
        print("Doe call")
        re_u = inputs[..., 0]  # Real part
        im_u = inputs[..., 1]  # Imaginary part
        print("checking for nans and infs in diffraction layer at the beginning")
        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
    
        # Ensure phase is properly managed by TensorFlow
        phase = tf.cast(tf.identity(self.phase), tf.float16)  # Cast phase to float16
        print("phase shape:", phase.shape)
        # Apply phase modulation
        if im_u is None:
            out_real = re_u * tf.cos(phase)
            out_imag = re_u * tf.sin(phase)
        else:
            out_real = re_u * tf.cos(phase) - im_u * tf.sin(phase)
            out_imag = re_u * tf.sin(phase) + im_u * tf.cos(phase)
        print("checking for nans and infs in diffraction layer at the end")
        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
        out_real = tf.where(tf.math.is_nan(out_real) | tf.math.is_inf(out_real), tf.zeros_like(out_real), out_real)
        out_imag = tf.where(tf.math.is_nan(out_imag) | tf.math.is_inf(out_imag), tf.zeros_like(out_imag), out_imag)
        print("end doe call")
        return tf.stack([out_real, out_imag], axis=-1)  # [B, H, W, 2]