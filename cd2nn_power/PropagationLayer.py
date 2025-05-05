import tensorflow as tf
import numpy as np

"""
PropagationLayer: A TensorFlow custom layer for simulating light propagation.

This layer models the propagation of an optical field over a specified distance using FFT-based convolutions. It is designed for use in computational diffractive neural networks (CDNNs).

Key Features:
- Simulates light propagation efficiently using FFT.
- Handles complex input fields represented as two channels (Re, Im).
- Precomputes the propagation kernel during initialization for faster execution.

Attributes:
- `h_real`: Precomputed real part of the propagation kernel.
- `h_imag`: Precomputed imaginary part of the propagation kernel.

Methods:
- `call(inputs)`: Applies the propagation kernel to the input field and returns the propagated field.

Inputs:
- Tensor of shape `[B, H, W, 2]` representing the complex input field (real and imaginary parts).

Outputs:
- Tensor of shape `[B, H, W, 2]` representing the propagated complex field (real and imaginary parts).
"""

class PropagationLayer(tf.keras.layers.Layer):
    def __init__(self, wavelength, distance, pixel_size, shape, name=None):
        super(PropagationLayer, self).__init__(name=name)
        self.wavelength = wavelength
        self.distance = distance
        self.pixel_size = pixel_size
        self.shape_ = shape

    def build(self, input_shape):
        H, W = self.shape_
        dx = self.pixel_size
        x = np.arange(-W // 2, W // 2) * dx
        y = np.arange(-H // 2, H // 2) * dx
        X, Y = np.meshgrid(x, y)

        r2 = X**2 + Y**2
        k = 2 * np.pi / self.wavelength
        arg = k * self.distance + (np.pi * r2) / (self.wavelength * self.distance)
        denom = self.wavelength * self.distance

        # Compute h_real and h_imag using NumPy
        h_real = np.sin(arg) / denom
        h_imag = -np.cos(arg) / denom

        # Apply fftshift to the kernel during initialization
        h_real = np.fft.fftshift(h_real)
        h_imag = np.fft.fftshift(h_imag)

        # Use NumPy arrays for tf.constant_initializer
        self.h_real = self.add_weight(
            name="h_real",
            shape=h_real.shape,
            initializer=tf.constant_initializer(h_real),
            trainable=False
        )
        self.h_imag = self.add_weight(
            name="h_imag",
            shape=h_imag.shape,
            initializer=tf.constant_initializer(h_imag),
            trainable=False
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]

        # Perform 4 fft convolutions
        print("Start of convolutions")
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_real))
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_imag))
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_imag))
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_real))
        print("End of convolutions")

        # Compute real and imaginary parts of the output
        out_real = re_re - im_im
        out_imag = re_im + im_re

        # # Normalize the outputs
        # out_real = out_real / tf.reduce_max(out_real)
        # out_imag = out_imag / tf.reduce_max(out_imag)

        # print("Min and max of out_real:", tf.reduce_min(out_real), tf.reduce_max(out_real))
        # print("Min and max of out_imag:", tf.reduce_min(out_imag), tf.reduce_max(out_imag))
        print("Output real shape:", out_real.shape)
        print("Output imaginary shape:", out_imag.shape)
        print("output shape:", tf.stack([out_real, out_imag], axis=-1).shape)
        # Check for NaN or Inf in the outputs using TensorFlow operations
        # tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real")
        # tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag")

        return tf.stack([out_real, out_imag], axis=-1)
