import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
        x = np.arange(-W, W) * dx
        y = np.arange(-H, H) * dx
        fx = np.fft.fftfreq(2*W, d=dx)  # oś pozioma
        fy = np.fft.fftfreq(2*H, d=dx)  # oś pionowa
        # Siatka częstotliwości
        FX, FY = np.meshgrid(fx, fy)
        r2 = FX**2 + FY**2
        k = 2 * np.pi / self.wavelength
        # arg = -1j * 2 * np.pi * self.distance / self.wavelength * np.sqrt(1 - self.wavelength**2 * r2)
        # h = np.exp(arg)
        arg = -1j*np.pi*self.distance * self.wavelength * r2
        h = np.exp(1j*k*self.distance)*np.exp(arg)

        h_real = np.real(h)
        h_imag = np.imag(h)

        print("h_real shape:", h_real.shape)
        print("h_imag shape:", h_imag.shape)
        
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
        inputs = tf.cast(inputs, tf.float32)
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]

        # Replace NaN values in re_u and im_u with zeros
        re_u = tf.where(tf.math.is_nan(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u), tf.zeros_like(im_u), im_u)

        # Ensure inputs are finite
        tf.debugging.assert_all_finite(re_u, "NaN or Inf detected in re_u after replacement")
        tf.debugging.assert_all_finite(im_u, "NaN or Inf detected in im_u after replacement")

        power_before = tf.reduce_sum(re_u**2 + im_u**2)
        size = int(re_u.shape[1] / 2)

        # Add zero padding
        re_u = tf.expand_dims(re_u, axis=-1)
        im_u = tf.expand_dims(im_u, axis=-1)
        re_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(re_u)
        im_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(im_u)
        re_u = tf.squeeze(re_u, axis=-1)
        im_u = tf.squeeze(im_u, axis=-1)

        # Ensure padded inputs are finite
        tf.debugging.assert_all_finite(re_u, "NaN or Inf detected in padded re_u")
        tf.debugging.assert_all_finite(im_u, "NaN or Inf detected in padded im_u")

        print("Start of convolutions")
        N = tf.cast(tf.shape(re_u)[1] * tf.shape(re_u)[2], tf.float32)
        h_real_frequency = tf.cast(self.h_real[None, :, :self.shape_[1] + 1], tf.complex64)
        h_imag_frequency = tf.cast(self.h_imag[None, :, :self.shape_[1] + 1], tf.complex64)

        # Perform FFT-based convolutions with normalization
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_real_frequency) / tf.sqrt(N)
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_imag_frequency) / tf.sqrt(N)
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_imag_frequency) / tf.sqrt(N)
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_real_frequency) / tf.sqrt(N)

        print("End of convolutions")

        # Compute real and imaginary parts of the output
        out_real = re_re - im_im
        out_imag = re_im + im_re

        # Ensure outputs are finite before cropping
        tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real before cropping")
        tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag before cropping")

        # Crop to original size
        out_real = tf.expand_dims(out_real, axis=-1)
        out_imag = tf.expand_dims(out_imag, axis=-1)
        out_real = tf.keras.layers.Cropping2D(cropping=(size, size))(out_real)
        out_imag = tf.keras.layers.Cropping2D(cropping=(size, size))(out_imag)
        out_real = tf.squeeze(out_real, axis=-1)
        out_imag = tf.squeeze(out_imag, axis=-1)
        #normalize the outputs
        out_real = out_real / tf.reduce_max(out_real)
        out_imag = out_imag / tf.reduce_max(out_imag)
        # Ensure outputs are finite after cropping
        tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real after cropping")
        tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag after cropping")

        print("output shape:", tf.stack([out_real, out_imag], axis=-1).shape)

        return tf.stack([out_real, out_imag], axis=-1)
