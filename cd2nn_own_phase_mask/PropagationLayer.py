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
        mask = self.wavelength**2 * r2 > 1
        print("PROBBLEEMMYYYY: ", np.sum(mask))  # ile punktów ma problematyczny pierwiastek

        
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
        power_before = tf.reduce_sum(re_u**2 + im_u**2)
        # Compute intensity as the sum of squares of real and imaginary parts
        # Replace .numpy() with tf.print for compatibility during graph execution
        tf.print("Power before:", tf.reduce_sum(power_before))
        # print("re_u shape:", re_u.shape)
        # print("im_u shape:", im_u.shape)
        size = int(re_u.shape[1]/2)
        # print("Size of input:", size)
        # #Add zero padding
        re_u = tf.expand_dims(re_u, axis=-1)
        im_u = tf.expand_dims(im_u, axis=-1)
        # print("re_u shape after expand_dims:", re_u.shape)
        re_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(re_u)
        im_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(im_u)
        # print("re_u shape after padding:", re_u.shape)

        #Delete last dimension
        re_u = tf.squeeze(re_u, axis=-1)
        im_u = tf.squeeze(im_u, axis=-1)
        # print("re_u shape after squeeze:", re_u.shape)
        # print("im_u shape after squeeze:", im_u.shape)
        # Perform 4 fft convolutions
        intensity = re_u**2 + im_u**2
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='Intensity')
        plt.title('Intensity Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('intensity_plot_before.png')
        plt.close()
        print("Start of convolutions")
        # tf.print("h_real min/max:", tf.reduce_min(self.h_real), tf.reduce_max(self.h_real))
        # tf.print("h_imag min/max:", tf.reduce_min(self.h_imag), tf.reduce_max(self.h_imag))
        # tf.print("h_real sum:", tf.reduce_sum(self.h_real))
        print("Number of pixels in re_re:", tf.size(re_u).numpy())
        N = tf.cast(tf.shape(re_u)[1] * tf.shape(re_u)[2], tf.float32)  # szer. * wys.
        h_real_frequency = tf.cast(self.h_real[None, :, :self.shape_[1] + 1], tf.complex64)
        h_imag_frequency = tf.cast(self.h_imag[None, :, :self.shape_[1] + 1], tf.complex64)
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_real_frequency) /tf.sqrt(N**3)
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_imag_frequency) /tf.sqrt(N**3)
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_imag_frequency) /tf.sqrt(N**3)
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_real_frequency) /tf.sqrt(N**3)
        # print("re_re shape:", re_re.shape)
        print("End of convolutions")
        
        
        # Compute real and imaginary parts of the output
        out_real = re_re - im_im
        out_imag = re_im + im_re

        power_after_convolution = out_real**2 + out_imag**2
        tf.print("Power after convolutions:", tf.reduce_sum(power_after_convolution))


        #Crop to original size
        out_real = tf.expand_dims(out_real, axis=-1)
        out_imag = tf.expand_dims(out_imag, axis=-1)
        out_real = tf.keras.layers.Cropping2D(cropping=(size, size))(out_real)
        out_imag = tf.keras.layers.Cropping2D(cropping=(size, size))(out_imag)
        # print("out_real shape after cropping:", out_real.shape)
        # print("out_imag shape after cropping:", out_imag.shape)
        out_real = tf.squeeze(out_real, axis=-1)
        out_imag = tf.squeeze(out_imag, axis=-1)
        power_after = tf.reduce_sum(out_real**2 + out_imag**2)
        intensity = out_real**2 + out_imag**2
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='Intensity')
        plt.title('Intensity Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('intensity_plot_after.png')
        plt.close()
        # tf.print("Power after cropping:", tf.reduce_sum(power_after))
        # tf.print("Power loss after cropping:", power_after / power_after_convolution*100)
        # print("out_real shape after squeeze:", out_real.shape)
        # print("out_imag shape after squeeze:", out_imag.shape)
        # # Normalize the outputs
        # out_real = out_real / tf.reduce_max(out_real)
        # out_imag = out_imag / tf.reduce_max(out_imag)

        # print("Min and max of out_real:", tf.reduce_min(out_real), tf.reduce_max(out_real))
        # print("Min and max of out_imag:", tf.reduce_min(out_imag), tf.reduce_max(out_imag))
        # print("Output real shape:", out_real.shape)
        # print("Output imaginary shape:", out_imag.shape)
        print("output shape:", tf.stack([out_real, out_imag], axis=-1).shape)
        tf.print("Power after resizing:", tf.reduce_sum(tf.reduce_sum(out_real**2 + out_imag**2)))
        # Check for NaN or Inf in the outputs using TensorFlow operations
        # tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real")
        # tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag")
        
        

        return tf.stack([out_real, out_imag], axis=-1)
