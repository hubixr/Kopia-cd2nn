import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PropagationLayer(tf.keras.layers.Layer):
    def __init__(self, wavelength, distance, pixel_size, shape, name=None):
        super(PropagationLayer, self).__init__(name=name)
        self.wavelength = wavelength
        self.distance = distance
        self.pixel_size = pixel_size
        self.shape_ = shape
        self.H, self.W = shape
        self.power_loss = None  # Placeholder for power loss
        self.padding_multiplier = 8

    def build(self, input_shape):
        self.H = self.H * (self.padding_multiplier + 1)
        self.W = self.W * (self.padding_multiplier + 1)
        dx = self.pixel_size
        H = self.H
        W = self.W
        x = np.arange(-W, W) * dx
        y = np.arange(-H, H) * dx
        fx = np.fft.fftfreq(W, d=dx)  # oś pozioma
        fy = np.fft.fftfreq(H, d=dx)  # oś pionowa
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

        # print("h_real shape:", h_real.shape)
        # print("h_imag shape:", h_imag.shape)
        
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

        input_power = tf.reduce_sum(re_u**2 + im_u**2)
        # print("Power before propagation:", input_power)
        # size = int(re_u.shape[1] / 2) # old padding sieze
        size = int(self.H / (self.padding_multiplier+1) * self.padding_multiplier/2) # new padding size

        # Add zero padding
        re_u = tf.expand_dims(re_u, axis=-1)
        im_u = tf.expand_dims(im_u, axis=-1)
        re_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(re_u)
        im_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(im_u)
        re_u = tf.squeeze(re_u, axis=-1)
        im_u = tf.squeeze(im_u, axis=-1)
        # print("re_u shape after padding:", re_u.shape)
        # Ensure padded inputs are finite
        tf.debugging.assert_all_finite(re_u, "NaN or Inf detected in padded re_u")
        tf.debugging.assert_all_finite(im_u, "NaN or Inf detected in padded im_u")

        print("Start of convolutions")
        N = tf.cast(tf.shape(re_u)[1] * tf.shape(re_u)[2], tf.float32)
        H_padding = int(self.H // 2 + 1)
        h_real_frequency = tf.cast(self.h_real[None, :, :H_padding], tf.complex64)
        h_imag_frequency = tf.cast(self.h_imag[None, :, :H_padding], tf.complex64)

        # Perform FFT-based convolutions with normalization
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_real_frequency)  
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_imag_frequency)  
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_imag_frequency)  
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_real_frequency) 
        # tf.print("re_re min:", tf.reduce_min(re_re), "max:", tf.reduce_max(re_re))

        print("End of convolutions")

        output_power = tf.reduce_sum(re_re**2 + im_im**2)
        # print("Power after propagation:", output_power)
        self.power_loss = ((input_power - output_power) / input_power)
        # tf.print("Power loss percentage:", (self.power_loss * 100), " %")
        # Compute real and imaginary parts of the output
        out_real = re_re - im_im
        out_imag = re_im + im_re
        # # Plot out_real and out_imag for the first sample in the batch
        # out_real_np = out_real[0].numpy() if hasattr(out_real, 'numpy') else tf.make_ndarray(tf.make_tensor_proto(out_real[0]))
        # out_imag_np = out_imag[0].numpy() if hasattr(out_imag, 'numpy') else tf.make_ndarray(tf.make_tensor_proto(out_imag[0]))

        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("out_real")
        # plt.imshow(out_real_np, cmap='gray')
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.title("out_imag")
        # plt.imshow(out_imag_np, cmap='gray')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()
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
        epsilon = 1e-14
        out_real = out_real / (tf.reduce_max(tf.abs(out_real)) + epsilon)
        out_imag = out_imag / (tf.reduce_max(tf.abs(out_imag)) + epsilon)
        # Ensure outputs are finite after cropping
        tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real after cropping")
        tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag after cropping")

        # print("output shape:", tf.stack([out_real, out_imag], axis=-1).shape)

        return tf.stack([out_real, out_imag], axis=-1)
