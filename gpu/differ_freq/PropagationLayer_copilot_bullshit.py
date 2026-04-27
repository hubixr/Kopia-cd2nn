import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PropagationLayer(tf.keras.layers.Layer):
    def __init__(self, wavelength_min, wavelength_max, wavelength_step, distance, pixel_size, shape, name=None):
        super(PropagationLayer, self).__init__(name=name)
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.wavelength_step = wavelength_step
        self.distance = distance
        self.pixel_size = pixel_size
        self.shape_ = shape
        self.H, self.W = shape
        self.power_loss = None  # Placeholder for power loss
        self.padding_multiplier = 5
        # Prepare wavelength table
        self.wavelengths = np.arange(
            self.wavelength_min, 
            self.wavelength_max + self.wavelength_step, 
            self.wavelength_step
        )

    def build(self, input_shape):
        self.H = self.H * (self.padding_multiplier + 1)
        self.W = self.W * (self.padding_multiplier + 1)
        dx = self.pixel_size
        H = self.H
        W = self.W
        fx = np.fft.fftfreq(W, d=dx)
        fy = np.fft.fftfreq(H, d=dx)
        FX, FY = np.meshgrid(fx, fy)
        r2 = FX**2 + FY**2
        # Generate h_real, h_imag, and wavelength for each wavelength
        h_table = []
        for wl in self.wavelengths:
            k = 2 * np.pi / wl
            arg = -1j * np.pi * self.distance * wl * r2
            h = np.exp(1j * k * self.distance) * np.exp(arg)
            h_real = np.real(h)
            h_imag = np.imag(h)
            h_wavelength = np.full_like(h_real, wl)
            # Stack real, imag, wavelength as channels
            h_table.append(np.stack([h_real, h_imag, h_wavelength], axis=-1))  # shape: [H, W, 3]
        h_table = np.stack(h_table, axis=0)  # shape: [num_wavelengths, H, W, 3]
        print("h_table shape (full frequency domain):", h_table.shape, "total elements:", h_table.size)
        # print("h_table values:\n", h_table)
        self.h_table = self.add_weight(
            name="h_table",
            shape=h_table.shape,
            initializer=tf.constant_initializer(h_table),
            trainable=False
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]
        wavelength = inputs[..., 2]  # third channel: frequency or wavelength

        # Replace NaN values in re_u and im_u with zeros
        re_u = tf.where(tf.math.is_nan(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u), tf.zeros_like(im_u), im_u)

        # Ensure inputs are finite
        tf.debugging.assert_all_finite(re_u, "NaN or Inf detected in re_u after replacement")
        tf.debugging.assert_all_finite(im_u, "NaN or Inf detected in im_u after replacement")

        input_power = tf.reduce_sum(re_u**2 + im_u**2)
        size = int(self.H / (self.padding_multiplier+1) * self.padding_multiplier/2)

        # Add zero padding using tf.pad (faster than ZeroPadding2D layer)
        re_u = tf.pad(re_u, [[0, 0], [size, size], [size, size]])
        im_u = tf.pad(im_u, [[0, 0], [size, size], [size, size]])
        
        tf.debugging.assert_all_finite(re_u, "NaN or Inf detected in padded re_u")
        tf.debugging.assert_all_finite(im_u, "NaN or Inf detected in padded im_u")


        # Process each sample in batch with its corresponding wavelength
        batch_size = tf.shape(wavelength)[0]
        if len(wavelength.shape) == 2:
            raise ValueError("No wavelength channel found in input data. Please check the input data format.")
        
        # Get wavelength for each sample in batch
        wl_values = wavelength[:, 0, 0]  # Shape: (batch_size,)
        
        # Print current wavelength values being used
        # tf.print("Current wavelength values (m):", wl_values, summarize=-1)
        # tf.print("Current wavelength values (μm):", wl_values * 1e6, summarize=-1)
        
        # Find wavelength indices for all samples at once
        wl_indices = tf.map_fn(
            lambda wl: tf.argmin(tf.abs(self.wavelengths - wl)),
            wl_values,
            dtype=tf.int64
        )
        
        # tf.print("Selected wavelength indices for batch:", wl_indices)
        
        # Gather kernels for all wavelengths in the batch
        # Note: h_table contains Fresnel kernels in full frequency domain (H × W)
        h_real_batch = tf.gather(self.h_table[..., 0], wl_indices)  # Shape: (batch_size, H, W)
        h_imag_batch = tf.gather(self.h_table[..., 1], wl_indices)  # Shape: (batch_size, H, W)
        
        print("h_real_batch shape (full freq domain):", h_real_batch.shape)
        print("h_imag_batch shape (full freq domain):", h_imag_batch.shape)
        print("re_u shape (spatial before FFT):", re_u.shape)
        print("im_u shape (spatial before FFT):", im_u.shape)

        # FFT the input field
        re_u_fft = tf.signal.rfft2d(re_u)  # (batch, H, W//2+1)
        im_u_fft = tf.signal.rfft2d(im_u)  # (batch, H, W//2+1)
        
        # Extract only the rfft2d half from the kernel (first W//2+1 columns)
        W_fft = tf.shape(re_u_fft)[-1]  # Should be W//2+1
        h_real_fft = h_real_batch[:, :, :W_fft]  # (batch, H, W//2+1)
        h_imag_fft = h_imag_batch[:, :, :W_fft]  # (batch, H, W//2+1)
        
        print("h_real_fft shape (rfft2d half):", h_real_fft.shape)
        print("h_imag_fft shape (rfft2d half):", h_imag_fft.shape)
        print("re_u_fft shape:", re_u_fft.shape)
        print("im_u_fft shape:", im_u_fft.shape)
        
        # Convert to complex64 for operations
        re_u_fft = tf.cast(re_u_fft, tf.complex64)
        im_u_fft = tf.cast(im_u_fft, tf.complex64)
        h_real_fft = tf.cast(h_real_fft, tf.complex64)
        h_imag_fft = tf.cast(h_imag_fft, tf.complex64)
        
        # Combine into complex form
        h_fft = h_real_fft + 1j * h_imag_fft
        u_fft = re_u_fft + 1j * im_u_fft
        
        # Multiply input FFT with kernel FFT in frequency domain
        out_fft = u_fft * h_fft
        
        # Convert back to spatial domain
        out_complex = tf.signal.irfft2d(out_fft)
        out_real = tf.cast(tf.math.real(out_complex), tf.float32)
        out_imag = tf.cast(tf.math.imag(out_complex), tf.float32)

        output_power = tf.reduce_sum(out_real**2 + out_imag**2)
        self.power_loss = ((input_power - output_power) / input_power)
        
        tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real before cropping")
        tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag before cropping")

        # Crop using slicing instead of Cropping2D layer (faster)
        out_real = out_real[:, size:size+128, size:size+128]
        out_imag = out_imag[:, size:size+128, size:size+128]
        
        epsilon = 1e-14
        out_real = out_real / (tf.reduce_max(tf.abs(out_real)) + epsilon)
        out_imag = out_imag / (tf.reduce_max(tf.abs(out_imag)) + epsilon)
        tf.debugging.assert_all_finite(out_real, "NaN or Inf detected in out_real after cropping")
        tf.debugging.assert_all_finite(out_imag, "NaN or Inf detected in out_imag after cropping")

        return tf.stack([out_real, out_imag, wavelength], axis=-1)
