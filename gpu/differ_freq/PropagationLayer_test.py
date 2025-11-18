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
        self.power_loss = None
        self.padding_multiplier = 8
        
        # Store as NumPy array (NOT TensorFlow constant) for build() phase
        self.wavelengths_np = np.arange(
            self.wavelength_min, 
            self.wavelength_max + self.wavelength_step, 
            self.wavelength_step
        ).astype(np.float32)

    def build(self, input_shape):
        """
        FIXED: All computations use NumPy arrays, converted to TF only at the end
        """
        # OPTIMIZATION 3: Compute padded dimensions only once
        self.H_padded = self.H * (self.padding_multiplier + 1)
        self.W_padded = self.W * (self.padding_multiplier + 1)
        self.pad_size = int(self.H_padded / (self.padding_multiplier + 1) * 
                           self.padding_multiplier / 2)
        
        dx = self.pixel_size
        
        # Generate frequency grids with NumPy
        fx = np.fft.fftfreq(self.W_padded, d=dx).astype(np.float32)
        fy = np.fft.fftfreq(self.H_padded, d=dx).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)
        r2 = (FX**2 + FY**2).astype(np.float32)
        
        # Generate h_real, h_imag for each wavelength using NumPy
        h_table = []
        
        # FIX: Iterate over NumPy array, not TensorFlow constant
        for wl in self.wavelengths_np:
            k = 2 * np.pi / wl
            
            # FIX: All operations in NumPy (no TensorFlow types)
            arg = -1j * np.pi * self.distance * wl * r2
            h = np.exp(1j * k * self.distance) * np.exp(arg)
            
            h_real = np.real(h).astype(np.float32)
            h_imag = np.imag(h).astype(np.float32)
            h_wavelength = np.full_like(h_real, wl, dtype=np.float32)
            
            # Stack real, imag, wavelength as channels
            h_table.append(np.stack([h_real, h_imag, h_wavelength], axis=-1))
        
        h_table = np.stack(h_table, axis=0)  # shape: [num_wavelengths, H_padded, W_padded, 3]
        
        print(f"h_table shape: {h_table.shape}")
        print(f"h_table memory: {h_table.nbytes / 1e9:.2f} GB")
        
        # Convert to TensorFlow weight AFTER all NumPy computations
        self.h_table = self.add_weight(
            name="h_table",
            shape=h_table.shape,
            initializer=tf.constant_initializer(h_table),
            trainable=False,
            dtype=tf.float32
        )
        
        # Convert wavelengths to TensorFlow AFTER build phase
        self.wavelengths = tf.constant(self.wavelengths_np, dtype=tf.float32)

    def call(self, inputs):
        """
        FIX: Improved efficiency and numerical stability
        """
        inputs = tf.cast(inputs, tf.float32)
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]
        wavelength = inputs[..., 2]

        # Replace NaN values
        re_u = tf.where(tf.math.is_nan(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u), tf.zeros_like(im_u), im_u)

        # Calculate input power
        input_power = tf.reduce_sum(re_u**2 + im_u**2)

        # OPTIMIZATION 11: Use tf.pad instead of Keras ZeroPadding2D (faster)
        re_u = tf.pad(re_u, [[0, 0], [self.pad_size, self.pad_size], 
                              [self.pad_size, self.pad_size]])
        im_u = tf.pad(im_u, [[0, 0], [self.pad_size, self.pad_size], 
                              [self.pad_size, self.pad_size]])

        # Get wavelength for each sample in batch
        batch_size = tf.shape(wavelength)[0]
        wl_values = wavelength[:, 0, 0]  # Shape: (batch_size,)

        # OPTIMIZATION 12: Use searchsorted instead of map_fn (5-10x faster)
        wl_indices = tf.searchsorted(self.wavelengths, wl_values)
        wl_indices = tf.minimum(wl_indices, tf.shape(self.wavelengths)[0] - 1)

        # Gather kernels for all wavelengths in batch
        h_real_batch = tf.gather(self.h_table[..., 0], wl_indices)  # (batch, H_padded, W_padded)
        h_imag_batch = tf.gather(self.h_table[..., 1], wl_indices)

        # Convert to complex for FFT operations
        h_real_batch = tf.cast(h_real_batch, tf.complex64)
        h_imag_batch = tf.cast(h_imag_batch, tf.complex64)

        # Perform FFT
        fft_re = tf.signal.rfft2d(re_u)  # (batch, H_padded, freq_width)
        fft_im = tf.signal.rfft2d(im_u)

        # Truncate kernels to match FFT output size
        freq_width = tf.shape(fft_re)[-1]
        h_real_batch = h_real_batch[..., :freq_width]
        h_imag_batch = h_imag_batch[..., :freq_width]

        # Convert FFT to complex64
        fft_re = tf.cast(fft_re, tf.complex64)
        fft_im = tf.cast(fft_im, tf.complex64)

        # OPTIMIZATION 15: Fused multiplication (real and imaginary parts)
        out_real_fft = fft_re * h_real_batch - fft_im * h_imag_batch
        out_imag_fft = fft_re * h_imag_batch + fft_im * h_real_batch

        # Inverse FFT
        out_real = tf.cast(tf.signal.irfft2d(out_real_fft), tf.float32)
        out_imag = tf.cast(tf.signal.irfft2d(out_imag_fft), tf.float32)

        # Calculate output power before cropping
        output_power = tf.reduce_sum(out_real**2 + out_imag**2)
        self.power_loss = (input_power - output_power) / (input_power + 1e-10)

        # OPTIMIZATION 16: Use slicing instead of Cropping2D layer (faster)
        out_real = out_real[:, self.pad_size:self.pad_size+self.H, 
                           self.pad_size:self.pad_size+self.W]
        out_imag = out_imag[:, self.pad_size:self.pad_size+self.H, 
                           self.pad_size:self.pad_size+self.W]

        # OPTIMIZATION 17: Improved normalization with higher epsilon
        epsilon = 1e-7  # Increased from 1e-14 for numerical stability
        max_real = tf.reduce_max(tf.abs(out_real), keepdims=True)
        max_imag = tf.reduce_max(tf.abs(out_imag), keepdims=True)

        out_real = out_real / (max_real + epsilon)
        out_imag = out_imag / (max_imag + epsilon)

        # Stack and return
        return tf.stack([out_real, out_imag, wavelength], axis=-1)
