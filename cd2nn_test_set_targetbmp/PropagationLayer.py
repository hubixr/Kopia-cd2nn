import tensorflow as tf
import numpy as np
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
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]
         # Ensure the input tensor has a batch dimension
        if len(inputs.shape) == 3:  # If the input is missing the batch dimension
            inputs = tf.expand_dims(inputs, axis=0)  # Add batch dimension
            # Debugging: Check for NaN or Inf values in tensors
        if tf.reduce_any(tf.math.is_nan(re_u)) or tf.reduce_any(tf.math.is_inf(re_u)):
            print("NaN or Inf detected in re_u")
        if tf.reduce_any(tf.math.is_nan(im_u)) or tf.reduce_any(tf.math.is_inf(im_u)):
            print("NaN or Inf detected in im_u")
        if tf.reduce_any(tf.math.is_nan(self.h_real)) or tf.reduce_any(tf.math.is_inf(self.h_real)):
            print("NaN or Inf detected in h_real")
        if tf.reduce_any(tf.math.is_nan(self.h_imag)) or tf.reduce_any(tf.math.is_inf(self.h_imag)):
            print("NaN or Inf detected in h_imag")

        # Expand dimensions for convolution
        h_real = tf.expand_dims(self.h_real, axis=-1)  # Shape: [H, W, 1]
        h_real = tf.expand_dims(h_real, axis=-1)       # Shape: [H, W, 1, 1]
        h_imag = tf.expand_dims(self.h_imag, axis=-1)  # Shape: [H, W, 1]
        h_imag = tf.expand_dims(h_imag, axis=-1)       # Shape: [H, W, 1, 1]
        if tf.reduce_any(tf.math.is_nan(re_u)) or tf.reduce_any(tf.math.is_inf(re_u)):
            print("2 NaN or Inf detected in re_u")
        if tf.reduce_any(tf.math.is_nan(im_u)) or tf.reduce_any(tf.math.is_inf(im_u)):
            print("2 NaN or Inf detected in im_u")
        if tf.reduce_any(tf.math.is_nan(h_real)) or tf.reduce_any(tf.math.is_inf(h_real)):
            print("2 NaN or Inf detected in h_real")
        if tf.reduce_any(tf.math.is_nan(h_imag)) or tf.reduce_any(tf.math.is_inf(h_imag)):
            print("2 NaN or Inf detected in h_imag")

        print("first conv")
        re_re = tf.nn.conv2d(
            input=tf.expand_dims(re_u, axis=-1),
            filters=h_real,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        print("second conv")
        im_im = tf.nn.conv2d(
            input=tf.expand_dims(im_u, axis=-1),
            filters=h_imag,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        print("third conv")
        re_im = tf.nn.conv2d(
            input=tf.expand_dims(re_u, axis=-1),
            filters=h_imag,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        print("fourth conv")
        im_re = tf.nn.conv2d(
            input=tf.expand_dims(im_u, axis=-1),
            filters=h_real,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

        print("end of conv")

        out_real = tf.squeeze(re_re - im_im, axis=-1)
        out_imag = tf.squeeze(re_im + im_re, axis=-1)
        out_real = out_real / tf.reduce_max(out_real) 
        out_imag = out_imag / tf.reduce_max(out_imag) 
        return tf.stack([out_real, out_imag], axis=-1)