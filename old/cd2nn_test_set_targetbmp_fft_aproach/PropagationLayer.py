import tensorflow as tf
import numpy as np
class PropagationLayer(tf.keras.layers.Layer):
    def __init__(self, wavelength, distance, pixel_size, shape, name=None):
        super(PropagationLayer, self).__init__(name=name)
        self.wavelength = wavelength
        self.distance = distance
        self.pixel_size = pixel_size
        self.shape_ = shape
        self.h_real = None
        self.h_imag = None

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

        # Assign h_real and h_imag to instance variables
        self.h_real = tf.convert_to_tensor(h_real, dtype=tf.float32)
        self.h_imag = tf.convert_to_tensor(h_imag, dtype=tf.float32)

        # Debugging: Log kernel initialization values
        print("Kernel initialization debugging:")
        tf.print("h_real min:", tf.reduce_min(self.h_real), "max:", tf.reduce_max(self.h_real))
        tf.print("h_imag min:", tf.reduce_min(self.h_imag), "max:", tf.reduce_max(self.h_imag))

        # # Replace NaN or Inf values in kernel initialization
        # h_real = tf.where(tf.math.is_nan(h_real) | tf.math.is_inf(h_real), tf.zeros_like(h_real), h_real)
        # h_imag = tf.where(tf.math.is_nan(h_imag) | tf.math.is_inf(h_imag), tf.zeros_like(h_imag), h_imag)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]


        # Debugging: Log kernel values
        tf.print("h_real min:", tf.reduce_min(self.h_real), "max:", tf.reduce_max(self.h_real))
        tf.print("h_imag min:", tf.reduce_min(self.h_imag), "max:", tf.reduce_max(self.h_imag))

        # Replace NaN or Inf values in kernel
        h_real = tf.where(tf.math.is_nan(self.h_real) | tf.math.is_inf(self.h_real), tf.zeros_like(self.h_real), self.h_real)
        h_imag = tf.where(tf.math.is_nan(self.h_imag) | tf.math.is_inf(self.h_imag), tf.zeros_like(self.h_imag), self.h_imag)

        # Wrap around tensors with a matrix of zeros (2x the size of shape)
        padded_shape = [dim * 2 for dim in self.shape_]
        print("padded_shape:", padded_shape)
        re_u = tf.pad(re_u, [[0, 0], [padded_shape[0] // 4, padded_shape[0] // 4], [padded_shape[1] // 4, padded_shape[1] // 4]], mode='CONSTANT')
        im_u = tf.pad(im_u, [[0, 0], [padded_shape[0] // 4, padded_shape[0] // 4], [padded_shape[1] // 4, padded_shape[1] // 4]], mode='CONSTANT')

        h_real = tf.pad(h_real, [[padded_shape[0] // 4, padded_shape[0] // 4], [padded_shape[1] // 4, padded_shape[1] // 4]], mode='CONSTANT')
        h_imag = tf.pad(h_imag, [[padded_shape[0] // 4, padded_shape[0] // 4], [padded_shape[1] // 4, padded_shape[1] // 4]], mode='CONSTANT')

        # Expand dimensions for convolution
        h_real = tf.expand_dims(h_real, axis=-1)  # Shape: [H, W, 1]
        h_real = tf.expand_dims(h_real, axis=-1)       # Shape: [H, W, 1, 1]
        h_imag = tf.expand_dims(h_imag, axis=-1)  # Shape: [H, W, 1]
        h_imag = tf.expand_dims(h_imag, axis=-1)       # Shape: [H, W, 1, 1]
        print("h_real shape:", h_real.shape)
        print("h_imag shape:", h_imag.shape)

        # Debugging: Log tensor shapes before convolution
        print("re_u shape:", re_u.shape)
        print("im_u shape:", im_u.shape)
        print("h_real shape:", h_real.shape)
        print("h_imag shape:", h_imag.shape)

        # Debugging: Check for NaN or Inf values in tensors
        if tf.reduce_any(tf.math.is_nan(re_u)) or tf.reduce_any(tf.math.is_inf(re_u)):
            print("NaN or Inf detected in re_u")
        if tf.reduce_any(tf.math.is_nan(im_u)) or tf.reduce_any(tf.math.is_inf(im_u)):
            print("NaN or Inf detected in im_u")
        if tf.reduce_any(tf.math.is_nan(h_real)) or tf.reduce_any(tf.math.is_inf(h_real)):
            print("NaN or Inf detected in h_real")
        if tf.reduce_any(tf.math.is_nan(h_imag)) or tf.reduce_any(tf.math.is_inf(h_imag)):
            print("NaN or Inf detected in h_imag")

        # Debugging: Log input tensor statistics
        print("Debugging input tensors:")
        tf.print("re_u min:", tf.reduce_min(re_u), "max:", tf.reduce_max(re_u))
        tf.print("im_u min:", tf.reduce_min(im_u), "max:", tf.reduce_max(im_u))

        # Debugging: Log kernel statistics
        print("Debugging kernels:")
        tf.print("h_real min:", tf.reduce_min(h_real), "max:", tf.reduce_max(h_real))
        tf.print("h_imag min:", tf.reduce_min(h_imag), "max:", tf.reduce_max(h_imag))

        # Replace NaN or Inf values with zeros
        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
        h_real = tf.where(tf.math.is_nan(h_real) | tf.math.is_inf(h_real), tf.zeros_like(h_real), h_real)
        h_imag = tf.where(tf.math.is_nan(h_imag) | tf.math.is_inf(h_imag), tf.zeros_like(h_imag), h_imag)

        # Clamp tensor values to prevent numerical instability
        re_u = tf.clip_by_value(re_u, -1e6, 1e6)
        im_u = tf.clip_by_value(im_u, -1e6, 1e6)
        h_real = tf.clip_by_value(h_real, -1e6, 1e6)
        h_imag = tf.clip_by_value(h_imag, -1e6, 1e6)

        # Debugging: Log tensor values after clamping
        tf.print("After clamping - re_u min:", tf.reduce_min(re_u), "max:", tf.reduce_max(re_u))
        tf.print("After clamping - im_u min:", tf.reduce_min(im_u), "max:", tf.reduce_max(im_u))
        tf.print("After clamping - h_real min:", tf.reduce_min(h_real), "max:", tf.reduce_max(h_real))
        tf.print("After clamping - h_imag min:", tf.reduce_min(h_imag), "max:", tf.reduce_max(h_imag))

        re_u = re_u / tf.reduce_max(tf.abs(re_u))
        im_u = im_u / tf.reduce_max(tf.abs(im_u))
        h_real = h_real / tf.reduce_max(tf.abs(h_real))
        h_imag = h_imag / tf.reduce_max(tf.abs(h_imag))

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

        # Remove the outer zeroes added before
        crop_start = padded_shape[0] // 4
        crop_end = padded_shape[0] - crop_start
        print("crop_start:", crop_start)
        print("crop_end:", crop_end)
        out_real = out_real[:, crop_start:crop_end, crop_start:crop_end]
        out_imag = out_imag[:, crop_start:crop_end, crop_start:crop_end]
        print("out_real shape:", out_real.shape)
        print("out_imag shape:", out_imag.shape)

        return tf.stack([out_real, out_imag], axis=-1)