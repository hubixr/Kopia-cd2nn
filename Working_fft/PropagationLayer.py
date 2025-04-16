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
        inputs = tf.cast(inputs, tf.float32)  # Cast inputs to float32
        re_u = inputs[..., 0]
        im_u = inputs[..., 1]

        print("first conv")
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_real))
        print("first conv shape", re_re.shape)
        print("second conv")
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_imag))
        print("third conv")
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_imag))
        print("fourth conv")
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_real))

        # Do not apply ifftshift to the outputs
        print("end of conv")

        # Remove tf.squeeze as it is not needed for tensors without singleton dimensions
        out_real = re_re - im_im
        out_imag = re_im + im_re

        # Normalize the outputs
        out_real = out_real / tf.reduce_max(out_real)
        out_imag = out_imag / tf.reduce_max(out_imag)

        print("min and max of out_real", tf.reduce_min(out_real), tf.reduce_max(out_real))
        print("min and max of out_imag", tf.reduce_min(out_imag), tf.reduce_max(out_imag))
        return tf.stack([out_real, out_imag], axis=-1)