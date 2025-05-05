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
        x = np.arange(-W, W) * dx
        y = np.arange(-H, H) * dx
        X, Y = np.meshgrid(x, y)

        r2 = X**2 + Y**2
        k = 2 * np.pi / self.wavelength
        arg = k * self.distance + (np.pi * r2) / (self.wavelength * self.distance)
        h = 1/(1j * self.wavelength * self.distance) * np.exp(1j * arg)
        h_real = np.real(h)
        h_imag = np.imag(h)

        # Apply fftshift to the kernel during initialization
        h_real = np.fft.fftshift(h_real)
        h_imag = np.fft.fftshift(h_imag)

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
        print("re_u shape:", re_u.shape)
        print("im_u shape:", im_u.shape)
        size = int(re_u.shape[1]/2)
        print("Size of input:", size)
        # #Add zero padding
        re_u = tf.expand_dims(re_u, axis=-1)
        im_u = tf.expand_dims(im_u, axis=-1)
        print("re_u shape after expand_dims:", re_u.shape)
        re_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(re_u)
        im_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(im_u)
        print("re_u shape after padding:", re_u.shape)

        #Delete last dimension
        re_u = tf.squeeze(re_u, axis=-1)
        im_u = tf.squeeze(im_u, axis=-1)
        print("re_u shape after squeeze:", re_u.shape)
        print("im_u shape after squeeze:", im_u.shape)
        # Perform 4 fft convolutions
        print("Start of convolutions")
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_real))
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_imag))
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * tf.signal.rfft2d(self.h_imag))
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * tf.signal.rfft2d(self.h_real))
        print("re_re shape:", re_re.shape)
        print("End of convolutions")
        
        # Compute real and imaginary parts of the output
        out_real = re_re - im_im
        out_imag = re_im + im_re
        #Crop to original size
        out_real = tf.expand_dims(out_real, axis=-1)
        out_imag = tf.expand_dims(out_imag, axis=-1)
        out_real = tf.keras.layers.Cropping2D(cropping=(size, size))(out_real)
        out_imag = tf.keras.layers.Cropping2D(cropping=(size, size))(out_imag)
        print("out_real shape after cropping:", out_real.shape)
        print("out_imag shape after cropping:", out_imag.shape)
        out_real = tf.squeeze(out_real, axis=-1)
        out_imag = tf.squeeze(out_imag, axis=-1)
        print("out_real shape after squeeze:", out_real.shape)
        print("out_imag shape after squeeze:", out_imag.shape)
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