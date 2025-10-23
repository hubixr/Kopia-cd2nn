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
        tf.print("Power before:", tf.reduce_sum(power_before))
        size = int(re_u.shape[1]/2)
        re_u = tf.expand_dims(re_u, axis=-1)
        im_u = tf.expand_dims(im_u, axis=-1)
        re_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(re_u)
        im_u = tf.keras.layers.ZeroPadding2D(padding=(size, size))(im_u)

        #Delete last dimension
        re_u = tf.squeeze(re_u, axis=-1)
        im_u = tf.squeeze(im_u, axis=-1)

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
        print("Number of pixels in re_re:", tf.size(re_u).numpy())
        N = tf.cast(tf.shape(re_u)[1] * tf.shape(re_u)[2], tf.float32)  # szer. * wys.
        h_real_frequency = tf.cast(self.h_real[None, :, :self.shape_[1] + 1], tf.complex64)
        h_imag_frequency = tf.cast(self.h_imag[None, :, :self.shape_[1] + 1], tf.complex64)

        # Plot h_real in frequency domain
        plt.figure(figsize=(8, 6))
        plt.imshow(h_real_frequency[0].numpy().real, cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='h_real Frequency')
        plt.title('h_real Frequency Domain Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('h_real_frequency_plot.png')
        plt.close()

        # Plot h_imag in frequency domain
        plt.figure(figsize=(8, 6))
        plt.imshow(h_imag_frequency[0].numpy().real, cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='h_imag Frequency')
        plt.title('h_imag Frequency Domain Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('h_imag_frequency_plot.png')
        plt.close()
        re_re = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_real_frequency) /tf.sqrt(N)
        im_im = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_imag_frequency) /tf.sqrt(N)
        re_im = tf.signal.irfft2d(tf.signal.rfft2d(re_u) * h_imag_frequency) /tf.sqrt(N)
        im_re = tf.signal.irfft2d(tf.signal.rfft2d(im_u) * h_real_frequency) /tf.sqrt(N)

        # Plot re_re
        plt.figure(figsize=(8, 6))
        plt.imshow(re_re[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='re_re')
        plt.title('re_re Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('re_re_plot.png')
        plt.close()

        # Plot im_im
        plt.figure(figsize=(8, 6))
        plt.imshow(im_im[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='im_im')
        plt.title('im_im Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('im_im_plot.png')
        plt.close()

        # Plot re_im
        plt.figure(figsize=(8, 6))
        plt.imshow(re_im[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='re_im')
        plt.title('re_im Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('re_im_plot.png')
        plt.close()

        # Plot im_re
        plt.figure(figsize=(8, 6))
        plt.imshow(im_re[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='im_re')
        plt.title('im_re Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('im_re_plot.png')
        plt.close()
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
        out_real = tf.squeeze(out_real, axis=-1)
        out_imag = tf.squeeze(out_imag, axis=-1)
        intensity = out_real**2 + out_imag**2
        plt.figure(figsize=(8, 6))
        plt.imshow(intensity[0], cmap='hot')  # Assuming batch size is 1
        plt.colorbar(label='Intensity')
        plt.title('Intensity Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig('intensity_plot_after.png')
        plt.close()
        print("output shape:", tf.stack([out_real, out_imag], axis=-1).shape)
        tf.print("Power after resizing:", tf.reduce_sum(tf.reduce_sum(out_real**2 + out_imag**2)))

        return tf.stack([out_real, out_imag], axis=-1)
