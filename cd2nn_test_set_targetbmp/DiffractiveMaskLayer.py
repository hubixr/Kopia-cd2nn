import tensorflow as tf
import numpy as np

"""
Assumptions: 
1. The trainable parameter is the phase map, which is added to the DOE transmittance.
2. Network operation: DOE -> propagation ->  detection.
3. The real amplitude value of the field must be expanded into 2 channels (Re, Im) to perform convolution.
4. The propagation layer is not trainable but is added to the model as a layer.
5. Propagation is described as a convolution with the impulse response h(x, y).
6. The amplitude values of the field are normalized to 0-1 before propagation.

Parameters:
- shape: shape of the layer (H, W) - size of the DOE layer, e.g., (128, 128), plus 2 channels responsible for amplitude Re and Im (128, 128, 2).
- wavelength: wavelength of light in meters.
- distance: distance to the detector in meters.
- pixel_size: size of a pixel in meters.

Technical details:
Input: tensor of shape [B, H, W, 2], where inputs represent U (Re, Im).
Output: tensor of shape [B, H, W, 2], where outputs[..., 0] is Re(U), and outputs[..., 1] is Im(U).

Source:
https://www.tensorflow.org/tutorials/customization/custom_layers
"""

"""
Class Assumptions: Diffractive DOE layer introducing phase delay.
Contains only the transformation of the field due to the DOE element.
Input: tensor [B, H, W, 2] (Re, Im).
Output: tensor [B, H, W, 2] (Re, Im).
"""
# Currently, the phase is initialized randomly, but it can be replaced with a constant value.

class DiffractiveMaskLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(DiffractiveMaskLayer, self).__init__(name=name)
        self.shape_ = shape

    def build(self, input_shape):
        # Initialize phase as a trainable weight
        self.phase = self.add_weight(
            name="phase",
            shape=self.shape_,
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2 * np.pi),
            trainable=True
        )
        super(DiffractiveMaskLayer, self).build(input_shape)
    
    # Add explicit casting to float16 for phase and inputs
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)  # Cast inputs to float16
        print("Doe call")
        re_u = inputs[..., 0]  # Real part
        im_u = inputs[..., 1]  # Imaginary part
        print("checking for nans and infs in diffraction layer at the beginning")
        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
    
        # Ensure phase is properly managed by TensorFlow
        phase = tf.cast(tf.identity(self.phase), tf.float16)  # Cast phase to float16

        # Apply phase modulation
        if im_u is None:
            out_real = re_u * tf.cos(phase)
            out_imag = re_u * tf.sin(phase)
        else:
            out_real = re_u * tf.cos(phase) - im_u * tf.sin(phase)
            out_imag = re_u * tf.sin(phase) + im_u * tf.cos(phase)
        print("checking for nans and infs in diffraction layer at the end")
        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
        out_real = tf.where(tf.math.is_nan(out_real) | tf.math.is_inf(out_real), tf.zeros_like(out_real), out_real)
        out_imag = tf.where(tf.math.is_nan(out_imag) | tf.math.is_inf(out_imag), tf.zeros_like(out_imag), out_imag)
        print("end doe call")
        return tf.stack([out_real, out_imag], axis=-1)  # [B, H, W, 2]