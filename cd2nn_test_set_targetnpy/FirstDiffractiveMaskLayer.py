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
Input: tensor [B, H, W, 1] (U).
Output: tensor [B, H, W, 2] (Re, Im).
"""
# Currently, the phase is initialized randomly, but it can be replaced with a constant value.
# Assuming firt layer as takes only Amplitude/Intensity as input.

class FirstDiffractiveMaskLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(FirstDiffractiveMaskLayer, self).__init__(name=name)
        self.shape_ = shape

    def build(self, input_shape):
        # Trenowalna mapa fazowa – reprezentuje DOE
        self.phase = self.add_weight(
        name="phase",
        shape=self.shape_,
        initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2 * np.pi),
        trainable=True
        )
        super(FirstDiffractiveMaskLayer, self).build(input_shape)   

    def call(self, inputs):
        print("first doe call")
        """
        inputs: tf.Tensor [B, H, W, 2] — input complex field (Re, Im)
        returns: tf.Tensor [B, H, W, 2] — complex field after phase modulation
        """
        # Extract real and imaginary parts
        re_u = inputs[..., 0]  # Real part
        im_u = inputs[..., 1]  # Imaginary part

        """# Combine into a complex tensor
        U = tf.complex(re_u, im_u)  # Shape: [B, H, W]"""

        # Broadcast phase to match input dimensions
        phase = tf.expand_dims(self.phase, axis=0)  # Shape: [1, H, W]
        # phase = tf.broadcast_to(phase, tf.shape(U))  # Shape: [B, H, W]

        # Apply phase modulation
        # out = U * tf.exp(tf.complex(0.0, phase))  # Modulation of the input field by the phase mask
        out_real = re_u * tf.cos(phase) - im_u * tf.sin(phase)
        out_imag = re_u * tf.sin(phase) + im_u * tf.cos(phase)

        # Separate real and imaginary parts
        # out_real = tf.math.real(out)
        # out_imag = tf.math.imag(out)
        print("end first doe call")
        return tf.stack([out_real, out_imag], axis=-1)  # Shape: [B, H, W, 2]