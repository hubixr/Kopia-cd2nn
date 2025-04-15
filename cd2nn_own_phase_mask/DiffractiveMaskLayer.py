import tensorflow as tf
import numpy as np
from PIL import Image
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
phase_mask_path = "validation_data_lenses/phase_mask/lens_px_0.9mm_size_128_frequency300GHz_f_100mm.bmp"  

# Currently, the phase is initialized randomly, but it can be replaced with a constant value.
def load_bmp_as_input(file_path, target_shape):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    image = image.resize(target_shape, Image.Resampling.LANCZOS)  # Resize to target shape using LANCZOS
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    print("image_array shape before normalization:", image_array.shape)
    print("image_array min before normalization:", np.min(image_array))
    print("image_array max before normalization:", np.max(image_array))
    image_array = (image_array / 255.0) * 2 * np.pi  # Normalize to 0-2π
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    print("image_array shape:", image_array.shape)
    print("image_array min:", np.min(image_array))
    print("image_array max:", np.max(image_array))
    return image_array

class DiffractiveMaskLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(DiffractiveMaskLayer, self).__init__(name=name)
        self.shape_ = shape

    def build(self, input_shape):
        # Initialize phase as a trainable weight
        # Load phase mask
        phase_mask = load_bmp_as_input(phase_mask_path, self.shape_)  # Preprocess to match model input shape
        self.phase = self.add_weight(
            name="phase",
            shape=self.shape_,
            initializer=tf.keras.initializers.Constant(phase_mask),
            trainable=False
        )
        super(DiffractiveMaskLayer, self).build(input_shape)
    
    # Add explicit casting to float16 for phase and inputs
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)  # Cast inputs to float16
        print("Doe call")
        re_u = inputs[..., 0]  # Real part
        im_u = inputs[..., 1]  # Imaginary part

        # Ensure phase is properly managed by TensorFlow
        phase = tf.cast(tf.identity(self.phase), tf.float16)  # Cast phase to float16

        # Apply phase modulation
        if im_u is None:
            out_real = re_u * tf.cos(phase)
            out_imag = re_u * tf.sin(phase)
        else:
            out_real = re_u * tf.cos(phase) - im_u * tf.sin(phase)
            out_imag = re_u * tf.sin(phase) + im_u * tf.cos(phase)
        print("end doe call")
        return tf.stack([out_real, out_imag], axis=-1)  # [B, H, W, 2]