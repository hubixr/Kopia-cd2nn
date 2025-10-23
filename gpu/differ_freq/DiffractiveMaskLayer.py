import tensorflow as tf
import numpy as np
from PIL import Image

phase_mask_path = "validation_data_lenses/phase_mask/lens_px_0.9mm_size_128_frequency180GHz_f_200mm.bmp"  

# Currently, the phase is initialized randomly, but it can be replaced with a constant value.
def load_bmp_as_input(file_path, target_shape):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    image = image.resize(target_shape, Image.Resampling.LANCZOS)  # Resize to target shape using LANCZOS
    image_array = np.array(image, dtype=np.float32)  # Convert to numpy array
    print("image_array shape before normalization:", image_array.shape)
    print("image_array min before normalization:", np.min(image_array))
    print("image_array max before normalization:", np.max(image_array))
    image_array = (image_array / 255.0) * 2 * np.pi  # Normalize to 0-2π
    # image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    print("image_array shape:", image_array.shape)
    print("image_array min:", np.min(image_array))
    print("image_array max:", np.max(image_array))
    return image_array

class DiffractiveMaskLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None, init='zero'):
        super(DiffractiveMaskLayer, self).__init__(name=name)
        self.shape_ = shape
        self.init_ = init

    def build(self, input_shape):
        # Initialize phase as a trainable weight
        phase_mask = load_bmp_as_input(phase_mask_path, self.shape_)
        if self.init_ == 'random_full':
            initializer = tf.keras.initializers.RandomUniform(0.0, 2 * np.pi, seed=42)
        elif self.init_ == 'random_specified':
            # Initialize phase with random values sampled from a given list
            phase_values = [np.pi/2, np.pi,3*np.pi/2, 2*np.pi]
            random_indices = np.random.randint(0, len(phase_values), size=self.shape_)
            random_phases = np.array([phase_values[idx] for idx in random_indices.flat], dtype=np.float32).reshape(self.shape_)
            initializer = tf.keras.initializers.Constant(random_phases)
        elif self.init_ == 'zero':
            initializer = tf.keras.initializers.Constant(0.0)
        elif self.init_ == 'pi':
            initializer = tf.keras.initializers.Constant(np.pi)
        else:
            raise ValueError(f"Unknown init: {self.init_}")
        
        self.phase = self.add_weight(
            name="phase",
            shape=self.shape_,
            initializer=initializer,
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True
        )
        super(DiffractiveMaskLayer, self).build(input_shape)
    
    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float16)  # Cast inputs to float16
        # print("Doe call")
        re_u = inputs[..., 0]  # Real part
        im_u = inputs[..., 1]  # Imaginary part
        wavelength = inputs[..., 2]  # Wavelengths

        re_u = tf.where(tf.math.is_nan(re_u) | tf.math.is_inf(re_u), tf.zeros_like(re_u), re_u)
        im_u = tf.where(tf.math.is_nan(im_u) | tf.math.is_inf(im_u), tf.zeros_like(im_u), im_u)
    
        # Ensure phase is properly managed by TensorFlow
        phase = tf.cast(tf.identity(self.phase), tf.float16)  # Cast phase to float16
        if im_u is None:
            print("im_u is None")
            out_real = re_u * tf.cos(phase)
            out_imag = re_u * tf.sin(phase)
        else:
            print("im_u is not None")
            out_real = re_u * tf.cos(phase) - im_u * tf.sin(phase)
            out_imag = re_u * tf.sin(phase) + im_u * tf.cos(phase)
        print("end doe call")
        out_real = out_real/tf.reduce_max(tf.abs(out_real))
        # tf.print("out_real stats:", tf.reduce_min(out_real), tf.reduce_max(out_real), tf.reduce_mean(out_real))
        epsilon = 1e-14  # Small constant to avoid division by zero
        max_imag = tf.reduce_max(tf.abs(out_imag))
        out_imag = tf.cond(
            max_imag > epsilon,
            lambda: out_imag / (max_imag + epsilon),
            lambda: out_imag  # If max_imag is zero, leave out_imag unchanged
        )
        return tf.stack([out_real, out_imag, wavelength], axis=-1)  # [B, H, W, 3]