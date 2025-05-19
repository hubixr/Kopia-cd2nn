import tensorflow as tf
from DiffractiveMaskLayer import DiffractiveMaskLayer
from FirstDiffractiveMaskLayer import FirstDiffractiveMaskLayer
from PropagationLayer import PropagationLayer
import logging
logging.getLogger('tensorflow').addFilter(lambda record: '+ptx85' not in record.getMessage())

class CDNNModel(tf.keras.Model):

    def __init__(self, num_layers, shape, wavelength, distance_between_layers, distance_to_plane, pixel_size, name=None):
        super(CDNNModel, self).__init__(name=name)
        self.shape_ = shape
        self.doe_layers = []
        self.prop_layers = []

        for i in range(num_layers-1):
            self.doe_layers.append(DiffractiveMaskLayer(shape,name=f"doe_{i+1}"))
            self.prop_layers.append(PropagationLayer(wavelength, distance_between_layers, pixel_size, shape, name=f"prop_{i+1}"))
            print(f"Layer {i+1}: DOE + Propagation z={distance_between_layers} m")
            
        self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{num_layers}"))
        self.prop_layers.append(PropagationLayer(wavelength, distance_to_plane, pixel_size, shape, name=f"prop_{num_layers}"))
        print(f"Final Layer: DOE + Propagation z={distance_to_plane} m")

    def sanitize_tensor(self, tensor, name):
        """
        Replaces NaN or Inf values in the tensor with zeros and logs the issue.
        """
        if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
            tf.print(f"Warning: NaN or Inf detected in {name}. Replacing with zeros.")
            tensor = tf.where(tf.math.is_nan(tensor) | tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
        return tensor

    def call(self, inputs):
        """
        inputs: tensor [B, H, W, 2] — zespolone pole wejściowe (Re, Im)
        returns: tensor [B, H, W] — amplituda pola po przejściu przez całą strukturę
        """
        field = inputs
        field = self.sanitize_tensor(field, "inputs")
        for i, (doe, prop) in enumerate(zip(self.doe_layers, self.prop_layers)):
            field = doe(field)
            field = self.sanitize_tensor(field, f"field after DOE Layer {i+1}")

            field = prop(field)
            field = self.sanitize_tensor(field, f"field after Propagation Layer {i+1}")

        intensity = tf.reduce_sum(tf.square(field), axis=-1)  # Amplitude
        normalized_intensity = intensity / tf.reduce_max(intensity)  # Normalization to 0-1
        return normalized_intensity