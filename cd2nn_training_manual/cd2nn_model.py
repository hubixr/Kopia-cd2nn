import tensorflow as tf
from DiffractiveMaskLayer import DiffractiveMaskLayer
from PropagationLayer import PropagationLayer

class CDNNModel(tf.keras.Model):

    def __init__(self, num_layers, shape, wavelength, distance_between_layers, distance_to_plane, pixel_size, name=None):
        super(CDNNModel, self).__init__(name=name)
        self.shape_ = shape
        self.doe_layers = []
        self.prop_layers = []
        
        # Initialize DOE layers with small random values
        init_value = 'random_full'
        
        # Create intermediate DOE and propagation layers
        for i in range(num_layers - 1):
            init_value = 'random_small'
            print(f"Layer {i + 1}: DOE + Propagation z={distance_between_layers} m init={init_value}")
            self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{i + 1}", init=init_value))
            self.prop_layers.append(PropagationLayer(
                wavelength, distance_between_layers, pixel_size, shape, name=f"prop_{i + 1}"
            ))

        # Create final DOE and propagation layer (to target)
        self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{num_layers}", init=init_value))
        self.prop_layers.append(PropagationLayer(wavelength, distance_to_plane, pixel_size, shape, name=f"prop_{num_layers}"))
        print(f"Final Layer: DOE + Propagation z={distance_to_plane} m")


    def call(self, inputs):
        """
        Forward pass through the CDNN model.
        
        Args:
            inputs: Complex field tensor of shape [B, H, W, 2] (real, imaginary)
            
        Returns:
            intensity: Intensity tensor of shape [B, H, W]
        """
        field = inputs

        # Propagate through all DOE and propagation layers
        for i, (doe, prop) in enumerate(zip(self.doe_layers, self.prop_layers)):
            field = doe(field)
            field = prop(field)
            
        # Store power loss from final propagation layer
        self.last_power_loss = self.prop_layers[-1].power_loss  # shape: (batch,)
        
        # Extract real and imaginary parts
        U_real = field[..., 0]
        U_imag = field[..., 1]
        
        # Calculate intensity |U|^2
        intensity = tf.square(U_real) + tf.square(U_imag)
        
        # Debug prints
        print(
            "Intensity min:", tf.reduce_min(intensity),
            "max:", tf.reduce_max(intensity),
            "mean:", tf.reduce_mean(intensity)
        )
        
        # Calculate normalized amplitude for debugging
        amplitude = tf.sqrt(intensity)
        amplitude = amplitude / tf.reduce_max(amplitude)
        print(
            "Amplitude min:", tf.reduce_min(amplitude),
            "max:", tf.reduce_max(amplitude),
            "mean:", tf.reduce_mean(amplitude)
        )

        return intensity