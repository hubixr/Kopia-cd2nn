import tensorflow as tf
from DiffractiveMaskLayer import DiffractiveMaskLayer
from PropagationLayer import PropagationLayer

class CDNNModel(tf.keras.Model):

    def __init__(self, num_layers, shape, wavelength_min, wavelength_max, wavelength_step, distance_between_layers, distance_to_plane, pixel_size, name=None):
        super(CDNNModel, self).__init__(name=name)
        self.shape_ = shape
        self.doe_layers = []
        self.prop_layers = []
          
        # Create intermediate DOE and propagation layers
        for i in range(num_layers - 1):
            if i == 0:
                init_value = 'random_specified'  # First layer initialized randomly
            else:
                init_value = 'random_specified'
            print(f"Layer {i + 1}: DOE + Propagation z={distance_between_layers} m init={init_value}")
            self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{i + 1}", init=init_value))
            self.prop_layers.append(PropagationLayer(
                wavelength_min, wavelength_max, wavelength_step, distance_between_layers, pixel_size, shape, name=f"prop_{i + 1}"
            ))

        init_value = 'random_specified'
        # Create final DOE and propagation layer (to target)
        self.doe_layers.append(DiffractiveMaskLayer(shape, name=f"doe_{num_layers}", init=init_value))
        self.prop_layers.append(PropagationLayer(wavelength_min, wavelength_max, wavelength_step, distance_to_plane, pixel_size, shape, name=f"prop_{num_layers}"))
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
            # # Split channels
            # real = field[..., 0]
            # imag = field[..., 1]
            # wavelength = field[..., 2]
            # # Apply DOE mask (should only affect real/imag)
            # real, imag, wavelength = doe(tf.stack([real, imag, wavelength], axis=-1))
            # # Re-stack with wavelength
            # field = tf.concat([real, imag, wavelength], axis=-1)
            field = doe(field)
            field = prop(field)
            
        # Store power loss from final propagation layer (for backward compatibility)
        self.last_power_loss = self.prop_layers[-1].power_loss  # shape: (batch,)
        
        # Store power losses from all propagation layers
        self.all_power_losses = [layer.power_loss for layer in self.prop_layers]  # list of (batch,) tensors
        
        # Extract real and imaginary parts
        U_real = field[..., 0]
        U_imag = field[..., 1]
        
        # Calculate intensity |U|^2
        intensity = tf.square(U_real) + tf.square(U_imag)
        
        # Debug prints
        # print(
        #     "Intensity min:", tf.reduce_min(intensity),
        #     "max:", tf.reduce_max(intensity),
        #     "mean:", tf.reduce_mean(intensity)
        # )
        
        # Calculate normalized amplitude for debugging
        amplitude = tf.sqrt(intensity)
        amplitude = amplitude / tf.reduce_max(amplitude)
        # print(
        #     "Amplitude min:", tf.reduce_min(amplitude),
        #     "max:", tf.reduce_max(amplitude),
        #     "mean:", tf.reduce_mean(amplitude)
        # )

        return intensity