import tensorflow as tf
from DiffractiveMaskLayer import DiffractiveMaskLayer
from PropagationLayer import PropagationLayer

class CDNNModel(tf.keras.Model):

    def __init__(self, num_layers, shape, wavelength_min, wavelength_max, wavelength_step, distance_between_layers, distance_to_plane, pixel_size, dwl=300e9, name=None):
        super(CDNNModel, self).__init__(name=name)
        self.shape_ = shape
        self.dwl = dwl
        self.doe_layers = []
        self.prop_layers = []
        
        # Create intermediate DOE and propagation layers
        for i in range(num_layers - 1):
            if i == 0:
                init_value = '200mm'  # First layer initialized randomly
            else:
                init_value = '200mm'
            print(f"Layer {i + 1}: DOE + Propagation z={distance_between_layers} m init={init_value}")
            self.doe_layers.append(DiffractiveMaskLayer(shape, dwl=dwl, name=f"doe_{i + 1}", init=init_value))
            self.prop_layers.append(PropagationLayer(
                wavelength_min, wavelength_max, wavelength_step, distance_between_layers, pixel_size, shape, name=f"prop_{i + 1}"
            ))

        init_value = '200mm'
        # Create final DOE and propagation layer (to target)
        self.doe_layers.append(DiffractiveMaskLayer(shape, dwl=dwl, name=f"doe_{num_layers}", init=init_value))
        self.prop_layers.append(PropagationLayer(wavelength_min, wavelength_max, wavelength_step, distance_to_plane, pixel_size, shape, name=f"prop_{num_layers}"))
        print(f"Final Layer: DOE + Propagation z={distance_to_plane} m")


    def call(self, inputs):
        field = inputs
        # Propagate through all DOE and propagation layers
        for i, (doe, prop) in enumerate(zip(self.doe_layers, self.prop_layers)):
            field = doe(field)
            field = prop(field)
        self.last_power_loss = self.prop_layers[-1].power_loss  # shape: (batch,)
        # Store power losses from all propagation layers
        self.all_power_losses = [layer.power_loss for layer in self.prop_layers]  # list of (batch,) tensors
        # Extract real and imaginary parts
        U_real = field[..., 0]
        U_imag = field[..., 1]
        intensity = tf.square(U_real) + tf.square(U_imag)
        amplitude = tf.sqrt(intensity)
        # amplitude = amplitude / tf.reduce_max(amplitude)
        return amplitude