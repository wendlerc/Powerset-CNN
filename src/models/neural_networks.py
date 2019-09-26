"""Module containing neural networks used in models.

Here we define basic architectures such as MLP, ResNet, ConvNets etc.
"""
import abc

import numpy as np
import tensorflow as tf

from .utils import training_placeholder


class SaveableNN(abc.ABC):
    """Abstract Base Class which allows NN to be saved."""

    @property
    @abc.abstractmethod
    def variables(self):
        """Get variables used in this NN."""
        raise NotImplementedError


class MLP(SaveableNN):
    """Multi Layer Perceptron."""

    def __init__(self, model_arch, activation_fn=tf.nn.relu,
                 use_biases=True, last_linear=True, name='SimpleNN', **kwargs):
        """Construct Multi layer perceptron.

        Args:
            model_arch: list of integers defining the width of the individual
                        layers
            activation_fn: default relu, otherwise function provided
            use_biases: Use biases for individual layers
            last_linear: If the last layer should have a linear activation
                         function
            name: Namespace of associated variables

        """
        self._model_arch = model_arch
        self._act_fn = activation_fn
        self._use_biases = use_biases
        self._last_linear = last_linear
        self._additional_parameters = kwargs
        self.name = name
        self.layers = self._build_layers()

    def _build_single_layer(self, layer_definition, name, **kwargs):
        # Try to convert layer definition to int
        # If this works then we treat it as a regular dense layer
        try:
            layer_definition = int(layer_definition)
        except ValueError:
            pass

        if isinstance(layer_definition, (int, np.integer)):
            # int is interpreted as a dense layer with int units
            # Initialize with class default
            layer_settings = {
                'activation': self._act_fn,
                'use_bias': self._use_biases
            }
            layer_settings.update(self._additional_parameters)
            # Allow to override using kwargs
            layer_settings.update(kwargs)
            return tf.layers.Dense(
                layer_definition, name=name, **layer_settings
            )
        elif isinstance(layer_definition, (str)):
            if layer_definition == 'bn':
                # Batchnorm layer
                # TODO: Think about which axis we should use for computing
                # statistics (should mask invalid observations).
                return tf.layers.BatchNormalization()
            elif layer_definition.startswith('d'):
                # Dropout layer d0.1 would correspond to 10% dropout
                return tf.layers.Dropout(float(layer_definition[1:]))
        else:
            raise ValueError('Layer definition has to be of integer type')

    def _get_layer_name(self, n_layer):
        return f'{self.name}/layer_{n_layer}'

    def _build_layers(self):
        if self._last_linear:
            last_activation = None
        else:
            last_activation = self._act_fn

        # Build model architecture
        layers = []
        for i, l in enumerate(self._model_arch[:-1]):
            layer = self._build_single_layer(l, self._get_layer_name(i))
            layers.append(layer)

        # Last layer could be linear
        last_layer_index = len(self._model_arch) - 1
        last_layer = self._build_single_layer(
            self._model_arch[-1],
            name=self._get_layer_name(last_layer_index),
            activation=last_activation)
        layers.append(last_layer)
        return layers

    def _apply_layers_to_tensor(self, tensor, **kwargs):
        previous_output = tensor
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (tf.layers.BatchNormalization,
                                  tf.layers.Dropout)):
                # The BatchNorm and Dropout layers require the additional
                # definition of a training mode when being called.
                previous_output = layer(
                    previous_output,
                    training=training_placeholder(),
                    **kwargs
                )
            else:
                previous_output = layer(previous_output, **kwargs)

        return previous_output

    def __call__(self, inputs):
        """Apply MLP to inputs."""
        transformed_input = self._apply_layers_to_tensor(inputs)
        return transformed_input

    @property
    def variables(self):
        """Get variables of MLP."""
        all_variables = []
        for layer in self.layers:
            all_variables.extend(layer.variables)
        return all_variables

