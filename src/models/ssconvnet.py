"""Module implementing simple autoencoder models."""
import numpy as np
import tensorflow as tf

from .neural_networks import MLP, SaveableNN
from .model import SavableModel
from .spectral_layers import _SSConvSpectral, _SSConvSpectralElementary, _SSConvSpectralLocalized
from .pooling_layers import _SSPoolNaive, _SSPoolFourier, _SSPoolGroundSet
from .graph_layers import _HypercubeConvKipf, _HypercubeAdjacency


class PowersetConvNet(SavableModel):
    def __init__(self, model, n_classes, conv_arch, mlp_arch, use_sum=False):
        self.convnet = PowersetConvLayer(model, conv_arch, last_linear=False)
        self.mlp_arch = mlp_arch + [n_classes]
        self.mlp = MLP(self.mlp_arch)
        self.use_sum = use_sum
        print('My architecture is: ', conv_arch, mlp_arch)
        super().__init__(locals(), [self.convnet, self.mlp])

    def loss(self, inputs, targets):
        prediction = self.predict(inputs)
        return tf.losses.softmax_cross_entropy(targets, prediction)

    def predict(self, inputs):
        feature_maps = self.convnet(inputs)
        if self.use_sum:
            features = tf.reduce_sum(feature_maps, axis=1)
        else:
            features = tf.layers.flatten(feature_maps)
        prediction = self.mlp(features)
        return prediction

    @classmethod
    def _restore(cls, parameters):
        return cls(**parameters)


class PowersetConvLayer(SaveableNN):
    def __init__(self, signal_model, model_arch, activation_fn=tf.nn.relu,
                 use_biases=True, last_linear=True, name='SSCN', **kwargs):
        """Construct Multi layer perceptron.

        Args:
            model_arch: list of ints/strings defining the type and amount of filters of the individual
                        layers
            activation_fn: default relu, otherwise function provided
            use_biases: Use biases for individual layers
            last_linear: If the last layer should have a linear activation
                         function
            name: Namespace of associated variables

        """
        self._signal_model = signal_model
        self._model_arch = model_arch
        self._act_fn = activation_fn
        self._use_biases = use_biases
        self._last_linear = last_linear
        self._additional_parameters = kwargs
        self.name = name
        self.layers = self._build_layers()

    def _build_single_layer(self, layer_definition, name, **kwargs):
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
            return _SSConvSpectral(layer_definition, self._signal_model, name=name, **layer_settings)
        elif isinstance(layer_definition, (tuple, list)):
            layer_settings = {
                'activation': self._act_fn,
                'use_bias': self._use_biases
            }
            layer_settings.update(self._additional_parameters)
            layer_settings.update(kwargs)
            if layer_definition[0] == 'E':
                return _SSConvSpectralElementary(layer_definition[1], self._signal_model, name=name, **layer_settings)
            elif layer_definition[0] == 'EE':
                return _SSConvSpectralLocalized(layer_definition[1], self._signal_model, name=name, **layer_settings)
            elif layer_definition[0] == 'F':
                return _SSConvSpectral(layer_definition[1], self._signal_model, name=name, **layer_settings)
            elif layer_definition[0] == 'Kipf':
                return _HypercubeConvKipf(layer_definition[1], name=name, **layer_settings)
            elif layer_definition[0] == 'Adj':
                return _HypercubeAdjacency(layer_definition[1], name=name, **layer_settings)
            else:
                raise NotImplementedError('Unrecognized layer type.')
        elif isinstance(layer_definition, str):
            if layer_definition == 'ps':
                return _SSPoolNaive(name=name)
            elif layer_definition == 'pf':
                return _SSPoolFourier(model=self._signal_model, name=name)
            elif layer_definition == 'pm':
                return _SSPoolGroundSet(name=name)
            else:
                raise NotImplementedError('pooling method unknown')
        else:
            raise NotImplementedError('layer setup string not recognized '+str(layer_definition))

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
            if isinstance(layer, (_SSConvSpectral, _SSConvSpectralElementary, _SSPoolNaive, _SSPoolFourier,
                                  _SSPoolGroundSet, _HypercubeConvKipf, _SSConvSpectralLocalized, _HypercubeAdjacency)):
                previous_output = layer.apply(previous_output)
            else:
                raise NotImplementedError('unsupported layer type')

        return previous_output

    def __call__(self, inputs):
        transformed_input = self._apply_layers_to_tensor(inputs)
        return transformed_input

    @property
    def variables(self):
        """Get variables of MLP."""
        all_variables = []
        for layer in self.layers:
            all_variables.extend(layer.variables)
        return all_variables





