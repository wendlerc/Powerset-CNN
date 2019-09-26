import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
from .spectral_layers import FourierTransform


class _SSPoolNaive(base.Layer):
    """Subset pooling layer.
    """
    def __init__(self, aggregation_operator=tf.maximum, name=None, **kwargs):
        super(_SSPoolNaive, self).__init__(name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_vertices = None
        self.aggr = aggregation_operator

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_channels = input_shape[2].value
        self.n_vertices = int(np.log2(self.n_subsets))
        super(_SSPoolNaive, self).build(input_shape)

    def call(self, inputs):
        first = inputs[:, :self.n_subsets//2]
        second = inputs[:, self.n_subsets//2:]
        first = tf.reshape(first, [-1, (self.n_subsets // 2) * self.n_channels])
        second = tf.reshape(second, [-1, (self.n_subsets // 2) * self.n_channels])
        return tf.reshape(self.aggr(first, second), [-1, self.n_subsets // 2, self.n_channels])

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1]//2, input_shape[2]])


def pool(inputs, aggregation_operator=tf.maximum, name=None):
    layer = _SSPoolNaive(aggregation_operator=aggregation_operator, name=name)
    return layer.apply(inputs)


class _SSPoolFourier(base.Layer):
    """Subset pooling layer.
    """
    def __init__(self, model=5, name=None, **kwargs):
        super(_SSPoolFourier, self).__init__(name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_vertices = None
        self.model = model
        self.ft_before = None
        self.ft_after = None

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_channels = input_shape[2].value
        self.n_vertices = int(np.log2(self.n_subsets))
        self.ft_before = FourierTransform(self.n_vertices, self.model)
        self.ft_after = FourierTransform(self.n_vertices - 1, self.model)
        super(_SSPoolFourier, self).build(input_shape)

    def call(self, inputs):
        inputs_hat = self.ft_before.fft(inputs)
        first = inputs_hat[:, :self.n_subsets // 2]
        second = inputs_hat[:, self.n_subsets // 2:]
        inputs_hat_sampled = first + second
        return self.ft_after.ifft(inputs_hat_sampled)

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1]//2, input_shape[2]])


def poolf(inputs, signal_model=5, name=None):
    layer = _SSPoolFourier(model=signal_model, name=name)
    return layer.apply(inputs)


class _SSPoolGroundSet(base.Layer):
    """Subset pooling layer.
    """
    def __init__(self, name=None, **kwargs):
        super(_SSPoolGroundSet, self).__init__(name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_vertices = None
        self.indices = None

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_channels = input_shape[2].value
        self.n_vertices = int(np.log2(self.n_subsets))
        subsets = np.arange(self.n_subsets)
        combined = 2**0 + 2**1 #combine first and second element in ground set
        self.indices = np.where((subsets & combined == combined) | (subsets & combined == 0))[0]
        super(_SSPoolGroundSet, self).build(input_shape)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1]//2, input_shape[2]])


def poolm(inputs, name=None):
    layer = _SSPoolGroundSet(name=name)
    return layer.apply(inputs)





