"""Module implementing simple baseline models."""
import numpy as np
import tensorflow as tf

from .neural_networks import MLP
from .model import SavableModel


class MLPModel(SavableModel):
    def __init__(self, n_classes, arch):
        """Initialize autoencoder model."""
        self.mlp = MLP(arch + [n_classes])
        super().__init__(locals(), [self.mlp])

    def loss(self, inputs, targets):
        prediction = self.predict(inputs)
        return tf.losses.softmax_cross_entropy(targets, prediction)

    def predict(self, inputs):
        inputs = tf.layers.flatten(inputs)
        prediction = self.mlp(inputs)
        return prediction

    @classmethod
    def _restore(cls, parameters):
        return cls(**parameters)
