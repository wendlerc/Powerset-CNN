import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base


def hypercube_adjacency(N):
    if N == 1:
        return np.asarray([[0, 1], [1, 0]])
    A = np.zeros((2**N, 2**N))
    A_ = hypercube_adjacency(N - 1)
    I = np.eye(2**(N-1))
    A[:2**(N-1)][:, :2**(N-1)] = A_
    A[2**(N-1):][:, 2**(N-1):] = A_
    A[2**(N-1):][:, :2**(N-1)] = I
    A[:2**(N-1)][:, 2**(N-1):] = I
    return A


class _HypercubeConvKipf(base.Layer):
    """Kipf approximate Laplacian Hypercube Graph Convolutional Layer.
    """

    def __init__(self, n_filters, activation=None, kernel_initializer=None,
                 name=None, trainable=True, use_bias=False, **kwargs):
        super(_HypercubeConvKipf, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_filters = n_filters
        self.activation = activation
        self.use_bias = use_bias
        self.L_tilde = None
        self.Phi = None
        self.bias = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1]
        self.n_channels = input_shape[2]
        A = hypercube_adjacency(int(np.log2(self.n_subsets.value)))
        A_tilde = A + np.eye(A.shape[0])
        D_inv = np.diag(1 / np.sum(A_tilde, axis=1))
        D_inv_sqrt = np.sqrt(D_inv)
        self.L_tilde = np.dot(np.dot(D_inv_sqrt, A_tilde), D_inv_sqrt)
        self.L_tilde = tf.convert_to_tensor(self.L_tilde)
        self.L_tilde = tf.cast(self.L_tilde, self.dtype)
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            self.Phi = self.add_variable('Phi', shape=[self.n_channels, self.n_filters],
                                         dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=self.dtype, trainable=True,
                                              initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters))*0.01))
        super(_HypercubeConvKipf, self).build(input_shape)

    def call(self, inputs):
        X = inputs
        L = self.L_tilde
        Phi = self.Phi
        LX = tf.tensordot(L, X, axes=[[1], [1]])
        LX = tf.transpose(LX, [1, 0, 2])
        Z = tf.tensordot(LX, Phi, axes=[[2], [0]])
        if self.use_bias:
            Z += self.bias
        if self.activation is not None:
            return self.activation(Z)
        return Z

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])


class _HypercubeAdjacency(base.Layer):
    """Hypergraph Adjacency Shift Based Convolutional Layer.
    """

    def __init__(self, n_filters, activation=None, kernel_initializer=None,
                 name=None, trainable=True, use_bias=False, **kwargs):
        super(_HypercubeAdjacency, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_filters = n_filters
        self.activation = activation
        self.use_bias = use_bias
        self.A = None
        self.Phi0 = None #weight of identity
        self.Phi1 = None #weight of shifted version
        self.bias = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1]
        self.n_channels = input_shape[2]
        N = int(np.log2(self.n_subsets.value))
        self.A = (1/N)*hypercube_adjacency(N) #1/lambda_max for normalization purposes
        self.A = tf.cast(self.A, self.dtype)
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            self.Phi0 = self.add_variable('Phi0', shape=[self.n_channels, self.n_filters],
                                         dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            self.Phi1 = self.add_variable('Phi1', shape=[self.n_channels, self.n_filters],
                                         dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=self.dtype, trainable=True,
                                              initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters))*0.01))
        super(_HypercubeAdjacency, self).build(input_shape)

    def call(self, inputs):
        X = inputs
        AX = tf.tensordot(self.A, X, axes=[[1], [1]])
        AX = tf.transpose(AX, [1, 0, 2])
        Z0 = tf.tensordot(X, self.Phi0, axes=[[2], [0]])
        Z1 = tf.tensordot(AX, self.Phi1, axes=[[2], [0]])
        Z = Z0 + Z1
        if self.use_bias:
            Z += self.bias
        if self.activation is not None:
            return self.activation(Z)
        return Z

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])
