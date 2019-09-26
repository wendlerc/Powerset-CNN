import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base
from scipy.sparse import coo_matrix
import itertools
from .. import dssp


class FourierTransform:
    F_kernels = {1: np.asarray([[1, 1], [1, 0]]), 2: np.asarray([[1, 1], [0, 1]]),
                  3: np.asarray([[1, 0], [1, -1]]), 4: np.asarray([[0, 1], [1, -1]]),
                  5: np.asarray([[1, 1], [1, -1]])}
    iF_kernels = {1: np.asarray([[0, 1], [1, -1]]), 2: np.asarray([[1, -1], [0, 1]]),
                  3: np.asarray([[1, 0], [1, -1]]), 4: np.asarray([[1, 1], [1, 0]]),
                  5: np.asarray([[1, 1], [1, -1]])}

    def __init__(self, n, model, F_kernel = None, iF_kernel = None, dtype=np.float32):
        self.n = n
        if F_kernel is None:
            F_kernel = FourierTransform.F_kernels[model].copy()
        if iF_kernel is None:
            iF_kernel = FourierTransform.iF_kernels[model].copy()
        self.F = F_kernel.copy()
        self.iF = iF_kernel.copy()

        for i in range(n-1):
            self.F = np.kron(self.F, F_kernel)
            self.iF = np.kron(self.iF, iF_kernel)
        if model == 5:
            self.F = self.F.astype(dtype)
            self.iF = self.F.astype(dtype)
            self.F *= (1 / 2) ** (n / 2)
            self.iF *= (1 / 2) ** (n / 2)
        self.F = coo_matrix(self.F)
        self.iF = coo_matrix(self.iF)
        idx_F = np.mat([self.F.row, self.F.col]).transpose()
        idx_iF = np.mat([self.iF.row, self.iF.col]).transpose()
        self.F = tf.SparseTensor(idx_F, self.F.data.astype(dtype), self.F.shape)
        self.iF = tf.SparseTensor(idx_iF, self.iF.data.astype(dtype), self.iF.shape)
        self.F_dense = tf.sparse_tensor_to_dense(self.F, validate_indices=True)
        self.iF_dense = tf.sparse_tensor_to_dense(self.iF, validate_indices=True)

    def fft(self, s):
        if self.n == 0:
            return s
        F = self.F_dense
        s_hat = tf.tensordot(F, s, axes=[[1], [1]])
        s_hat = tf.transpose(s_hat, [1, 0, 2])
        return s_hat

    def ifft(self, s_hat):
        if self.n == 0:
            return s_hat
        F = self.iF_dense
        s = tf.tensordot(F, s_hat, axes=[[1], [1]])
        s = tf.transpose(s, [1, 0, 2])
        return s


class _SSConvSpectral(base.Layer):
    """Subset convolutional layer.
    """

    def __init__(self, n_filters, model, activation=None, kernel_initializer=None,
                 name=None, trainable=True, use_bias=True, scale=None, **kwargs):
        super(_SSConvSpectral, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_subsets = None
        self.n_channels = None
        self.n_filters = n_filters
        self.model = model
        self.ft = None
        self.activation = activation
        self.use_bias = use_bias
        self.scale = scale
        self.W = None
        self.bias = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1]
        self.n_channels = input_shape[2]
        self.ft = FourierTransform(int(np.log2(self.n_subsets.value)), self.model, dtype=self.dtype)
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            self.W = self.add_variable('W', shape=[self.n_channels, self.n_filters, self.n_subsets],
                                       dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            if self.scale is not None:
                self.W *= self.scale[tf.newaxis, tf.newaxis, :]

            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=self.dtype, trainable=True,
                                              initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters))*0.01))
        super(_SSConvSpectral, self).build(input_shape)

    def call(self, inputs):
        in_hat = self.ft.fft(inputs)  # [n_batch, n_subsets, n_channels]
        in_hat = tf.tile(tf.expand_dims(in_hat, -1),
                         [1, 1, 1, self.n_filters])  # [n_batch, n_subsets, n_channels, n_filters]
        W = tf.transpose(self.W, [2, 0, 1])
        W = W[tf.newaxis, :, :, :]
        outputs = in_hat * W #[n_batch, n_subsets, n_channels, n_filters]
        outputs = tf.reduce_sum(outputs, axis=2)
        outputs = self.ft.ifft(outputs)  # [n_batch, n_subsets, n_filters]

        if self.use_bias:
            outputs += self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])


def conv(inputs, model, filters=32, activation=None, kernel_initializer=None, name=None, reuse=None,
         use_bias=True, use_h0=True, weights=None):
    layer = _SSConvSpectral(filters,
                            model,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            name=name,
                            use_bias=use_bias,
                            use_h0=use_h0,
                            scale=weights,
                            dtype=inputs.dtype.base_dtype,
                            _scope=name,
                            _reuse=reuse)
    return layer.apply(inputs)


class _SSConvSpectralElementary(base.Layer):
    """Subset convolutional layer.
    """

    def __init__(self, n_filters, model, activation=None, kernel_initializer=None, name=None, trainable=True,
                 use_bias=True, **kwargs):
        super(_SSConvSpectralElementary, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_filters = n_filters
        self.n_subsets = None
        self.n_channels = None
        self.n_vertices = None
        self.model = model
        self.use_bias = use_bias
        self.activation = activation
        self.w = None
        self.W = None
        self.bias = None
        self.ft = None
        self.fr = None
        self.coef_indices = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_channels = input_shape[2].value
        self.n_vertices = int(np.log2(self.n_subsets))
        self.ft = FourierTransform(self.n_vertices, self.model, dtype=self.dtype)
        if self.model == 5:
            self.fr = self.ft
        else:
            self.fr = FourierTransform(self.n_vertices, 1, dtype=self.dtype)

        self.coef_indices = np.concatenate([np.zeros(1, dtype=np.int32), 2**np.arange(self.n_vertices)], axis=0)
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            self.w = self.add_variable('w', shape=[self.coef_indices.shape[0], self.n_channels, self.n_filters],
                                       dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=self.dtype, trainable=True,
                                              initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters))* 0.01))
        ind_subsets = self.coef_indices
        ind_channels = np.arange(self.n_channels)
        ind_filters = np.arange(self.n_filters)
        indices = list(itertools.product(ind_subsets, ind_channels, ind_filters))
        indices = np.asarray(indices)
        values = tf.reshape(self.w, [-1])
        W_sparse = tf.SparseTensor(indices, values, [self.n_subsets, self.n_channels, self.n_filters])
        W_sdom = tf.sparse_add(tf.zeros(W_sparse.dense_shape, dtype=self.dtype), W_sparse)
        W_sdom = tf.transpose(W_sdom, [1, 0, 2])
        self.W = self.fr.fft(W_sdom) #[n_chnanels, n_subsets, n_filters]
        self.W = tf.transpose(self.W, [1, 0, 2])
        self.built = True

    def call(self, inputs):
        in_hat = self.ft.fft(inputs)  # [n_batch, n_subsets, n_channels]
        in_hat = tf.tile(tf.expand_dims(in_hat, -1),
                         [1, 1, 1, self.n_filters])  # [n_batch, n_subsets, n_channels, n_filters]
        W = self.W[tf.newaxis, :, :, :]
        outputs = in_hat * W  # [n_batch, n_subsets, n_channels, n_filters]
        outputs = tf.reduce_sum(outputs, axis=2)
        outputs = self.ft.ifft(outputs)  # [n_batch, n_subsets, n_filters]
        if self.use_bias:
            outputs += self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])


class _SSConvSpectralLocalized(base.Layer):
    """Subset convolutional layer.
    """

    def __init__(self, n_filters, model, n_hops=2, activation=None, kernel_initializer=None, name=None, trainable=True,
                 use_bias=True, **kwargs):
        super(_SSConvSpectralLocalized, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_hops = n_hops
        self.n_filters = n_filters
        self.n_subsets = None
        self.n_channels = None
        self.n_vertices = None
        self.model = model
        self.use_bias = use_bias
        self.activation = activation
        self.w = None
        self.W = None
        self.bias = None
        self.ft = None
        self.fr = None
        self.coef_indices = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_channels = input_shape[2].value
        self.n_vertices = int(np.log2(self.n_subsets))
        self.ft = FourierTransform(self.n_vertices, self.model, dtype=self.dtype)
        if self.model == 5:
            self.fr = self.ft
        else:
            self.fr = FourierTransform(self.n_vertices, 1, dtype=self.dtype)
        popc = np.asarray([dssp.pypopcount(A) for A in np.arange(self.n_subsets)])
        self.coef_indices = np.where(popc <= self.n_hops)[0]
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            self.w = self.add_variable('w', shape=[self.coef_indices.shape[0], self.n_channels, self.n_filters],
                                       dtype=self.dtype, initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=self.dtype, trainable=True,
                                              initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters))* 0.01))
        ind_subsets = self.coef_indices
        ind_channels = np.arange(self.n_channels)
        ind_filters = np.arange(self.n_filters)
        indices = list(itertools.product(ind_subsets, ind_channels, ind_filters))
        indices = np.asarray(indices)
        values = tf.reshape(self.w, [-1])
        W_sparse = tf.SparseTensor(indices, values, [self.n_subsets, self.n_channels, self.n_filters])
        W_sdom = tf.sparse_add(tf.zeros(W_sparse.dense_shape, dtype=self.dtype), W_sparse)
        W_sdom = tf.transpose(W_sdom, [1, 0, 2])
        self.W = self.fr.fft(W_sdom) #[n_chnanels, n_subsets, n_filters]
        self.W = tf.transpose(self.W, [1, 0, 2])
        self.built = True

    def call(self, inputs):
        in_hat = self.ft.fft(inputs)  # [n_batch, n_subsets, n_channels]
        in_hat = tf.tile(tf.expand_dims(in_hat, -1),
                         [1, 1, 1, self.n_filters])  # [n_batch, n_subsets, n_channels, n_filters]
        W = self.W[tf.newaxis, :, :, :]
        outputs = in_hat * W  # [n_batch, n_subsets, n_channels, n_filters]
        outputs = tf.reduce_sum(outputs, axis=2)
        outputs = self.ft.ifft(outputs)  # [n_batch, n_subsets, n_filters]
        if self.use_bias:
            outputs += self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])


class _SSConvElementary(base.Layer):
    """Subset convolutional layer (elementary filters). Way slower than the full ones with matrix multiplication?!
    """
    def __init__(self, n_filters, model, activation=None, kernel_initializer=None, name=None, trainable=True,
                 use_bias=True, **kwargs):

        super(_SSConvElementary, self).__init__(trainable=trainable, name=name, **kwargs)
        self.n_filters = n_filters
        self.activation = activation
        self.model = model
        self.use_bias = use_bias
        self.n_subsets = None
        self.n_vertices = None
        self.n_channels = None
        self.w = None
        self.bias = None
        self.indices = None
        if kernel_initializer is None:
            self.kernel_initializer = tf.glorot_uniform_initializer()
        else:
            self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.n_subsets = input_shape[1].value
        self.n_vertices = int(np.log2(self.n_subsets))
        self.n_channels = input_shape[2].value
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            self.w = self.add_variable('w', shape=[self.n_vertices + 1, self.n_channels, self.n_filters],
                                       dtype=tf.float32, initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.bias = self.add_variable('bias', shape=[1, 1, self.n_filters], dtype=tf.float32, trainable=True,
                                            initializer=tf.constant_initializer(np.ones((1, 1, self.n_filters)) * 0.01))
        range = tf.range(self.n_subsets)
        shifts = tf.concat((tf.zeros([1], dtype=tf.int32), 2 ** tf.range(self.n_vertices)), axis=0)
        if self.model == 3:
            self.indices = tf.bitwise.bitwise_and(range[tf.newaxis, :], tf.bitwise.invert(shifts[:, tf.newaxis]))
        elif self.model == 4:
            self.indices = tf.bitwise.bitwise_or(range[tf.newaxis, :], shifts[:, tf.newaxis])
        elif self.model == 5:
            A_minus_Q = tf.bitwise.bitwise_and(range[tf.newaxis, :], tf.bitwise.invert(shifts[:, tf.newaxis]))
            Q_minus_A = tf.bitwise.bitwise_and(tf.bitwise.invert(range[tf.newaxis, :]), shifts[:, tf.newaxis])
            self.indices = tf.bitwise.bitwise_or(A_minus_Q, Q_minus_A)
        else:
            raise NotImplementedError("Only model 3, 4 and 5 are implemented.")
        self.built = True

    def call(self, inputs):
        in_shifted = tf.gather(inputs, tf.reshape(self.indices, [-1]), axis=1)
        in_shifted = tf.reshape(in_shifted, [-1, self.n_vertices + 1, self.n_subsets, self.n_channels])
        outputs = in_shifted[:, :, :, :, tf.newaxis] * self.w[tf.newaxis, :, tf.newaxis, :, :]
        outputs = tf.reduce_sum(outputs, axis=1)
        outputs = tf.reduce_sum(outputs, axis=-2)
        if self.use_bias:
            outputs += self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.convert_to_tensor([input_shape[0], input_shape[1], self.n_filters])


def sconv_el(inputs, filters=32, model=5, activation=None, kernel_initializer=None, name=None, reuse=None):
    layer = _SSConvSpectralElementary(filters, model,
                                      activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      name=name,
                                      dtype=inputs.dtype.base_dtype,
                                      _scope=name,
                                      _reuse=reuse)
    return layer.apply(inputs)






