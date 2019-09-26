import numpy as np
import tensorflow as tf
from .. import dssp

from .dataset import Dataset


def artificial_hard(N, MIX_FACTOR=1):
    support_class1 = np.random.randint(0, 2, 2**N).astype(np.bool)
    support_class2 = np.random.randint(0, 2, 2**N).astype(np.bool)
    data1 = np.random.normal(0, 1, size=(10000, 2**N))
    data1[:, support_class1] += MIX_FACTOR*np.random.normal(0, 1, size=(10000, np.sum(support_class1)))
    data2 = np.random.normal(0, 1, size=(10000, 2**N))
    data2[:, support_class2] += MIX_FACTOR*np.random.normal(0, 1, size=(10000, np.sum(support_class2)))
    data3 = np.random.normal(0, 1, size=(10000, 2**N))
    data4 = np.random.normal(0, 1, size=(10000, 2**N))
    data4[:, (support_class1 == support_class2)] += MIX_FACTOR*np.random.normal(0, 1,
                                                                                size=(10000,
                                                                                      np.sum((support_class1 ==
                                                                                              support_class2))))
    data = np.concatenate((data1, data2, data3, data4), axis=0)
    data = data.astype(np.float32)
    labels = [0]*data1.shape[0] + [1]*data2.shape[0] + [2]*data3.shape[0] + [3]*data4.shape[0]
    labels = np.asarray(labels)
    for idx, d in enumerate(data):
        data[idx] = dssp.fidsft3(d)
    return data, labels


def artificial_junta(N):
    data = []
    labels = []
    for dummies in range(3, 8):
        for _ in range(10000):
            s = np.random.rand(2**N).astype(np.float32)
            perm = 2**np.random.permutation(N)
            for x in perm[:dummies]:
                s = dssp.shift(x, s, 3)
            data += [s[np.newaxis, :]]
            labels += [dummies - 3]
    data = np.concatenate(data, axis=0)
    labels = np.asarray(labels)
    return data, labels


def artificial_xmodular(N, k=10000):
    def random_coverage(N):
        s_hat = -np.random.rand(2**N)
        s_hat[1:] /= np.linalg.norm(s_hat[1:], 1)
        s_hat[0] = 1
        return dssp.fidsft4(s_hat)

    def random_almostcoverage(N):
        s_hat = -np.random.rand(2 ** N)
        s_hat[1:] /= np.linalg.norm(s_hat[1:], 1)
        s_hat[0] = 1
        n_flips = np.random.randint(1, 2**(N-1))
        flips = np.random.permutation(2**N)[:n_flips]
        s_hat[flips] = -s_hat[flips]
        return dssp.fidsft4(s_hat)

    submodular = [random_coverage(N) for _ in range(k)]
    almost = [random_almostcoverage(N) for _ in range(k)]

    data = submodular + almost
    data = [d[np.newaxis, :] for d in data]
    data = np.concatenate(data, axis=0)
    data -= np.min(data, axis=1, keepdims=True)
    data /= np.linalg.norm(data, np.inf, axis=1, keepdims=True)
    labels = np.asarray([0] * k + [1] * k)
    data = data.astype(np.float32)
    return data, labels


def domain(N, n_classes=4):
    if n_classes == 6:
        return np.load('data/dom6/data_train.npy'), np.load('data/dom6/labels_train.npy'), \
               np.load('data/dom6/data_test.npy'), np.load('data/dom6/labels_test.npy')
    elif n_classes == 4:
        return np.load('data/dom4/data_train.npy'), np.load('data/dom4/labels_train.npy'), \
               np.load('data/dom4/data_test.npy'), np.load('data/dom4/labels_test.npy')
    else:
        raise NotImplementedError


def congress10(N):
    data_train = np.load('data/congress10/data_train.npy').astype(np.float32)
    labels_train = np.load('data/congress10/labels_train.npy')
    data_test = np.load('data/congress10/data_test.npy').astype(np.float32)
    labels_test = np.load('data/congress10/labels_test.npy')
    data_train = np.minimum(1, data_train)
    data_test = np.minimum(1, data_test)
    return data_train, labels_train, data_test, labels_test


def coauth10(N):
    train_name = 'coauth-DBLP'
    test_names = ['coauth-MAG-Geology', 'coauth-MAG-History']
    train_data = np.load('data/%s/data.npy'%(train_name)).astype(np.float32)
    train_labels = np.load('data/%s/labels.npy'%(train_name)).astype(np.float32)
    test_data = []
    test_labels = []
    for name in test_names:
        test_data.append(np.load('data/%s/data.npy' % (name)))
        test_labels.append(np.load('data/%s/labels.npy' % (name)))
    test_data = np.concatenate(test_data, axis=0).astype(np.float32)
    test_labels = np.concatenate(test_labels, axis=0).astype(np.float32)
    test_data = np.minimum(1, test_data)
    train_data = np.minimum(1, train_data)
    return test_data, test_labels, train_data, train_labels


class SetfunctionDataset(Dataset):
    def __init__(self, N, data_generator=artificial_junta, train_fraction=0.8, *args, **kwargs):
        self.N = N
        data_tuple = data_generator(N)
        if len(data_tuple) == 2:
            data, labels = data_tuple
            self._n_classes = len(np.unique(labels))
            perm = np.random.permutation(np.arange(len(data)))
            data = data[perm]
            labels = labels[perm]
            labels = tf.one_hot(labels, self._n_classes)
            data = data[:, :, np.newaxis]
            self.data_train = data[:int(len(data)*train_fraction)]
            self.labels_train = labels[:int(len(data)*train_fraction)]
            self.data_test = data[int(len(data)*train_fraction):]
            self.labels_test = labels[int(len(data)*train_fraction):]
        elif len(data_tuple) == 4:
            data_train, labels_train, data_test, labels_test = data_tuple
            self._n_classes = len(np.unique(labels_train))
            self.data_train = data_train[:, :, np.newaxis]
            self.labels_train = tf.one_hot(labels_train, self._n_classes)
            self.data_test = data_test[:, :, np.newaxis]
            self.labels_test = tf.one_hot(labels_test, self._n_classes)
        else:
            raise ValueError('data_generator return values are not of supported format')
        self._size = len(self.data_train)
        self._test_size = len(self.data_test)
        self._n_groundset = int(np.log2(self.data_train.shape[1]))

    def get_tf_dataset(self, *args, **kwargs):
        train_set_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.data_train))
        train_set_labels = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.labels_train))
        train_set = tf.data.Dataset.zip((train_set_data, train_set_labels))
        return train_set

    def get_testing_data(self):
        test_set_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.data_test))
        test_set_labels = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.labels_test))
        test_set = tf.data.Dataset.zip((test_set_data, test_set_labels))
        return test_set

    @property
    def test_size(self):
        return self._test_size

    @property
    def size(self):
        return self._size

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_groundset(self):
        return self._n_groundset
