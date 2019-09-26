import numpy as np


def conv3(h, s):
    frequency_response = fr(h)
    s_hat = fdsft3(s)
    return fidsft3(frequency_response * s_hat)


def conv4(h, s):
    frequency_response = fr(h)
    s_hat = fdsft4(s)
    return fidsft4(frequency_response * s_hat)


def fr(signal):
    return fidsft4(signal)


def fdsft3(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = x
                transform[j + h] = x - y
        h *= 2
    return transform


def fidsft3(transform):
    return fdsft3(transform)


def fdsft4(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = y
                transform[j + h] = x - y
        h *= 2
    return transform


def fidsft4(signal):
    N = len(signal)
    h = 1
    transform = signal.copy()
    while h < N:
        for i in range(0, N, 2*h):
            for j in range(i, i+h):
                x = transform[j]
                y = transform[j + h]
                transform[j] = x + y
                transform[j + h] = x
        h *= 2
    return transform


def popcount(arr):
    N = arr.shape[0]
    out = np.asarray([pypopcount(A) for A in arr])
    return out


def pypopcount(n):
    """ this is actually faster """
    return bin(n).count('1')


def int2indicator(A, n_groundset):
    indicator = [int(b) for b in bin(2**n_groundset + A)[3:][::-1]]
    return np.asarray(indicator, dtype=np.bool)


def int2elements(A, n_groundset):
    indicator = int2indicator(A, n_groundset)
    return 2**np.arange(n_groundset)[indicator]


def shift(Q, s, model=3):
    """
    :param Q: integer representation of subset Q
    :param s:
    :param model:
    :return:
    """
    N = int(np.log2(len(s)))
    subsets = np.arange(2**N, dtype=np.int64)
    if model == 1:
        if pypopcount(Q) != 1:
            raise NotImplementedError('model 1 is not implemented')
        shifted = s + shift(Q, s, 3)
        shifted[subsets & Q != Q] = 0
        return shifted
    elif model == 2:
        if pypopcount(Q) != 1:
            raise NotImplementedError('model 1 is not implemented')
        shifted = s + shift(Q, s, 4)
        shifted[subsets & Q == Q] = 0
        return shifted
    elif model == 3:
        return s[subsets & (~Q)]
    elif model == 4:
        return s[subsets | Q]
    elif model == 5:
        return s[(subsets & (~Q)) | (Q & (~subsets))]
    else:
        raise NotImplementedError('model not implemented')



