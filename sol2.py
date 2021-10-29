import numpy as np


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal: rray of dtype float64 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    N = signal.shape[0] # num of sumples
    range_N = np.arange(N)
    vandermonde_mat = np.exp((-2j * np.pi / N ) * (range_N.reshape(N,1) * range_N))
    return vandermonde_mat @ signal

