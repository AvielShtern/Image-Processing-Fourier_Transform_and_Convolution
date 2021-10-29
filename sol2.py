import numpy as np


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal: rray of dtype float64 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    return vandermonde_mat(signal.shape[0], -1, 1) @ signal


def IDFT(fourier_signal):
    """
    transform a Fourier representation of 1D discrete signal back to original signal
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    return vandermonde_mat(fourier_signal.shape[0], 1, fourier_signal.shape[0]) @ fourier_signal


def vandermonde_mat(N, type, divider):
    """
    :param N: num sumples
    :param type: 1 or -1
    :param divider: 1 or N
    :return:
    """
    range_N = np.arange(N)
    return np.exp((type * 2j * np.pi / N) * (range_N.reshape(N, 1) * range_N)) / divider


if __name__ == '__main__':
    SIGNAL = np.arange(20)
    print(np.allclose(SIGNAL,IDFT(DFT(SIGNAL)).real))
