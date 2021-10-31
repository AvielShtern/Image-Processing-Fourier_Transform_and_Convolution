import numpy as np
from numpy.linalg import norm
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from numpy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift
import time
from scipy.io.wavfile import read, write
from ex2_helper import *

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2
MIN_LEVEL = 0
MAX_LEVEL = 255
DIM_IMAGE_GRAY = 2
DIM_IMAGE_RGB = 3


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: either 1 or 2 defining whether the output should be a grayscale image (1) or an RGB image (2).
    :return: image is represented by a matrix of type np.float64 with intensities (either grayscale or RGB channel
             intensities) normalized to the range [0, 1].
    """
    rgb_or_gray_scale_image = imread(filename).astype(np.float64) / MAX_LEVEL
    if representation == RGB_REPRESENTATION or (
            representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == DIM_IMAGE_GRAY):
        return rgb_or_gray_scale_image
    elif representation == GRAYSCALE_REPRESENTATION and rgb_or_gray_scale_image.ndim == DIM_IMAGE_RGB:
        return rgb2gray(rgb_or_gray_scale_image)


def imdisplay(filename, representation):
    """
     open a new figure and display the loaded image in the converted representation
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: either 1 or 2 defining whether the display should be a grayscale image (1) or an RGB image (2).
    :return: None
    """
    im = read_image(filename, representation)
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.axis("off")
    plt.show()


def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal: rray of dtype float64 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    # return vandermonde_mat(signal.shape[0], -1, 1) @ signal
    return fft(signal)


def IDFT(fourier_signal):
    """
    transform a Fourier representation of 1D discrete signal back to original signal
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    # return vandermonde_mat(fourier_signal.shape[0], 1, fourier_signal.shape[0]) @ fourier_signal
    return ifft(fourier_signal)


def vandermonde_mat(N, type, divider):
    """
    :param N: num sumples
    :param type: 1 or -1
    :param divider: 1 or N
    :return:
    """
    range_N = np.arange(N)
    return np.exp((type * 2j * np.pi / N) * (range_N.reshape(N, 1) * range_N)) / divider


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    dft_on_each_row = np.array([DFT(image[row]) for row in range(image.shape[0])])
    return np.array([DFT(dft_on_each_row[:, col]) for col in range(image.shape[1])]).T


def IDFT2(fourier_image):
    """
    convert a Fourier representation of 2D discrete signal to its back to original signal
    :param fourier_image: 2D array of dtype complex128 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    idft_on_each_row = np.array([IDFT(fourier_image[row]) for row in range(fourier_image.shape[0])])
    return np.array([IDFT(idft_on_each_row[:, col]) for col in range(fourier_image.shape[1])]).T


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples.
    :param filename: a string representing the path to a WAV file
    :param ratio: positive float64 repre- senting the duration change. (assume that 0.25 < ratio < 4)
    :return: None (saves the audio in a new file called "change_rate.wav")
    """
    rate, data = read(filename)
    write("change_rate.wav", int(np.around(rate * ratio)), data)


def change_samples(filename, ratio):
    """
    changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 repre- senting the duration change
    :return: None. (result should be saved in a file called "change_samples.wav")
    """
    rate, data = read(filename)
    write("change_samples.wav", rate, resize(data,ratio).real)

def resize(data, ratio):
    """
    resize the signal
    :param data: is a 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: a positive float64 repre- senting the duration change
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    num_of_samples = data.shape[0]
    num_to_add_or_reduce = int(np.abs(num_of_samples * (1 / ratio) - num_of_samples))
    dft_shifted = fftshift(DFT(data))
    before = int(num_to_add_or_reduce/ 2)
    end = int(np.ceil(num_to_add_or_reduce / 2))
    menipulate_dft_shifted = np.pad(dft_shifted, (before, end), 'constant', constant_values=(0)) if ratio < 1 \
                     else dft_shifted[before: -end] if ratio > 1 else dft_shifted

    return IDFT(ifftshift(menipulate_dft_shifted))


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data.
    """
    stft_of_data = stft(data).T # stft return a metrix with shape (win_length,n_frames) ??? why every "row" and not col??
    data_menipulate = np.array([resize(stft_of_data[row], ratio) for row in range(stft_of_data.shape[0])]).T
    ret_val = istft(data_menipulate, int(stft_of_data[0].shape[0] * 1/ratio)) # change the window size????
    # ret_val = istft(data_menipulate)
    return ret_val


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: return the given data rescaled according to ratio with the same datatype as data.
    """
    return istft(phase_vocoder(stft(data),ratio))


if __name__ == '__main__':
    path = "/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX2/ex2_presubmit/aria_4kHz.wav"
    rate,data = read(path)
    data = resize_spectrogram(data,2)
    write("res.wav",rate,data.real)
    # rate, data = read("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX2/disco_dancing.wav")
    # write("new.wav",rate,data[:,0])
    # write("new.wav",rate,resize_spectrogram(data[:,0],2).real)
    # data_to_work = data[:2000,0]
    # print(data_to_work.shape)
    # stft_data = stft(data_to_work)
    # print(stft_data.shape)
    # istft_data  = istft(stft_data)
    # print(istft_data.shape)
    # resize_spectrogram(data[:,0],2)

