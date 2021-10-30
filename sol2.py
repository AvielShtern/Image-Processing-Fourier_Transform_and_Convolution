import numpy as np
from numpy.linalg import norm
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from numpy.fft import fft, fft2, ifft, ifft2
import time
from scipy.io.wavfile import read, write

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


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    dft_on_each_row = np.array([DFT(image[row]) for row in range(image.shape[0])])
    return np.array([DFT(dft_on_each_row[:,col]) for col in range(image.shape[1])]).T

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
    :return: None (saves the audio in a new file "called change_rate.wav")
    """

if __name__ == '__main__':
    # im = read_image("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX2/ract.jpg",1)
    # print(im.shape)
    # im_dft = DFT2(im)
    # plt.figure()
    # plt.imshow(norm(im_dft.reshape(im.shape[0],im.shape[1],1), axis=2), cmap='gray')
    # plt.axis("off")
    # plt.show()

    signal = np.random.randint(255, size=(500,500))
    beforeme = time.time()
    my_dft = DFT2(signal)
    afterme = time.time()
    before_np = time.time()
    np_dft = fft2(signal)
    after_numpy = time.time()
    print(np.allclose(my_dft,np_dft))
    print(f"me: {afterme - beforeme}. numpy: {after_numpy - before_np}")
