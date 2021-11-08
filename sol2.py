import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from numpy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift
from scipy.io.wavfile import read, write
from scipy.signal import convolve2d

GRAYSCALE_REPRESENTATION = 1
RGB_REPRESENTATION = 2
MIN_LEVEL = 0
MAX_LEVEL = 255
DIM_IMAGE_GRAY = 2
DIM_IMAGE_RGB = 3
ROW = 0
COL = 1
TO_FREQUENCY_SPACE = -1
FROM_FREQUENCY_SPACE = 1


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
    return (vandermonde_mat(signal.shape[ROW], TO_FREQUENCY_SPACE, 1) @ signal).astype(np.complex128)


def IDFT(fourier_signal):
    """
    transform a Fourier representation of 1D discrete signal back to original signal
    :param fourier_signal: array of dtype complex128 with shape (N,) or (N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    return (vandermonde_mat(fourier_signal.shape[ROW], FROM_FREQUENCY_SPACE,
                            fourier_signal.shape[ROW]) @ fourier_signal).astype(np.complex128)


def vandermonde_mat(N, type, divider):
    """
    :param N: num sumples
    :param type: 1 or -1. 1 if we want transition matrix from frequency space, and -1 to frequency space
    :param divider: 1 or N
    :return: Base transition matrix (to or from the frequency space)
    """
    range_N = np.arange(N)
    return np.exp((type * 2j * np.pi / N) * (range_N.reshape(N, 1) * range_N)) / divider


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    van_row_and_col = [vandermonde_mat(dim, TO_FREQUENCY_SPACE, 1) for dim in image.shape[:2]]
    return (van_row_and_col[ROW] @ image.reshape(image.shape[:2]) @ van_row_and_col[COL].T).astype(
        np.complex128).reshape(image.shape)


def IDFT2(fourier_image):
    """
    convert a Fourier representation of 2D discrete signal to its back to original signal
    :param fourier_image: 2D array of dtype complex128 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    van_row_and_col = [vandermonde_mat(dim, FROM_FREQUENCY_SPACE, dim) for dim in fourier_image.shape[:2]]
    return (van_row_and_col[ROW] @ fourier_image.reshape(fourier_image.shape[:2]) @ van_row_and_col[COL].T).astype(
        np.complex128).reshape(fourier_image.shape)


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples.
    :param filename: a string representing the path to a WAV file
    :param ratio: positive float64 repre- senting the duration change. (assume that 0.25 < ratio < 4)
    :return: None (saves the audio in a new file called "change_rate.wav")
    """
    rate, data = read(filename)
    write("change_rate.wav", int(rate * ratio), data)


def change_samples(filename, ratio):
    """
    changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 repre- senting the duration change
    :return: a 1D ndarray of dtype float64 representing the new sample points.
             (in addition result should be saved in a file called "change_samples.wav")
    """
    rate, data = read(filename)
    resized_data = resize(data, ratio)
    write("change_samples.wav", rate, np.around(resized_data).astype(data.dtype))
    return resized_data.astype(np.float64)


def resize(data, ratio):
    """
    resize the signal
    :param data: is a 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: a positive float64 repre- senting the duration change
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    num_of_samples = data.shape[0]
    new_sample_size = int(num_of_samples * (1 / ratio))
    if new_sample_size == 0:
        return np.array([]).astype(data.dtype)
    num_to_add_or_reduce = int(np.abs(new_sample_size - num_of_samples))

    remainder = num_to_add_or_reduce % 2
    before_reduce = int(np.ceil(num_to_add_or_reduce / 2))
    end_reduce = num_of_samples - before_reduce + remainder
    before_add = int(num_to_add_or_reduce / 2)
    end_add = before_add + remainder

    dft_shifted = fftshift(DFT(data))
    menipulate_dft_shifted = np.pad(dft_shifted, (before_add, end_add), 'constant', constant_values=(0)) if ratio < 1 \
        else dft_shifted[before_reduce: end_reduce] if ratio > 1 else dft_shifted
    return IDFT(ifftshift(menipulate_dft_shifted)) if data.dtype == np.complex128 \
            else IDFT(ifftshift(menipulate_dft_shifted)).real.astype(data.dtype)


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data.
    """
    stft_of_data = stft(data)
    data_menipulate = np.array([resize(stft_of_data[row], ratio) for row in range(stft_of_data.shape[0])])
    ret_val = istft(data_menipulate)
    return ret_val.astype(data.dtype)


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: return the given data rescaled according to ratio with the same datatype as data.
    """
    return istft(phase_vocoder(stft(data), ratio)).astype(data.dtype)


def conv_der(im):
    """
    computes the magnitude of image derivatives using convolution
    :param im: grayscale image of type float64
    :return: the magnitude of the derivative (using convolution)
    """
    der_x_kernel = np.array([[0.5, 0, -0.5]])  # kernel of derivative (df/dx)
    der_y_kernel = der_x_kernel.T
    x_derivative = convolve2d(im, der_x_kernel, mode='same')
    y_derivative = convolve2d(im, der_y_kernel, mode='same')
    magnitude = np.sqrt(np.abs(x_derivative) ** 2 + np.abs(y_derivative) ** 2)
    return magnitude.astype(im.dtype)


def fourier_der(im):
    """
    computes the magnitude of the image derivatives using Fourier transform
    :param im: grayscale image of type float64
    :return: the magnitude of the derivative (using Fourier transform)
    """
    fourier_shifted = fftshift(DFT2(im))
    # fourier_mult = [(2j * np.pi / dim) * np.arange(int(-dim / 2), int(dim / 2) + dim % 2) for dim in im.shape]

    fourier_mult = [np.arange(int(-dim / 2), int(dim / 2) + dim % 2) for dim in im.shape]
    # 2PIi * [-N/2... N/2],2PIi [-M/2....M/2] where im.shape = (N,M)
    fourier_der_row = ifftshift((fourier_shifted.T * fourier_mult[0]).T)  # (2PIi/N)uF(u,v)
    fourier_der_col = ifftshift(fourier_shifted * fourier_mult[1])  # (2PIi/M)vF(u,v)
    magnitude = np.sqrt(np.abs(IDFT2(fourier_der_row)) ** 2 + np.abs(IDFT2(fourier_der_col)) ** 2)
    return magnitude.astype(im.dtype)


############################################### copy from ex2.helper ###############################################

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float64)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex128)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int64)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec