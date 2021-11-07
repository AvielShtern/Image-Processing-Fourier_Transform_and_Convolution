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
    # dft_on_each_row = np.array([DFT(image[row]) for row in range(image.shape[0])])
    # return np.array([DFT(dft_on_each_row[:, col]) for col in range(image.shape[1])]).T

    # van_row = vandermonde_mat(image.shape[0], -1, 1)
    # van_col = vandermonde_mat(image.shape[1], -1, 1)
    # return van_row @ image @ van_col.T
    return fft2(image)


def IDFT2(fourier_image):
    """
    convert a Fourier representation of 2D discrete signal to its back to original signal
    :param fourier_image: 2D array of dtype complex128 with shape (M,N) or (M,N,1)
    :return: array of dtype complex128 with the same shape of input
    """
    # idft_on_each_row = np.array([IDFT(fourier_image[row]) for row in range(fourier_image.shape[0])])
    # return np.array([IDFT(idft_on_each_row[:, col]) for col in range(fourier_image.shape[1])]).T
    #
    # van_row = vandermonde_mat(fourier_image.shape[0], 1, fourier_image.shape[0])
    # van_col = vandermonde_mat(fourier_image.shape[1], 1, fourier_image.shape[1])
    # return van_row @ fourier_image @ van_col.T

    return ifft2(fourier_image)


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
    write("change_samples.wav", rate, resize(data, ratio).real)


def resize(data, ratio):
    """
    resize the signal
    :param data: is a 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: a positive float64 repre- senting the duration change
    :return: 1D ndarray of the dtype of data representing the new sample points
    """
    # num_of_samples = data.shape[0]
    # num_to_add_or_reduce = int(np.abs(num_of_samples * (1 / ratio) - num_of_samples))
    # dft_shifted = fftshift(DFT(data))
    # before = int(num_to_add_or_reduce / 2)
    # end = int(np.ceil(num_to_add_or_reduce / 2))
    # menipulate_dft_shifted = np.pad(dft_shifted, (before, end), 'constant', constant_values=(0)) if ratio < 1 \
    #     else dft_shifted[before: -end] if ratio > 1 else dft_shifted
    #
    # return IDFT(ifftshift(menipulate_dft_shifted))
    num_of_samples = data.shape[0]
    new = int(num_of_samples * (1 / ratio))
    dft_shifted = fftshift(DFT(data))
    num_to_add_or_reduce = int(np.abs(new - num_of_samples))
    before = np.ceil(num_to_add_or_reduce / 2)
    fac = num_to_add_or_reduce % 2
    end = num_of_samples - before + fac
    menipulate_dft_shifted = np.pad(dft_shifted, (before, before + fac), 'constant', constant_values=(0)) if ratio < 1 \
        else dft_shifted[before: end] if ratio > 1 else dft_shifted
    return IDFT(ifftshift(menipulate_dft_shifted))



def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling.
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data.
    """
    stft_of_data = stft(data)  # stft return a metrix with shape (win_length,n_frames) ??? why every "row" and not col??
    data_menipulate = np.array([resize(stft_of_data[row], ratio) for row in range(stft_of_data.shape[0])])
    ret_val = istft(data_menipulate)  # change the window size????
    # ret_val = istft(data_menipulate)
    return ret_val


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: return the given data rescaled according to ratio with the same datatype as data.
    """
    return istft(phase_vocoder(stft(data), ratio))


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
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
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
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

# if __name__ == '__main__':
#     # path = "/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX2/ex2_presubmit/aria_4kHz.wav"
#     # rate, data = read(path)
#     # datatype = data.dtype
#     # data = resize_vocoder(data, 1.5)
#     # write("res.wav", rate, data.real.astype(datatype))
#     # rate, data = read("/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX2/disco_dancing.wav")
#     # write("new.wav",rate,data[:,0])
#     # write("new.wav",rate,resize_spectrogram(data[:,0],2).real)
#     # data_to_work = data[:2000,0]
#     # print(data_to_work.shape)
#     # stft_data = stft(data_to_work)
#     # print(stft_data.shape)
#     # istft_data  = istft(stft_data)
#     # print(istft_data.shape)
#     # resize_spectrogram(data[:,0],2)
#     path = "/Users/avielshtern/Desktop/third_year/IMAGE_PROCESSING/EX/EX1/ex1-aviel.shtern/examples/monkey.jpg"
#     im = read_image(path,1)
#     mag_ber = conv_der(im)
#     mag_four = fourier_der(im)
#     print(np.allclose(mag_ber,mag_four))

