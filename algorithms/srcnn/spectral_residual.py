import numpy as np

from srcnn.utils import average_filter


class SpectralResidual:

    @staticmethod
    def spectral_residual(values: np.ndarray) -> np.ndarray:
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """
        EPS = 1e-8
        trans: np.ndarray = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

        log_mag = np.array(np.log(item) if abs(item) > EPS else 0 for item in mag)

        spectral = np.exp(log_mag - average_filter(log_mag, window_size=3))

        trans.real = [i_real * i_spectral / imag if abs(imag) > EPS else 0
                      for i_real, i_spectral, imag in zip(trans.real, spectral, mag)]
        trans.imag = [i_imag * i_spectral / imag if abs(imag) > EPS else 0
                      for i_imag, i_spectral, imag in zip(trans.imag, spectral, mag)]

        wave_r = np.fft.ifft(trans)
        mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

        return mag

    @staticmethod
    def predict_next(values):
        """
        Predicts the next value by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j)
        :param values: list.
            a list of float numbers.
        :return : float.
            the predicted next value.
        """

        if len(values) <= 1:
            raise ValueError(f'data should contain at least 2 numbers')

        v_last = values[-1]
        n = len(values)

        slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

        return values[1] + sum(slopes)

    @staticmethod
    def extend_series(values, extend_num=5, look_ahead=5):
        """
        extend the array data by the predicted next value
        :param values: list.
            a list of float numbers.
        :param extend_num: int, default 5.
            number of values added to the back of data.
        :param look_ahead: int, default 5.
            number of previous values used in prediction.
        :return: list.
            The result array.
        """

        if look_ahead < 1:
            raise ValueError('look_ahead must be at least 1')

        extension = [SpectralResidual.predict_next(values[-look_ahead - 2:-1])] * extend_num
        return np.concatenate((values, extension), axis=0)
