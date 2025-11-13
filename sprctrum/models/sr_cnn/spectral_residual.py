import numpy as np
import polars as pl


def spectral_residual(values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real**2 + trans.imag**2)

    maglog = [np.log(item) if abs(item) > EPS else 0 for item in mag]

    spectral = np.exp(maglog - average_filter(maglog, n=3))

    trans.real = [
        ireal * ispectral / imag if abs(imag) > EPS else 0
        for ireal, ispectral, imag in zip(trans.real, spectral, mag)
    ]
    trans.imag = [
        iimag * ispectral / imag if abs(imag) > EPS else 0
        for iimag, ispectral, imag in zip(trans.imag, spectral, mag)
    ]

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real**2 + wave_r.imag**2)

    return mag


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
        raise ValueError(f"data should contain at least 2 numbers")

    v_last = values[-1]
    n = len(values)

    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

    return values[1] + sum(slopes)


def extend_series(values: pl.Series, extend_num=5, look_ahead=5) -> np.ndarray:
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
        raise ValueError("look_ahead must be at least 1")

    extension = [predict_next(values[-look_ahead - 2 : -1])] * extend_num
    return np.concatenate((values, extension), axis=0)


def average_filter(values, n=3):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= i + 1

    return res
