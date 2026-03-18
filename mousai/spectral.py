"""Spectral utilities for Harmonic Balance methods."""

import numpy as np
import scipy.fftpack as fftp


def harmonic_deriv(omega, r):
    r"""Return derivative of a harmonic function using frequency methods.

    Args:
        omega (float): Fundamendal frequency, in rad/sec, of repeating signal
        r (array_like): | Array of rows of time histories to take the derivative of.
            | The 1 axis (each row) corresponds to a time history.
            | The length of the time histories *must be an odd integer*.

    Returns:
        s (array_like): Function derivatives.
            The 1 axis (each row) corresponds to a time history.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mousai import *
    >>> import scipy as sp
    >>> from numpy import pi, sin, cos
    >>> f = 2
    >>> omega = 2.*pi * f
    >>> numsteps = 11
    >>> t = np.arange(0,1/omega*2*pi,1/omega*2*pi/numsteps)
    >>> x = np.array([sin(omega*t)])
    >>> v = np.array([omega*cos(omega*t)])
    >>> states = np.append(x,v,axis = 0)
    >>> state_derives = harmonic_deriv(omega,states)
    >>> plt.plot(t,states.T,t,state_derives.T,'x')
    [<matplotlib.line...]

    """
    s = np.zeros_like(r)
    for i in np.arange(r.shape[0]):
        s[i, :] = fftp.diff(r[i, :]) * omega
    return np.real(s)


def time_history(t, x, num_time_points=200, realify=True):
    r"""Generate refined time history from harmonic balance solution.

    Harmonic balance solutions presume a limited number of harmonics in the
    solution. The result is that the time history is usually a very limited
    number of values. Plotting these results implies that the solution isn't
    actually a continuous one. This function fills in the gaps using the
    harmonics obtained in the solution.

    Args:
        t (array_like): 1 x m array where m is the number of
            values representing the repeating solution.
        x (array_like): n x m array where m is the number of equations and m is the number of
            values representing the repeating solution.
        num_time_points (int, optional): number of points desired in the "smooth" time history. Defaults to 200.
        realify (boolean, optional): Force the returned results to be real. Defaults to True.

    Returns:
        t (array_like): 1 x num_time_points array.
        x (array_like): n x num_time_points array.

    Examples
    --------
    >>> import numpy as np
    >>> import mousai as ms
    >>> x = np.array([[-0.34996499,  1.36053998, -1.11828552]])
    >>> t = np.array([0.        , 2.991993  , 5.98398601])
    >>> t_full, x_full = ms.time_history(t, x, num_time_points=300)

    Notes
    -----
    The implication of this function is that the higher harmonics that
    were not determined in the solution are zero. This is indeed the assumption
    made when setting up the harmonic balance solution. Whether this is a valid
    assumption is something that the user must judge when obtaining the
    solution.

    """
    dt = t[1]
    t_length = t.size
    if num_time_points < 10 * t.size:
        num_time_points = 10 * t.size
    t = np.linspace(0, t_length * dt, num_time_points, endpoint=False)
    x_freq = fftp.fft(x)
    x_zeros = np.zeros((x.shape[0], t.size - x.shape[1]))
    x_freq = np.insert(x_freq, [t_length - t_length // 2], x_zeros, axis=1)

    x = fftp.ifft(x_freq) * num_time_points / t_length
    if realify is True:
        x = np.real(x)
    else:
        print("x was real")
    return t, x


def condense_fft(X_full, num_harmonics):
    """Create equivalent amplitude reduced-size FFT from longer FFT.

    Args:
        X_full (array_like): Full size FFT.
        num_harmonics (int): Number of harmonics to keep.

    Returns:
        X_red (array_like): Condensed FFT.
    """
    X_red = (
        np.hstack((X_full[:, 0 : (num_harmonics + 1)], X_full[:, -1 : -(num_harmonics + 1) : -1]))
        * (2 * num_harmonics + 1)
        / X_full[0, :].size
    )
    return X_red


def condense_rfft(X_full, num_harmonics):
    """Return real fft with fewer harmonics.

    Args:
        X_full (array_like): Full size real FFT.
        num_harmonics (int): Number of harmonics to keep.

    Returns:
        X_red (array_like): Condensed real FFT.
    """
    X_len = X_full.shape[1]
    X_red = X_full[:, : (num_harmonics) * 2 + 1] / X_len * (1 + 2 * num_harmonics)
    return X_red


def expand_rfft(X, num_harmonics):
    """Return real fft with mor harmonics.

    Args:
        X (array_like): Real FFT.
        num_harmonics (int): Number of harmonics desired.

    Returns:
        X_expanded (array_like): Expanded real FFT.
    """
    X_len = X.shape[1]
    cur_num_harmonics = (X_len - 1) / 2
    X_expanded = np.hstack(
        (
            X / X_len * (1 + 2 * num_harmonics),
            np.zeros((X.shape[0], int(2 * (num_harmonics - cur_num_harmonics)))),
        )
    )
    return X_expanded


def rfft_to_fft(X_real):
    """Switch from SciPy real fft form to complex fft form.

    Args:
        X_real (array_like): Real FFT.

    Returns:
        X (array_like): Complex FFT.
    """
    X = fftp.fft(fftp.irfft(X_real))
    return X


def fft_to_rfft(X):
    """Switch from complex form fft form to SciPy rfft form.

    Args:
        X (array_like): Complex FFT.

    Returns:
        X_real (array_like): Real FFT.
    """
    X_real = fftp.rfft(np.real(fftp.ifft(X)))
    return X_real


def time_history_r(t, x, num_time_points=200, realify=True):
    r"""Generate refined time history from harmonic balance solution.

    Harmonic balance solutions presume a limited number of harmonics in the
    solution. The result is that the time history is usually a very limited
    number of values. Plotting these results implies that the solution isn't
    actually a continuous one. This function fills in the gaps using the
    harmonics obtained in the solution.

    Args:
        t (array_like): 1 x m array where m is the number of
            values representing the repeating solution.
        x (array_like): n x m array where m is the number of equations and m is the number of
            values representing the repeating solution.
        num_time_points (int, optional): number of points desired in the "smooth" time history. Defaults to 200.
        realify (boolean, optional): Force the returned results to be real. Defaults to True.

    Returns:
        t (array_like): 1 x num_time_points array.
        x (array_like): n x num_time_points array.

    Examples
    --------
    >>> import numpy as np
    >>> import mousai as ms
    >>> x = np.array([[-0.34996499,  1.36053998, -1.11828552]])
    >>> t = np.array([0.        , 2.991993  , 5.98398601])
    >>> t_full, x_full = ms.time_history(t, x, num_time_points=300)

    Notes
    -----
    The implication of this function is that the higher harmonics that
    were not determined in the solution are zero. This is indeed the assumption
    made when setting up the harmonic balance solution. Whether this is a valid
    assumption is something that the user must judge when obtaining the
    solution.

    """
    dt = t[1]
    t_length = t.size
    t = np.linspace(0, t_length * dt, num_time_points, endpoint=False)
    x_freq = fftp.fft(x)
    x_zeros = np.zeros((x.shape[0], t.size - x.shape[1]))
    x_freq = np.insert(x_freq, [t_length - t_length // 2], x_zeros, axis=1)
    # print(x_freq)
    # x_freq = np.hstack((x_freq, x_zeros))
    # print(x_freq)
    x = fftp.ifft(x_freq) * num_time_points / t_length
    if realify is True:
        x = np.real(x)
    else:
        print("x was real")
    return t, x
