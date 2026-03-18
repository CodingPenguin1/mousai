"""Example models and constitutive equation helpers."""

import numpy as np
import scipy.linalg as la


def solmf(x, v, M, C, K, F):
    r"""Return acceleration of second order linear matrix system.

    Parameters
    ----------
    x, v, F : array_like
        :math:`n\times 1` arrays of current displacement, velocity, and Force.
    M, C, K : array_like
        Mass, damping, and stiffness matrices.

    Returns
    -------
    a : array_like
        :math:`n\\times 1` acceleration vector

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[2,0],[0,1]])
    >>> K = np.array([[2,-1],[-1,3]])
    >>> C = 0.01 * M + 0.01 * K
    >>> x = np.array([[1],[0]])
    >>> v = np.array([[0],[10]])
    >>> F = v * 0.1
    >>> a = solmf(x, v, M, C, K, F)
    >>> print(a)
        [[-0.95]
         [ 1.6 ]]

    """
    return -la.solve(M, C @ v + K @ x - F)


def duff_osc(x, v, params):
    """Duffing oscillator acceleration."""
    omega = params["omega"]
    t = params["cur_time"]
    acceleration = np.array([[-x - 0.1 * x**3.0 - 0.2 * v + np.sin(omega * t)]])
    return acceleration
