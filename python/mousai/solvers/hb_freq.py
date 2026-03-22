import logging
from typing import Any, Callable

import numpy as np
import scipy.fftpack as fftp

from mousai.spectral import condense_rfft, harmonic_deriv, time_history

from .common import (
    _SOLVERS,
    EquationForm,
    HarmonicBalanceSolution,
    SolverMethod,
    _prepare_hb_inputs,
)

log = logging.getLogger(__name__)


def hb_freq(
    sdfunc: Callable[..., np.ndarray],
    x0: np.ndarray | None = None,
    omega: float = 1,
    num_harmonics: int = 1,
    params: dict[str, Any] | None = None,
    eqform: EquationForm = "second_order",
    num_time_steps: int | None = None,
    mask_constant: bool = True,
    method: SolverMethod = "newton_krylov",
    num_variables: int | None = None,
    realify: bool = True,
    **kwargs: Any,
) -> HarmonicBalanceSolution:
    r"""Harmonic balance solver for first and second order ODEs.

    Obtains the solution of a first-order and second-order differential
    equation under the presumption that the solution is harmonic using a
    frequency-domain algebraic method.

    Args:
        sdfunc (function): For `eqform='first_order'`, name of function that returns **column
            vector** first derivative given `x`, and a dictionry of parameters.
            This is *NOT* a string (not the name of the function).
            :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`
            For `eqform='second_order'`, name of function that returns **column
            vector** second derivative given `x`, `v`, and a dictionary of
            parameters.
            :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)`
        x0 (array_like, optional): n x m array where n is the number of equations and m is the number of
            values representing the repeating solution.
            It is required that :math:`m = 1 + 2 num_{harmonics}`. If not provided,
            a zero-valued guess will be used.
        omega (float): assumed fundamental response frequency in radians per second. Defaults to 1.
        num_harmonics (int, optional): Number of harmonics to presume. The `omega` = 0 constant term is
            always presumed to exist. Defaults to 1.
        params (dict, optional): Dictionary of parameters needed by sdfunc. Defaults to None.
        eqform (str, optional): `second_order` or `first_order`. (`second order` is default). Defaults to 'second_order'.
        num_time_steps (int, optional): number of time steps to use in intermediate time histories for derivative
            calculations. A higher number increases accuracy at the cost of performance.
            Defaults to max(51, 4 * num_harmonics + 1).
        mask_constant (boolean, optional): If True, the DC (constant) term of the solution is not solved for
            and is assumed to be zero. Defaults to True.
        method (str, optional): Name of optimization method to be used. Defaults to 'newton_krylov'.
        num_variables (int, optional): Number of states for a state space model, or number of generalized
            dispacements for a second order form.
            If `x0` is defined, num_variables is inferred. An error will result if
            both `x0` and num_variables are left out of the function call.
            `num_variables` must be defined if `x0` is not. Defaults to None.
        realify (boolean, optional): Force the returned results to be real. Defaults to True.
        **kwargs: Other keyword arguments available to nonlinear solvers in
            `scipy.optimize.nonlin
            <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_.

    Returns:
        HarmonicBalanceSolution: A named tuple containing (t, x, e, amps, phases).

    Examples
    --------
    >>> import mousai as ms
    >>> # Initial guess is a time history, like in hb_time
    >>> x0 = np.array([[0, 1, -1]])
    >>> t, x, e, amps, phases = ms.hb_freq(ms.duff_osc, x0, omega=0.7)

    Notes
    -----
    This function solves for the Fourier coefficients of the solution directly.
    It transforms the differential equation into a set of algebraic equations
    in the frequency domain and solves them numerically.
    """
    # --- Input Validation and Initialization ---
    x0, num_variables, params = _prepare_hb_inputs(omega, num_harmonics, x0, num_variables, params)

    if num_time_steps is None:
        # Set a default number of time steps for intermediate calculations.
        # This should be higher than the Nyquist rate for the harmonics.
        num_time_steps = max(51, 4 * num_harmonics + 1)
    elif num_time_steps <= 2 * num_harmonics:
        raise ValueError(
            f"'num_time_steps' ({num_time_steps}) must be greater than "
            f"2 * num_harmonics ({2 * num_harmonics}) to avoid aliasing."
        )

    # Convert time-domain guess to frequency domain coefficients
    X0 = fftp.rfft(x0)
    if mask_constant:
        X0 = X0[:, 1:]

    # --- Harmonic Balance Error Function Definition (Closure) ---
    def hb_err(X: np.ndarray) -> np.ndarray:
        """
        Calculate the harmonic balance error in the frequency domain.

        This closure takes the Fourier coefficients of the guess, reconstructs
        the time-domain signal, evaluates the governing equations, and returns
        the error in the frequency domain.

        Args:
            X: Fourier coefficients (possibly lacking DC term if masked).

        Returns:
            The frequency-domain error for the current guess.
        """
        # If the DC term was masked, prepend a zero coefficient column.
        if mask_constant:
            X_full = np.hstack((np.zeros((X.shape[0], 1)), X))
        else:
            X_full = X

        # Reconstruct time history from coefficients
        x = fftp.irfft(X_full)
        # Create a finer time mesh for accurate nonlinear evaluation
        time_base = np.linspace(0, 2 * np.pi / omega, num=x.shape[1], endpoint=False)
        time_e, x = time_history(time_base, x, num_time_points=num_time_steps)

        # Calculate derivatives in the time domain
        vel = harmonic_deriv(omega, x)

        local_params = params.copy()
        local_params["omega"] = omega

        # Calculate derivatives from the governing equations
        if eqform == "second_order":
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            for i in range(num_time_steps):
                local_params["cur_time"] = time_e[i]
                accel_from_deriv[:, i] = sdfunc(x[:, i], vel[:, i], local_params)[:, 0]
            e_time = accel_from_deriv - accel
        elif eqform == "first_order":
            vel_from_deriv = np.zeros_like(vel)
            for i in range(num_time_steps):
                local_params["cur_time"] = time_e[i]
                vel_from_deriv[:, i] = sdfunc(x[:, i], local_params)[:, 0]
            e_time = vel_from_deriv - vel
        else:
            raise ValueError(f"eqform cannot have a value of '{eqform}'")

        # Transform error back to frequency domain
        e_fft = fftp.rfft(e_time)
        # Condense error to match the number of harmonics being solved
        e = condense_rfft(e_fft, num_harmonics)

        if mask_constant:
            e = e[:, 1:]

        return e

    # --- Setup for Solver ---
    # We only need to define `time` here for the initial guess structure if needed,
    # but `hb_err` generates its own time vector.

    # --- Invoke the Nonlinear Solver ---
    solver = _SOLVERS.get(method)
    if not solver:
        raise ValueError(
            f"Unknown solver '{method}'. Available solvers are: {list(_SOLVERS.keys())}"
        )

    log.info("Starting harmonic balance solver '%s' for omega=%.4f", method, omega)
    try:
        X = solver(hb_err, X0, **kwargs)
    except Exception as e:
        log.error(
            "The '%s' solver failed to converge for omega=%.4f.",
            method,
            omega,
            exc_info=True,
        )
        raise RuntimeError(
            f"The '{method}' solver failed to converge for omega={omega:.4f}."
        ) from e

    log.info("Solver '%s' converged.", method)

    # --- Post-process the Solution ---
    if mask_constant:
        X_full = np.hstack((np.zeros((X.shape[0], 1)), X))
    else:
        X_full = X

    # Reconstruct final time history
    x = fftp.irfft(X_full)
    e = hb_err(X)
    time = np.linspace(0, 2 * np.pi / omega, num=x.shape[1], endpoint=False)

    # Calculate amplitudes and phases.
    # For scipy.fftpack.rfft, index 1 is Re(1st), index 2 is Im(1st).
    # X_full shape is (n_vars, 1 + 2*n_harmonics).
    if X_full.shape[1] > 2:
        amps = np.sqrt(X_full[:, 1] ** 2 + X_full[:, 2] ** 2) * 2 / X_full.shape[1]
        phases = np.arctan2(X_full[:, 1], -X_full[:, 2])
    else:
        amps = np.zeros(X.shape[0])
        phases = np.zeros(X.shape[0])

    if realify:
        x = np.real(x)

    return HarmonicBalanceSolution(time, x, e, amps, phases)
