import logging
from typing import Any, Callable

import numpy as np
import scipy.fftpack as fftp

from mousai.spectral import harmonic_deriv

from .common import (
    _SOLVERS,
    EquationForm,
    HarmonicBalanceSolution,
    SolverMethod,
    _prepare_hb_inputs,
)

log = logging.getLogger(__name__)


def hb_time(
    sdfunc: Callable[..., np.ndarray],
    x0: np.ndarray | None = None,
    omega: float = 1,
    num_harmonics: int = 1,
    params: dict[str, Any] | None = None,
    eqform: EquationForm = "second_order",
    method: SolverMethod = "newton_krylov",
    num_variables: int | None = None,
    realify: bool = True,
    **kwargs: Any,
) -> HarmonicBalanceSolution:
    r"""Harmonic balance solver for first and second order ODEs.

    Args:
        sdfunc (function): For `eqform='first_order'`, name of function that returns **column
            vector** first derivative given `x`, and a dictionry of parameters.
            This is *NOT* a string (not the name of the function).

            :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`

            For `eqform='second_order'`, name of function that returns **column
            vector** second derivative given `x`, `v`, and a dictionary of
            parameters. This is *NOT* a string.

            :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)`
        x0 (array_like, optional): n x m array where n is the number of equations and m is the number of
            values representing the repeating solution.
            It is required that :math:`m = 1 + 2 num_{harmonics}`. (we will
            generalize allowable default values later.)
        omega (float): assumed fundamental response frequency in radians per second. Defaults to 0.
        num_harmonics (int, optional): Number of harmonics to presume. The omega = -1 constant term is always
            presumed to exist. Minimum (and default) is 0. If num_harmonics*2+1
            exceeds the number of columns of `x0` then `x0` will be expanded, using
            Fourier analaysis, to include additional harmonics with the starting
            presumption of zero values. Defaults to 0.
        params (dict, optional): Dictionary of parameters needed by sdfunc. Defaults to None.
        eqform (str, optional): `second_order` or `first_order`. (second order is default). Defaults to 'second_order'.
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
            See `Notes`.

    Returns:
        HarmonicBalanceSolution: A named tuple containing (t, x, e, amps, phases).

    Obtains the solution of a first-order and second-order differential
    equation under the presumption that the solution is harmonic using an
    algebraic time method.

    Returns `t` (time), `x` (displacement), `v` (velocity), and `a`
    (acceleration) response of a first- or second- order linear ordinary
    differential equation defined by
    :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)` or
    :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`.

    For the state space form, the function `sdfunc` should have the form::

        def duff_osc_ss(x, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            xdot = np.array([[x[1]],[-x[0]-.1*x[0]**3-.1*x[1]+1*np.sin(omega*t)]])
            return xdot

    In a state space form solution, the function must accept the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    For the second order form the function `sdfunc` should have the form::

        def duff_osc(x, v, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            return np.array([[-x-.1*x**3-.2*v+np.sin(omega*t)]])

    In a second-order form solution the function must take the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    Examples
    --------
    >>> import mousai as ms
    >>> t, x, e, amps, phases = ms.hb_time(ms.duff_osc,
    ...                                    np.array([[0,1,-1]]),
    ...                                    omega = 0.7)

    Notes
    -----
    .. seealso::

       ``hb_freq``

    This method is not reliable for a low number of harmonics.

    Calls a linear algebra function from
    `scipy.optimize.nonlin
    <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_ with
    `newton_krylov` as the default.

    Evaluates the differential equation/s at evenly spaced points in time. Each
    point in time yields a single equation. One harmonic plus the constant term
    results in 3 points in time over the cycle.

    Solver should gently "walk" solution up to get to nonlinearities for hard
    nonlinearities.

    Algorithm:
        1. calls `hb_err` with `x` as the variable to solve for.
        2. `hb_err` uses a Fourier representation of `x` to obtain
           velocities (after an inverse FFT) then calls `sdfunc` to determine
           accelerations.
        3. Accelerations are also obtained using a Fourier representation of x
        4. Error in the accelerations (or state derivatives) are the functional
           error used by the nonlinear algebraic solver
           (default `newton_krylov`) to be minimized by the solver.

    Options to the nonlinear solvers can be passed in by \*\*kwargs (keyword
    arguments) identical to those available to the nonlinear solver.
    """
    # --- Input Validation and Initialization ---
    x0, num_variables, params = _prepare_hb_inputs(omega, num_harmonics, x0, num_variables, params)

    # --- Harmonic Balance Error Function Definition ---
    def hb_err(x: np.ndarray) -> np.ndarray:
        """
        Calculate the harmonic balance error in the time domain.

        This inner function is a closure, capturing variables like `omega`,
        `num_harmonics`, `time`, `eqform`, `sdfunc`, and the user `params`
        from the surrounding `hb_time` scope.

        It computes the error between the derivatives calculated from the
        governing equations (`sdfunc`) and the derivatives calculated from
        the Fourier series of the current guess `x`. This error is what the
        nonlinear solver attempts to minimize.

        Args:
            x: The current guess for the time history of the states.

        Returns:
            The time-domain error for the current guess `x`.
        """
        m = 1 + 2 * num_harmonics
        vel = harmonic_deriv(omega, x)

        # The user's `sdfunc` expects a `params` dictionary. We'll create a
        # local copy of the user-provided dict and add solver-specific
        # values to it for each time step.
        local_params = params.copy()
        local_params["omega"] = omega

        if eqform == "second_order":
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            # Should subtract in place below to save memory for large problems
            for i in range(m):
                local_params["cur_time"] = time[i]
                # Call the user's function to get the derivative from the equation.
                accel_from_deriv[:, i] = sdfunc(x[:, i], vel[:, i], local_params)[:, 0]
            # The error is the difference between the derivative from the equation
            # and the derivative from the Fourier series of the guess `x`.
            e = accel_from_deriv - accel
        elif eqform == "first_order":
            vel_from_deriv = np.zeros_like(vel)
            # Should subtract in place below to save memory for large problems
            for i in range(m):
                local_params["cur_time"] = time[i]
                vel_from_deriv[:, i] = sdfunc(x[:, i], local_params)[:, 0]

            e = vel_from_deriv - vel
        else:
            raise ValueError(f"eqform cannot have a value of '{eqform}'")
        return e

    # --- Setup for Solver ---
    # The `hb_err` function (defined above) is a closure that captures the
    # necessary variables from this scope. We only need to define `time` here.
    time = np.linspace(0, 2 * np.pi / omega, num=x0.shape[1], endpoint=False)

    # --- Invoke the Nonlinear Solver ---
    # Select the solver function based on the user's 'method' string.
    solver = _SOLVERS.get(method)
    if not solver:
        raise ValueError(
            f"Unknown solver '{method}'. Available solvers are: {list(_SOLVERS.keys())}"
        )

    log.info("Starting harmonic balance solver '%s' for omega=%.4f", method, omega)
    # Call the solver. The solver will iteratively call `hb_err` with different
    # trial solutions `x` until the error returned by `hb_err` is minimized
    # (ideally, to zero).
    try:
        x = solver(hb_err, x0, **kwargs)
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
    # Now that we have the solution `x`, we can calculate the final error
    # and other useful quantities.
    e = hb_err(x)
    if x.shape[1] > 1:
        # Calculate the Fourier transform of the solution to get amplitudes/phases.
        xhar = fftp.fft(x) * 2 / len(time)
        # Extract amplitude and phase of the fundamental harmonic (index 1).
        amps = np.absolute(xhar[:, 1])
        phases = np.angle(xhar[:, 1])
    else:
        amps = np.zeros(x.shape[0])
        phases = np.zeros(x.shape[0])

    if realify:
        # The solution should be real-valued; discard any small imaginary part
        # that may have arisen from numerical inaccuracies.
        x = np.real(x)

    # Return the results in a structured format.
    return HarmonicBalanceSolution(time, x, e, amps, phases)
