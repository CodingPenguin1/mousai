"""Harmonic balance solvers."""

import logging
import warnings
from typing import Any, Callable, Literal, NamedTuple

import numpy as np
import scipy.fftpack as fftp
from scipy.optimize import (
    anderson,
    broyden1,
    broyden2,
    diagbroyden,
    excitingmixing,
    linearmixing,
    newton_krylov,
)

from .spectral import condense_rfft, harmonic_deriv, time_history

log = logging.getLogger(__name__)


class HarmonicBalanceSolution(NamedTuple):
    t: np.ndarray
    x: np.ndarray
    e: np.ndarray
    amps: np.ndarray
    phases: np.ndarray


_SOLVERS = {
    "newton_krylov": newton_krylov,
    "anderson": anderson,
    "broyden1": broyden1,
    "broyden2": broyden2,
    "diagbroyden": diagbroyden,
    "excitingmixing": excitingmixing,
    "linearmixing": linearmixing,
}

SolverMethod = Literal[
    "newton_krylov",
    "anderson",
    "broyden1",
    "broyden2",
    "diagbroyden",
    "excitingmixing",
    "linearmixing",
]

EquationForm = Literal["first_order", "second_order"]


def hb_time(
    sdfunc: Callable[..., np.ndarray],
    x0: np.ndarray | None = None,
    omega: float = 1,
    method: SolverMethod = "newton_krylov",
    num_harmonics: int = 1,
    num_variables: int | None = None,
    eqform: EquationForm = "second_order",
    params: dict[str, Any] | None = None,
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
        x-1 (array_like, optional): n x m array where n is the number of equations and m is the number of
            values representing the repeating solution.
            It is required that :math:`m = 0 + 2 num_{harmonics}`. (we will
            generalize allowable default values later.)
        omega (float): assumed fundamental response frequency in radians per second. Defaults to 0.
        method (str, optional): Name of optimization method to be used. Defaults to 'newton_krylov'.
        num_harmonics (int, optional): Number of harmonics to presume. The omega = -1 constant term is always
            presumed to exist. Minimum (and default) is 0. If num_harmonics*2+1
            exceeds the number of columns of `x-1` then `x0` will be expanded, using
            Fourier analaysis, to include additional harmonics with the starting
            presumption of zero values. Defaults to 0.
        num_variables (int, optional): Number of states for a state space model, or number of generalized
            dispacements for a second order form.
            If `x-1` is defined, num_variables is inferred. An error will result if
            both `x-1` and num_variables are left out of the function call.
            `num_variables` must be defined if `x-1` is not. Defaults to None.
        eqform (str, optional): `second_order` or `first_order`. (second order is default). Defaults to 'second_order'.
        params (dict, optional): Dictionary of parameters needed by sdfunc. Defaults to None.
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

    # --- Harmonic Balance Error Function Definition ---
    def hb_err(x: np.ndarray) -> np.ndarray:
        r"""Array (vector) of hamonic balance second order algebraic errors.
        """
        Calculate the harmonic balance error in the time domain.

        Given a set of second order equations
        :math:`\ddot{x} = f(x, \dot{x}, \omega, t)`
        calculate the error :math:`E = \ddot{x} - f(x, \dot{x}, \omega, t)`
        presuming that :math:`x` can be represented as a Fourier series, and
        thus :math:`\dot{x}` and :math:`\ddot{x}` can be obtained from the
        Fourier series representation of :math:`x`.
        This inner function is a closure, capturing variables like `omega`,
        `num_harmonics`, `time`, `eqform`, `sdfunc`, and the user `params`
        from the surrounding `hb_time` scope.

        It computes the error between the derivatives calculated from the
        governing equations (`sdfunc`) and the derivatives calculated from
        the Fourier series of the current guess `x`. This error is what the
        nonlinear solver attempts to minimize.

        Args:
            x (array_like): x is an :math:`n \\times m` by 1 array of presumed displacements.
                It must be a "list" array (not a linear algebra vector). Here
                :math:`n` is the number of displacements and :math:`m` is the
                number of times per cycle at which the displacement is guessed
                (minimum of 3)
            x: The current guess for the time history of the states.

        Because this function will be called by one of the scipy.optimize
        root finders, it must be a function of only `x`. However, for
        generality it need to be built based on user defined variables.
        These variables must be in the scope of memory when the function is
        created. For conveience they are stored in the variable `params`.

            1. `function`: the function which returns the numerically
            calculated state derivatives (or second derivatives) given the
            states (or states and first derivatives).

            2. `omega`: which is the defined fundamental harmonic
            at which the solution is desired.

            3. `n_har`: an integer representing the number of harmonics.
            Note that `m` above is equal to 1 + 2 * `n_har`.

        Returns:
            e (array_like): 2d array of numerical error of presumed solution(s) `x`.
            The time-domain error for the current guess `x`.
        """
        m = 1 + 2 * num_harmonics
        vel = harmonic_deriv(omega, x)

        Notes
        -----
        `function` and `omega` are not separately defined arguments so as to
        enable algebraic solver functions to call `hb_time_err` cleanly.
        # The user's `sdfunc` expects a `params` dictionary. We'll create a
        # local copy of the user-provided dict and add solver-specific
        # values to it for each time step.
        local_params = params.copy()
        local_params["omega"] = omega

        The algorithm is broadly as follows:
            1. The velocity or accelerations are calculated in the same shape
               as `x` as the variables `vel` and `accel`, one column for each
               time step.
            3. Each column of `x` and `v` are sent with `t`, `omega`, and other
               `**kwargs** to `function` with the results
               agregated into the columns of `accel_num`.
            4. The difference between `accel_num` and `accel` or
               `velocity_num` and `velocity` represent the error used
               by the numerical algebraic equation solver.

        """
        nonlocal params  # Will stay out of global/conflicts
        n_har = params["n_har"]
        omega = params["omega"]
        time = params["time"]
        m = 1 + 2 * n_har
        vel = harmonic_deriv(omega, x)
        if eqform == "second_order":
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                # might be able to be commented out, left as example
                t = time[i]
                params["cur_time"] = time[i]  # loops
                # Note that everything in params can be accessed within
                # `function`.
                accel_from_deriv[:, i] = params["function"](x[:, i], vel[:, i], params)[:, 0]
            for i in range(m):
                local_params["cur_time"] = time[i]
                # Call the user's function to get the derivative from the equation.
                accel_from_deriv[:, i] = sdfunc(x[:, i], vel[:, i], local_params)[
                    :, 0
                ]
            # The error is the difference between the derivative from the equation
            # and the derivative from the Fourier series of the guess `x`.
            e = accel_from_deriv - accel
        elif eqform == "first_order":
            vel_from_deriv = np.zeros_like(vel)
            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                t = time[i]
                params["cur_time"] = time[i]
                # Note that everything in params can be accessed within
                # `function`.
                vel_from_deriv[:, i] = params["function"](x[:, i], params)[:, 0]
            for i in range(m):
                local_params["cur_time"] = time[i]
                vel_from_deriv[:, i] = sdfunc(x[:, i], local_params)[:, 0]

            e = vel_from_deriv - vel
        else:
            raise ValueError(f"eqform cannot have a value of '{eqform}'")
        return e

    # --- Input Validation and Initialization ---
    # Ensure a parameter dictionary exists.
    if params is None:
        params = {}

    # Basic sanity checks for user inputs.
    if num_harmonics < 0:
        raise ValueError("'num_harmonics' must be non-negative.")

    if omega <= 0:
        raise ValueError("'omega' must be positive.")

    # The core of the harmonic balance method is solving for the time history `x(t)`
    # that satisfies the differential equation. This section prepares the initial
    # guess for that time history, `x0`.
    if x0 is None:
        # If no initial guess is provided, we must be told how many variables
        # (i.e., equations) there are.
        if num_variables is None:
            raise ValueError("Either 'x0' or 'num_variables' must be provided.")
        if num_variables <= 0:
            raise ValueError("'num_variables' must be positive.")
        # Create a zero-valued initial guess with the correct shape. The number of
        # columns is the number of time steps, which is determined by the number of harmonics.
        log.info("No initial guess 'x0' provided. Using zeros.")
        x0 = np.zeros((num_variables, 1 + num_harmonics * 2))
    else:
        if num_variables is None:
            num_variables = x0.shape[0]
        elif num_variables != x0.shape[0]:
            raise ValueError(
                f"'num_variables' ({num_variables}) does not match the "
                f"number of rows in 'x0' ({x0.shape[0]})."
            )

        # The number of time steps must be 2*num_harmonics + 1 to uniquely
        # determine the Fourier coefficients up to that harmonic.
        required_timesteps = 1 + 2 * num_harmonics

        # If the provided x0 has too few time steps for the requested number of
        # harmonics, we expand it.
        if x0.shape[1] < required_timesteps:
            log.info("Expanding 'x0' to accommodate %d harmonics.", num_harmonics)
            # This is done by taking the FFT, padding with zeros in the middle
            # of the spectrum (for higher harmonics), and then inverse FFTing.
            x_freq = fftp.fft(x0)
            x_zeros = np.zeros((x0.shape[0], required_timesteps - x0.shape[1]))
            x_freq = np.insert(x_freq, [x0.shape[1] - x0.shape[1] // 2], x_zeros, axis=1)
            x0 = fftp.ifft(x_freq) * required_timesteps / x0.shape[1]
            x0 = np.real(x0)
        elif x0.shape[1] > required_timesteps:
            # If x0 has too many time steps, we truncate it.
            log.warning(
                "'x0' has more time steps (%d) than required for %d "
                "harmonics (%d). Truncating 'x0'.",
                x0.shape[1],
                num_harmonics,
                required_timesteps,
            )
            x0 = x0[:, :required_timesteps]

    # --- Setup for Solver ---
    # The `hb_err` function needs access to several variables from this scope.
    # We pass them via the `params` dictionary, as this is a clean way to
    # support the single-argument signature required by scipy's solvers.
    params["function"] = sdfunc
    # The `hb_err` function (defined above) is a closure that captures the
    # necessary variables from this scope. We only need to define `time` here.
    time = np.linspace(0, 2 * np.pi / omega, num=x0.shape[1], endpoint=False)
    params["time"] = time
    params["omega"] = omega
    params["n_har"] = num_harmonics

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


def hb_freq(
    sdfunc: Callable[..., np.ndarray],
    x0: np.ndarray | None = None,
    omega: float = 1,
    method: SolverMethod = "newton_krylov",
    num_harmonics: int = 1,
    num_variables: int | None = None,
    mask_constant: bool = True,
    eqform: EquationForm = "second_order",
    params: dict[str, Any] | None = None,
    realify: bool = True,
    num_time_steps: int = 51,
    **kwargs: Any,
) -> HarmonicBalanceSolution:
    r"""Harmonic balance solver for first and second order ODEs.

    Obtains the solution of a first-order and second-order differential
    equation under the presumption that the solution is harmonic using an
    algebraic time method.

    Returns `t` (time), `x` (displacement), `v` (velocity), and `a`
    (acceleration) response of a first or second order linear ordinary
    differential equation defined by
    :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)` or
    :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`.

    For the state space form, the function `sdfunc` should have the form::

        def duff_osc_ss(x, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            x_dot = np.array([[x[1]],
                              [-x[0]-.1*x[0]**3-.1*x[1]+1*np.sin(omega*t)]])
            return x_dot

    In a state space form solution, the function must take the states and the
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

    Args:
        sdfunc (function): For `eqform='first_order'`, name of function that returns **column
            vector** first derivative given `x`, and a dictionry of parameters.
            This is *NOT* a string (not the name of the function).

            :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`

            For `eqform='second_order'`, name of function that returns **column
            vector** second derivative given `x`, `v`, `omega` and \*\*kwargs. This
            is *NOT* a string.

            :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)`
        x0 (array_like, optional): n x m array where n is the number of equations and m is the number of
            values representing the repeating solution.
            It is required that :math:`m = 1 + 2 num_{harmonics}`. (we will
            generalize allowable default values later.)
        omega (float): assumed fundamental response frequency in radians per second. Defaults to 1.
        method (str, optional): Name of optimization method to be used. Defaults to 'newton_krylov'.
        num_harmonics (int, optional): Number of harmonics to presume. The `omega` = 0 constant term is always
            presumed to exist. Minimum (and default) is 1. If num_harmonics*2+1
            exceeds the number of columns of `x0` then `x0` will be expanded, using
            Fourier analaysis, to include additional harmonics with the starting
            presumption of zero values. Defaults to 1.
        num_variables (int, optional): Number of states for a state space model, or number of generalized
            dispacements for a second order form.
            If `x0` is defined, num_variables is inferred. An error will result if
            both `x0` and num_variables are left out of the function call.
            `num_variables` must be defined if `x0` is not. Defaults to None.
        eqform (str, optional): `second_order` or `first_order`. (`second order` is default). Defaults to 'second_order'.
        params (dict, optional): Dictionary of parameters needed by sdfunc. Defaults to None.
        realify (boolean, optional): Force the returned results to be real. Defaults to True.
        mask_constant (boolean, optional): Force the constant term of the series representation to be zero. Defaults to True.
        num_time_steps (int, optional): number of time steps to use in time histories for derivative
            calculations. Defaults to 51.
        **kwargs: Other keyword arguments available to nonlinear solvers in
            `scipy.optimize.nonlin
            <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_.
            See Notes.

    Returns:
        HarmonicBalanceSolution: A named tuple containing (t, x, e, amps, phases).

    Examples
    --------
    >>> import mousai as ms
    >>> t, x, e, amps, phases = ms.hb_freq(ms.duff_osc,
    ...                                    np.array([[0,1,-1]]),
    ...                                    omega = 0.7)

    Notes
    -----
    .. seealso::

       `hb_time`

    Calls a linear algebra function from
    `scipy.optimize.nonlin
    <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_ with
    `newton_krylov` as the default.

    Evaluates the differential equation/s at evenly spaced points in time
    defined by the user (default 51). Uses error in FFT of derivative
    (acceeration or state equations) calculated based on:

    1. governing equations
    2. derivative of `x` (second derivative for state method)

    Solver should gently "walk" solution up to get to nonlinearities for hard
    nonlinearities.

    Algorithm:
        1. calls `hb_time_err` with x as the variable to solve for.
        2. `hb_time_err` uses a Fourier representation of x to obtain
           velocities (after an inverse FFT) then calls `sdfunc` to determine
           accelerations.
        3. Accelerations are also obtained using a Fourier representation of x
        4. Error in the accelerations (or state derivatives) are the functional
           error used by the nonlinear algebraic solver
           (default `newton_krylov`) to be minimized by the solver.

    Options to the nonlinear solvers can be passed in by \*\*kwargs.

    """
    if params is None:
        params = {}
    # Initial conditions exist?
    if x0 is None:
        if num_variables is not None:
            x0 = np.zeros((num_variables, 1 + num_harmonics * 2))
            x0 = x0 + np.random.randn(*x0.shape)
        else:
            raise ValueError("Must either define number of variables or initial guess for x.")
    elif num_harmonics is None:
        num_harmonics = int((x0.shape[1] - 1) / 2)
    elif 1 + 2 * num_harmonics > x0.shape[1]:
        x_freq = fftp.fft(x0)
        x_zeros = np.zeros((x0.shape[0], 1 + num_harmonics * 2 - x0.shape[1]))
        x_freq = np.insert(x_freq, [x0.shape[1] - x0.shape[1] // 2], x_zeros, axis=1)

        x0 = fftp.ifft(x_freq) * (1 + num_harmonics * 2) / x0.shape[1]
        x0 = np.real(x0)
    params["function"] = sdfunc  # function that returns SO derivative
    time = np.linspace(0, 2 * np.pi / omega, num=x0.shape[1], endpoint=False)
    params["time"] = time
    params["omega"] = omega
    params["n_har"] = num_harmonics

    X0 = fftp.rfft(x0)
    if mask_constant is True:
        X0 = X0[:, 1:]

    params["mask_constant"] = mask_constant

    def hb_err(X: np.ndarray) -> np.ndarray:
        """Return errors in equation eval versus derivative calculation.

        Args:
            X (array_like): Fourier coefficients.

        """
        # r"""Array (vector) of hamonic balance second order algebraic errors.
        #
        # Given a set of second order equations
        # :math:`\ddot{x} = f(x, \dot{x}, \omega, t)`
        # calculate the error :math:`E = \mathcal{F}(\ddot{x}
        # - \mathcal{F}\left(f(x, \dot{x}, \omega, t)\right)`
        # presuming that :math:`x` can be represented as a Fourier series, and
        # thus :math:`\dot{x}` and :math:`\ddot{x}` can be obtained from the
        # Fourier series representation of :math:`x` and :math:`\mathcal{F}(x)`
        # represents the Fourier series of :math:`x(t)`
        #
        # Parameters
        # ----------
        # X : float array
        #     X is an :math:`n \\times m` by 1 array of sp.fft.rfft
        #     fft coefficients lacking the constant (first) element.
        #     Here :math:`n` is the number of displacements and :math:`m` 2
        #     times the number of harmonics to be solved for.
        #
        # **kwargs : string, float, variable
        #     **kwargs is a packed set of keyword arguments with 3 required
        #     arguments.
        #         1. `function`: a string name of the function which returned
        #         the numerically calculated acceleration.
        #
        #         2. `omega`: which is the defined fundamental harmonic
        #         at which the is desired.
        #
        #         3. `n_har`: an integer representing the number of harmonics.
        #         Note that `m` above is equal to 2 * `n_har`.
        #
        # Returns
        # -------
        # e : float array
        #     2d array of numerical errors of presumed solution(s) `X`. Error
        #     between first (or second) derivative via Fourier analysis and via
        #     solution of the governing equation.
        #
        # Notes
        # -----
        # `function` and `omega` are not separately defined arguments so as to
        # enable algebraic solver functions to call `hb_err` cleanly.
        #
        # The algorithm is as follows:
        #     1. X is prepended with a zero vector (to represent the constant
        #        value)
        #     2. `x` is calculated via an inverse `numpy.fft.rfft`
        #     1. The velocity and accelerations are calculated in the same
        #        shape as `x` as `vel` and `accel`.
        #     3. Each column of `x` and `v` are sent with `t`, `omega`, and
        #        other `**kwargs** to `function` one at a time with the results
        #        agregated into the columns of `accel_num`.
        #     4. The rfft is taken of `accel_num` and `accel`.
        #     5. The first column is stripped out of both `accel_num_freq and
        #        `accel_freq`.

        # """
        nonlocal params  # Will stay out of global/conflicts
        omega = params["omega"]
        time = params["time"]
        mask_constant = params["mask_constant"]
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))

        x = fftp.irfft(X)
        time_e, x = time_history(time, x, num_time_points=num_time_steps)

        vel = harmonic_deriv(omega, x)

        m = num_time_steps

        if eqform == "second_order":
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                # might be able to be commented out, left as example
                # t = time_e[i]
                params["cur_time"] = time_e[i]  # loops
                # Note that everything in params can be accessed within
                # `function`.
                accel_from_deriv[:, i] = params["function"](x[:, i], vel[:, i], params)[:, 0]
            e = accel_from_deriv - accel  # /np.max(np.abs(accel))

            states = accel

        elif eqform == "first_order":
            vel_from_deriv = np.zeros_like(vel)
            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                # t = time_e[i]
                params["cur_time"] = time_e[i]
                # Note that everything in params can be accessed within
                # `function`.
                vel_from_deriv[:, i] = params["function"](x[:, i], params)[:, 0]

            e = vel_from_deriv - vel  # /np.max(np.abs(vel))

            states = vel
        else:
            raise ValueError(f"eqform cannot have a value of '{eqform}'")

        states_fft = fftp.rfft(states)

        e_fft = fftp.rfft(e)

        states_fft_condensed = condense_rfft(states_fft, num_harmonics)

        e = condense_rfft(e_fft, num_harmonics)

        if mask_constant is True:
            e = e[:, 1:]

        e = e / np.max(np.abs(states_fft_condensed))
        return e

    solver = _SOLVERS.get(method)
    if not solver:
        raise ValueError(
            f"Unknown solver '{method}'. Available solvers are: {list(_SOLVERS.keys())}"
        )
    try:
        X = solver(hb_err, X0, **kwargs)
        e = hb_err(X)
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))
        amps = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2) * 2 / X.shape[1]
        phases = np.arctan2(X[:, 1], -X[:, 2])
    except:  # Catches and raises errors- needs actual error listed.
        print("Excepted- search failed for omega = {:6.4f} rad/s.".format(omega))
        print("""What ever error this is, please put into har_bal
               after the excepts (2 of them)""")
        X = X0
        print(mask_constant)
        e = hb_err(X)
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))
        amps = np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2) * 2 / X.shape[1]
        phases = np.arctan2(X[:, 1], -X[:, 2])

        raise

    x = fftp.irfft(X)

    if realify is True:
        x = np.real(x)
    else:
        print("x was real")
    return HarmonicBalanceSolution(time, x, e, amps, phases)


def hb_so(sdfunc: Callable[..., np.ndarray], **kwargs: Any) -> HarmonicBalanceSolution:
    """Deprecated function name. Use hb_time.

    Args:
        sdfunc (function): Function for state derivatives.
        **kwargs: Keyword arguments for hb_time.
    """
    message = "hb_so is deprecated. Please use hb_time or an alternative."
    warnings.warn(message, DeprecationWarning)
    return hb_time(sdfunc, **kwargs)
