"""Object-oriented interface for Harmonic Balance."""

from typing import Any, Callable, Literal

import numpy as np

from .solvers import EquationForm, HarmonicBalanceSolution, SolverMethod, hb_freq, hb_time


class HarmonicBalance:
    """A class to define and solve Harmonic Balance problems.

    This class encapsulates the system definition (equation, parameters) and
    solver configuration, allowing for easier parameter sweeps and state management.

    Attributes:
        model (function): The state derivative function.
        num_harmonics (int): Number of harmonics to consider.
        num_variables (int): Number of state variables.
        eqform (str): 'first_order' or 'second_order'.
        params (dict): Physics parameters passed to the model function.
    """

    model: Callable[..., np.ndarray]
    num_harmonics: int
    num_variables: int | None
    eqform: EquationForm
    params: dict[str, Any]

    def __init__(
        self,
        model: Callable[..., np.ndarray],
        num_harmonics: int = 1,
        num_variables: int | None = None,
        eqform: EquationForm = "second_order",
        **params: Any,
    ) -> None:
        """Initialize the Harmonic Balance problem.

        Args:
            model (function): Function returning state derivatives/accelerations.
            num_harmonics (int, optional): Number of harmonics. Defaults to 1.
            num_variables (int, optional): Number of variables. inferred from x0 if None.
            eqform (str, optional): 'first_order' or 'second_order'. Defaults to 'second_order'.
            **params: Arbitrary keyword arguments to be passed to the model function.
        """
        self.model = model
        self.num_harmonics = num_harmonics
        self.num_variables = num_variables
        self.eqform = eqform
        self.params = params

    def solve(
        self,
        omega: float,
        x0: np.ndarray | None = None,
        method: SolverMethod = "newton_krylov",
        domain: Literal["time", "freq"] = "time",
        **kwargs: Any,
    ) -> HarmonicBalanceSolution:
        """Solve the system for a periodic response at a specific frequency.

        Args:
            omega (float): Fundamental frequency (rad/s).
            x0 (array_like, optional): Initial guess for states.
            method (str, optional): Optimization method (e.g., 'newton_krylov').
            domain (str, optional): 'time' for time-domain HB, 'freq' for frequency-domain HB.
            **kwargs: Additional arguments for the underlying solver.

        Returns:
            HarmonicBalanceSolution: A named tuple containing (t, x, e, amps, phases).
        """
        if domain == "freq":
            solver = hb_freq
        else:
            solver = hb_time

        # Update params with omega, as required by the underlying solvers
        solve_params = self.params.copy()
        solve_params["omega"] = omega

        return solver(
            self.model,
            x0=x0,
            omega=omega,
            method=method,
            num_harmonics=self.num_harmonics,
            num_variables=self.num_variables,
            eqform=self.eqform,
            params=solve_params,
            **kwargs,
        )
