import logging
from typing import Any, Literal, NamedTuple

import numpy as np
import scipy.fftpack as fftp
from scipy.optimize import (
    _nonlin,
    anderson,
    broyden1,
    broyden2,
    diagbroyden,
    excitingmixing,
    linearmixing,
    newton_krylov,
)

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


def _prepare_hb_inputs(
    omega: float,
    num_harmonics: int,
    x0: np.ndarray | None,
    num_variables: int | None,
    params: dict[str, Any] | None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Validates common inputs for harmonic balance solvers and prepares the initial guess `x0`.
    """
    if params is None:
        params = {}

    if omega <= 0:
        raise ValueError("'omega' must be positive.")
    if num_harmonics < 0:
        raise ValueError("'num_harmonics' must be non-negative.")

    required_timesteps = 1 + 2 * num_harmonics

    if x0 is None:
        if num_variables is None:
            raise ValueError("Either 'x0' or 'num_variables' must be provided.")
        if num_variables <= 0:
            raise ValueError("'num_variables' must be positive.")
        log.info("No initial guess 'x0' provided. Using zeros.")
        x0 = np.zeros((num_variables, required_timesteps))
    else:
        x0 = np.asarray(x0)
        if num_variables is None:
            num_variables = x0.shape[0]
        elif num_variables != x0.shape[0]:
            raise ValueError(
                f"'num_variables' ({num_variables}) does not match the "
                f"number of rows in 'x0' ({x0.shape[0]})."
            )

        if x0.shape[1] < required_timesteps:
            log.info("Expanding 'x0' to accommodate %d harmonics.", num_harmonics)
            x_freq = fftp.fft(x0)
            x_zeros = np.zeros((x0.shape[0], required_timesteps - x0.shape[1]))
            x_freq = np.insert(x_freq, [x0.shape[1] - x0.shape[1] // 2], x_zeros, axis=1)
            x0 = fftp.ifft(x_freq) * required_timesteps / x0.shape[1]
            x0 = np.real(x0)
        elif x0.shape[1] > required_timesteps:
            log.warning(
                "'x0' has more time steps (%d) than required for %d harmonics (%d). Truncating 'x0'.",
                x0.shape[1],
                num_harmonics,
                required_timesteps,
            )
            x0 = x0[:, :required_timesteps]

    return x0, num_variables, params
