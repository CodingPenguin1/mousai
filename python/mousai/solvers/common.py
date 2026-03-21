import logging
from typing import Literal, NamedTuple

import numpy as np
from scipy.optimize import (
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
