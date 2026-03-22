"""Backward compatibility module for Harmonic Balance solvers."""
import warnings

from .adapters import *
from .models import *
from .solvers import *
from .spectral import *

warnings.warn(
    "mousai.har_bal is deprecated. Use mousai.solvers, mousai.spectral, etc.", DeprecationWarning
)
