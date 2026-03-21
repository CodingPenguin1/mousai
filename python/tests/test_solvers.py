"""Tests for the mousai solvers module."""

import numpy as np
import pytest
from mousai import duff_osc, solvers


def test_hb_time_duffing_oscillator():
    """
    Test hb_time solver against the Duffing oscillator example.

    Verifies that the solution matches expected values for time, displacement,
    error, amplitude, and phase.
    """
    x0 = np.array([[0, 1, -1]])
    omega = 0.7

    # Run the harmonic balance time domain solver
    # f_tol=1e-12 is passed to the underlying scipy solver (newton_krylov)
    t, x, e, amps, phases = solvers.hb_time(duff_osc, x0=x0, omega=omega, f_tol=1e-12)

    # Expected output values
    t_expected = np.array([0.0, 2.991993, 5.98398601])
    x_expected = np.array([[-0.34996499, 1.36053998, -1.11828552]])
    amps_expected = np.array([1.4652053])
    phases_expected = np.array([-1.78681895])

    np.testing.assert_allclose(t, t_expected, rtol=1e-5)
    np.testing.assert_allclose(x, x_expected, rtol=1e-5)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    np.testing.assert_allclose(amps, amps_expected, rtol=1e-5)
    np.testing.assert_allclose(phases, phases_expected, rtol=1e-5)
