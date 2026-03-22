"""Tests for the mousai hb_freq solver."""

import numpy as np
import pytest
from common_systems import clearance_oscillator, mdof_duffing_chain
from mousai import duff_osc, solvers


def test_hb_freq_duffing_oscillator():
    """
    Test hb_freq solver against the Duffing oscillator example.
    """
    x0 = np.array([[0, 1, -1]])
    omega = 0.7
    num_harmonics = 1

    t, x, e, amps, phases = solvers.hb_freq(
        duff_osc,
        x0=x0,
        omega=omega,
        num_harmonics=num_harmonics,
        f_tol=1e-12,
    )

    t_expected = np.array([0.0, 2.991993, 5.98398601])
    x_expected = np.array([[-0.29877599, 1.38778831, -1.08901232]])
    amps_expected = np.array([1.46086078])
    phases_expected = np.array([-0.20597384])

    np.testing.assert_allclose(t, t_expected, rtol=1e-5)
    np.testing.assert_allclose(x, x_expected, rtol=1e-5)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    np.testing.assert_allclose(amps, amps_expected, rtol=1e-5)
    np.testing.assert_allclose(phases, phases_expected, rtol=1e-5)


def test_hb_freq_mdof_chain():
    """Tests the solver on a multi-degree-of-freedom system."""
    n_dof = 5
    num_harmonics = 3
    x0 = np.zeros((n_dof, 1 + 2 * num_harmonics))
    _, _, e, _, _ = solvers.hb_freq(
        mdof_duffing_chain, x0=x0, omega=0.8, num_harmonics=num_harmonics, f_tol=1e-8
    )
    assert np.allclose(e, 0, atol=1e-7)


@pytest.mark.slow
def test_hb_freq_clearance_oscillator():
    """Tests the solver on a non-smooth problem (impact/clearance)."""
    num_harmonics = 15
    x0 = np.zeros((1, 1 + 2 * num_harmonics))
    _, _, e, _, _ = solvers.hb_freq(
        clearance_oscillator, x0=x0, omega=0.5, num_harmonics=num_harmonics, f_tol=1e-6
    )
    assert np.allclose(e, 0, atol=1e-3)
