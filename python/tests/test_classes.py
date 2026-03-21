"""Tests for the HarmonicBalance class."""

import numpy as np
import pytest
from common_systems import mdof_duffing_chain
from mousai import duff_osc
from mousai.solvers import HarmonicBalance


def test_harmonic_balance_class_time_duffing():
    """
    Test HarmonicBalance class wrapper for hb_time.
    """
    x0 = np.array([[0, 1, -1]])
    omega = 0.7

    # Initialize class
    hb = HarmonicBalance(duff_osc, num_harmonics=1)

    # Solve using time domain
    # f_tol is passed to the underlying solver via **kwargs
    t, x, e, amps, phases = hb.solve(omega, x0=x0, domain="time", f_tol=1e-12)

    # Expected output values (matches hb_time tests)
    t_expected = np.array([0.0, 2.991993, 5.98398601])
    x_expected = np.array([[-0.34996499, 1.36053998, -1.11828552]])
    amps_expected = np.array([1.4652053])
    phases_expected = np.array([-1.78681895])

    np.testing.assert_allclose(t, t_expected, rtol=1e-5)
    np.testing.assert_allclose(x, x_expected, rtol=1e-5)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    np.testing.assert_allclose(amps, amps_expected, rtol=1e-5)
    np.testing.assert_allclose(phases, phases_expected, rtol=1e-5)


def test_harmonic_balance_class_freq_duffing():
    """
    Test HarmonicBalance class wrapper for hb_freq.
    """
    x0 = np.array([[0, 1, -1]])
    omega = 0.7

    # Initialize class
    hb = HarmonicBalance(duff_osc, num_harmonics=1)

    # Solve using frequency domain
    t, x, e, amps, phases = hb.solve(omega, x0=x0, domain="freq", f_tol=1e-12)

    # Expected output values (matches hb_freq results with default anti-aliasing)
    t_expected = np.array([0.0, 2.991993, 5.98398601])
    x_expected = np.array([[-0.29877599, 1.38778831, -1.08901232]])
    amps_expected = np.array([1.46086078])
    phases_expected = np.array([-0.20597384])

    np.testing.assert_allclose(t, t_expected, rtol=1e-5)
    np.testing.assert_allclose(x, x_expected, rtol=1e-5)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    np.testing.assert_allclose(amps, amps_expected, rtol=1e-5)
    np.testing.assert_allclose(phases, phases_expected, rtol=1e-5)


def test_harmonic_balance_class_mdof():
    """Test HarmonicBalance class on MDOF system."""
    n_dof = 5
    num_harmonics = 3
    x0 = np.zeros((n_dof, 1 + 2 * num_harmonics))

    hb = HarmonicBalance(mdof_duffing_chain, num_harmonics=num_harmonics)

    # Solve
    _, _, e, _, _ = hb.solve(omega=0.8, x0=x0, f_tol=1e-8)

    assert np.allclose(e, 0, atol=1e-7)


def test_harmonic_balance_class_params():
    """Test that params passed to __init__ are correctly used."""

    # We re-use simple duffing but fail if 'test_param' isn't in params
    def parameterized_model(x, v, params):
        assert params["test_param"] == 42.0
        return duff_osc(x, v, params)

    x0 = np.array([[0, 1, -1]])
    hb = HarmonicBalance(parameterized_model, num_harmonics=1, test_param=42.0)

    # Should run without raising AssertionError
    hb.solve(omega=0.7, x0=x0, f_tol=1e-8)
