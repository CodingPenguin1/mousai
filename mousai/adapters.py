"""Adapters for compatibility between Mousai and SciPy integrators."""

import inspect


def function_to_mousai(sdfunc):
    """Convert scipy.integrate functions to Mousai form.

    The form of the function returning state derivatives is
    `sdfunc(x, t, params)` where `x` are the current states as an `n` by `1`
    array, `t` is a scalar, and `params` is a dictionary of parameters, one of
    which must be `omega`. This is inconsistent with the SciPy numerical
    integrators for good cause, but can make simultaneous usage diffucult.

    This function returns a function compatible with Mousai by using the
    inspect package to determine the form of the function being used and to
    wrap it in Mousai form.

    Args:
        sdfunc (function): function in SciPy integrator form (`odeint`_ or `solve_ivp`_)

    Returns:
        new_function (function): function in Mousai form (accepting inputs like a standard Mousai
            function)

    Notes
    -----
    .. seealso::

       * ``old_mousai_to_new_mousai``
       * ``mousai_to_odeint``
       * ``mousai_to_solve_ivp``

    .. _`odeint` : https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
    .. _`solve_ivp` : https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

    """

    sig = inspect.signature(sdfunc)

    call_parameters = list(sig.parameters.keys())

    if len(call_parameters) == 2:
        if call_parameters[0] == "t" or call_parameters[0] == "time":
            # t and x must be swapped, params available in over-scope
            def newfunction(x, t, params={}):
                return sdfunc(t, x)
        else:  # params available in overscope

            def newfunction(x, t, params={}):
                return sdfunc(x, t)
    else:
        if call_parameters[0] == "t" or call_parameters[0] == "time":
            # t and x must be swapped, params available in over-scope
            def newfunction(x, t, params={}):
                # Extract arguments from params based on sdfunc signature
                # skipping the first two (t, x)
                other_params = [params[k] for k in call_parameters[2:]]
                return sdfunc(t, x, *other_params)
        else:  # params available in overscope

            def newfunction(x, t, params={}):
                # Extract arguments from params based on sdfunc signature
                # skipping the first two (x, t)
                other_params = [params[k] for k in call_parameters[2:]]
                return sdfunc(x, t, *other_params)

    return newfunction


def old_mousai_to_new_mousai(function):
    """Return derivative function converted to new Mousai format.

    The original format for the Mousai derivative function was
    `sdfunc(x, params)`. This is inconsistent with the SciPy integration
    functions. To act more as expected, the standard from 0.4.0 on will take
    the form `sdfunc(x, t, params)`.

    Args:
        function (function): function in old Mousai form. `sdfunc(y, params)`

    Returns:
        new_sdfunc (function): function in new Mousai form. `sdfunc(y, t, params)`

    Notes
    -----
    .. seealso::

       * ``function_to_mousai``
       * ``mousai_to_odeint``
       * ``mousai_to_solve_ivp``

    """

    def new_sdfunc(x, t, params):
        params["cur_time"] = t
        return function(x, params)

    return new_sdfunc


def mousai_to_solve_ivp(sdfunc, params):
    """Return function callable from solve_ivp given Mousai sdfunc.

    Args:
        sdfunc (function): Mousai-style function returning state derivatives.
        params (dictionary): dictionary of parameters used by `sdfunc`.

    Returns:
        solve_ivp_function (function): function ordered to work with `solve_ivp`_

    Notes
    -----
    The ability to pass parameters was deprecated in the new SciPy integrators:
    `https://stackoverflow.com/questions/48245765/pass-args-for-solve-ivp-new-scipy-ode-api`
    `https://github.com/scipy/scipy/issues/8352`

    .. seealso::

       * ``function_to_mousai``
       * ``old_mousai_to_new_mousai``
       * ``mousai_to_odeint``

    """
    sig = inspect.signature(sdfunc)

    call_parameters = list(sig.parameters.keys())

    if len(call_parameters) == 2:
        sdfunc = old_mousai_to_new_mousai(sdfunc)
        print("""Warning. The two-argument form of Mousai derivsative functions
                 is deprecated.""")

    def solve_ivp_function(t, y):
        return sdfunc(y, t, params)

    return solve_ivp_function


def mousai_to_odeint(sdfunc, params):
    """Return function callable from solve_ivp given Mousai a sdfunc.

    Args:
        sdfunc (function): Mousai-style function returning state derivatives.
        params (dictionary): dictionary of parameters used by `sdfunc`.

    Returns:
        odeint_function (function): function ordered to work with `odeint`_

    Notes
    -----
    .. seealso::

       * ``function_to_mousai``
       * ``old_mousai_to_new_mousai``
       * ``mousai_to_solve_ivp``

    """
    sig = inspect.signature(sdfunc)

    call_parameters = list(sig.parameters.keys())

    if len(call_parameters) == 2:
        sdfunc = old_mousai_to_new_mousai(sdfunc)
        print("""Warning. The two-argument form of Mousai derivative ⁠⁠\
                 functions is deprecated.""")

    if "sdfunc_params" not in globals():
        print("Define your parameters in the user created `sdfunc_params`", "dictionary.")
        sdfunc_params = {}

    def odeint_function(y, t):
        return sdfunc(y, t, params=sdfunc_params)

    return odeint_function
