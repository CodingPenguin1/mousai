import numpy as np


def mdof_duffing_chain(x: np.ndarray, v: np.ndarray, params: dict) -> np.ndarray:
    """A chain of N masses connected by linear dampers and cubic springs."""
    N = x.shape[0]
    omega = params["omega"]
    t = params["cur_time"]
    c = params.get("damping", 0.1)
    k = params.get("stiffness", 1.0)
    knl = params.get("cubic", 0.1)
    F = params.get("force", 1.0)

    x_aug = np.concatenate(([0], x.flatten(), [0]))
    d_left = x_aug[1:-1] - x_aug[0:-2]
    d_right = x_aug[1:-1] - x_aug[2:]

    f_elastic = -(k * d_left + knl * d_left**3) - (k * d_right + knl * d_right**3)
    f_damping = -c * v.flatten()

    f_ext = np.zeros(N)
    f_ext[0] = F * np.cos(omega * t)

    a = f_elastic + f_damping + f_ext
    return a.reshape(-1, 1)


def clearance_oscillator(x: np.ndarray, v: np.ndarray, params: dict) -> np.ndarray:
    """Oscillator with a piecewise linear spring (clearance/impact)."""
    omega = params["omega"]
    t = params["cur_time"]
    gap = 1.0
    k_linear = 1.0
    k_impact = 100.0
    c = 0.1

    x_val = x[0] if isinstance(x, np.ndarray) else x
    v_val = v[0] if isinstance(v, np.ndarray) else v

    force_stiffness = -k_linear * x_val
    if abs(x_val) > gap:
        force_stiffness -= k_impact * (x_val - np.sign(x_val) * gap)

    force_damping = -c * v_val
    force_drive = 5.0 * np.cos(omega * t)  # Increased force to ensure impact

    accel = force_stiffness + force_damping + force_drive
    return np.array([[accel]])
