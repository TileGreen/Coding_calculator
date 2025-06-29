# flow_solver.py

import numpy as np
from scipy.optimize import brentq
import logging
from typing import Callable, Tuple

logger = logging.getLogger(__name__)

def _sanitize_eta(val: float | np.ndarray) -> float:
    """
    Ensure we always work with a plain float viscosity.
    Strips 0-D numpy arrays if they sneak through.
    """
    return float(val)

def solve_dpdx_for_slot(
    h: float,
    w: float,
    V: float,
    Q: float,
    eta_fn: Callable[[float], float],
    max_iter: int = 5,
    tol: float = 1e-3,
) -> Tuple[float, float, float, float, float]:
    """
    Solve for the pressure gradient dp/dx in a rectangular slot by
    matching the total flow (Couette + Poiseuille) to the imposed Q,
    accounting for non-Newtonian viscosity via an outer loop.

    Returns:
      dpdx   : pressure gradient [Pa/m]
      gamma_w: wall shear rate [1/s]
      tau_w  : wall shear stress [Pa]
      Q_C    : pure Couette flow rate [m^3/s]
      Q_P    : pure Poiseuille flow component at convergence [m^3/s]
    """
    # 1) Pure drag‐only (Couette) capacity
    Q_C = w * h * V / 2.0

    # If Q ≃ Q_C, skip pressure solve
    if abs(Q - Q_C) < tol * max(Q_C, 1e-12):
        gamma_w = V / h
        tau_w   = _sanitize_eta(eta_fn(gamma_w)) * gamma_w
        return 0.0, gamma_w, tau_w, Q_C, 0.0

    # 2) Initial guess: no pressure gradient, viscosity at simple shear
    dpdx = 0.0
    eta  = _sanitize_eta(eta_fn(V / h))

    # 3) Outer loop: update viscosity based on wall shear → re-solve dpdx
    for _ in range(max_iter):
        # Estimate shear rate at the wall including pressure‐driven correction
        gamma_est = V/h + (h/2.0) * abs(dpdx) / max(eta, 1e-12)
        eta       = _sanitize_eta(eta_fn(gamma_est))

        # Residual: total flow (Couette + Poiseuille) minus imposed Q
        def residual(dpdx_val: float) -> float:
            Q_P = w * h**3 * dpdx_val / (12.0 * eta)
            return Q_C + Q_P - Q

        # 4) Solve for dpdx via Brent’s method
        try:
            # expand bracket in case you need more stroke
            dpdx_new = brentq(residual, -1e12, 1e12)
        except ValueError as e:
            logger.warning(
                f"[flow_solver] brentq failed ({e}), using fallback dpdx"
            )
            # pick an extreme gradient with the correct sign:
            dpdx = -1e7 if Q < Q_C else 1e7
            break

        # Check convergence
        if abs(dpdx_new - dpdx) < tol * max(abs(dpdx_new), 1.0):
            dpdx = dpdx_new
            break
        dpdx = dpdx_new

    # 5) Final wall shear rate & stress at converged dpdx
    gamma_w = V/h + (h/2.0) * abs(dpdx) / max(eta, 1e-12)
    tau_w   = _sanitize_eta(eta_fn(gamma_w)) * gamma_w
    Q_P     = w * h**3 * dpdx / (12.0 * eta)

    return dpdx, gamma_w, tau_w, Q_C, Q_P
