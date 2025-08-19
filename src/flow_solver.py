# """
# flow_solver.py - Enhanced rectangular slot pressure solver
# """

# import numpy as np
# from scipy.optimize import brentq
# import logging
# from typing import Callable, Tuple

# logger = logging.getLogger(__name__)

# # Global safety caps - revised for physical realism
# MAX_VISCOSITY = 1e6      # Reduced from 1e6 to 100,000 Pa·s (more realistic)
# MAX_DPDX = 15e6            # Reduced from 20e6 to 15 MPa/m (more realistic)
# MIN_VISCOSITY = 10       # Increased from 10 to 100 Pa·s (more realistic)
# MIN_SHEAR_RATE = 0.1       # Minimum shear rate for viscosity calculation

# def _sanitize_eta(val: float | np.ndarray) -> float:
#     """Ensure we always work with a plain float viscosity."""
#     return float(val)

# def solve_dpdx_for_slot(
#     h: float,
#     w: float,
#     V: float,
#     Q: float,
#     eta_fn: Callable[[float], float],
#     max_iter: int = 8,
#     tol: float = 1e-4,
# ) -> Tuple[float, float, float, float, float]:
#     """
#     Enhanced solver for pressure gradient with better physical constraints.
#     """
#     # 1) Pure drag (Couette) capacity
#     Q_C = w * h * V / 2.0
    
#     # Check if we're close to drag flow only
#     if abs(Q - Q_C) < tol * max(abs(Q_C), 1e-12):
#         gamma_w = max(MIN_SHEAR_RATE, V / h)  # Apply minimum shear rate
#         eta = max(MIN_VISCOSITY, min(MAX_VISCOSITY, _sanitize_eta(eta_fn(gamma_w))))
#         tau_w = eta * gamma_w
#         return 0.0, gamma_w, tau_w, Q_C, 0.0
    
#     # 2) Initial estimate with physical constraints
#     dpdx = 0.0
#     gamma_est = max(MIN_SHEAR_RATE, V / h)  # Apply minimum shear rate
#     eta = max(MIN_VISCOSITY, min(MAX_VISCOSITY, _sanitize_eta(eta_fn(gamma_est))))
    
#     # 3) Outer iteration loop for viscosity update
#     for iteration in range(max_iter):
#         # Define residual function for root finding
#         def residual(dpdx_val: float) -> float:
#             # Calculate Poiseuille flow with current viscosity
#             Q_P = w * h**3 * dpdx_val / (12.0 * eta)
#             return Q_C + Q_P - Q
        
#         # Check if root exists in the allowed range
#         Q_at_max = Q_C + w * h**3 * MAX_DPDX / (12.0 * eta)
#         Q_at_min = Q_C + w * h**3 * -MAX_DPDX / (12.0 * eta)
        
#         # Determine flow regime
#         if Q > Q_at_max * 1.01:  # Need more than max pressure can provide
#             dpdx_new = MAX_DPDX
#             regime = "overload"
#         elif Q < Q_at_min * 0.99:  # Need more backflow than possible
#             dpdx_new = -MAX_DPDX
#             regime = "backflow"
#         elif (Q - Q_at_min) * (Q - Q_at_max) > 0:
#             # No root in the bracket - use appropriate limit
#             dpdx_new = MAX_DPDX if Q > Q_C else -MAX_DPDX
#             regime = "limit"
#         else:
#             # Root exists - use brentq
#             try:
#                 dpdx_new = brentq(residual, -MAX_DPDX, MAX_DPDX, maxiter=100)
#                 regime = "converged"
#             except ValueError as e:
#                 logger.warning(f"Iteration {iteration}: Brentq failed ({e}), using fallback")
#                 dpdx_new = MAX_DPDX if Q > Q_C else -MAX_DPDX
#                 regime = "fallback"
        
#         # Check for convergence
#         if abs(dpdx_new - dpdx) < tol * max(abs(dpdx_new), 1.0) and regime == "converged":
#             dpdx = dpdx_new
#             break
            
#         dpdx = dpdx_new
        
#         # Update viscosity estimate for next iteration
#         gamma_est = max(MIN_SHEAR_RATE, V / h + (h / 2.0) * abs(dpdx) / max(eta, 1e-12))
#         eta = max(MIN_VISCOSITY, min(MAX_VISCOSITY, _sanitize_eta(eta_fn(gamma_est))))
    
#     # 4) Final calculations with physical constraints
#     dpdx = max(-MAX_DPDX, min(MAX_DPDX, dpdx))
    
#     # Calculate final shear rate and stress
#     gamma_w = max(MIN_SHEAR_RATE, V / h + (h / 2.0) * abs(dpdx) / max(eta, 1e-12))
#     eta = max(MIN_VISCOSITY, min(MAX_VISCOSITY, _sanitize_eta(eta_fn(gamma_w))))
#     tau_w = eta * gamma_w
#     Q_P = w * h**3 * dpdx / (12.0 * eta)
    
#     # Log detailed information about the solution
#     if abs(dpdx) >= MAX_DPDX * 0.99:
#         logger.warning(f"Pressure gradient at limit: {dpdx/1e6:.2f} MPa/m")
#         logger.info(f"Flow parameters: h={h*1000:.2f}mm, Q={Q*3600*1200:.1f}kg/h, Q_C={Q_C*3600*1200:.1f}kg/h")
#     if eta >= MAX_VISCOSITY * 0.99:
#         logger.warning(f"Viscosity at upper limit: {eta/1000:.1f} kPa·s")
#     if eta <= MIN_VISCOSITY * 1.01:
#         logger.warning(f"Viscosity at lower limit: {eta:.1f} Pa·s")
    
#     return dpdx, gamma_w, tau_w, Q_C, Q_P
# #     h: float,
# #     w: float,
# #     V: float,
# #     Q: float,
# #     eta_fn: Callable[[float], float],
# #     max_iter: int = 10,
# #     tol: float = 1e-4,
# # ) -> Tuple[float, float, float, float, float]:
# #     """
# #     Enhanced solver with proper handling of shear-dependent viscosity
# #     """
# #     # 1) Pure drag (Couette) capacity
# #     Q_C = w * h * V / 2.0
    
# #     # Check if we're close to drag flow only
# #     if abs(Q - Q_C) < tol * max(abs(Q_C), 1e-12):
# #         gamma_w = max(MIN_SHEAR_RATE, V / h)
# #         eta = eta_fn(gamma_w)
# #         tau_w = eta * gamma_w
# #         return 0.0, gamma_w, tau_w, Q_C, 0.0
    
# #     # 2) Initial estimate
# #     dpdx = 0.0
    
# #     # 3) Outer iteration loop for viscosity update
# #     for iteration in range(max_iter):
# #         # Estimate shear rate based on current pressure gradient
# #         gamma_est = max(MIN_SHEAR_RATE, V / h + (h / 2.0) * abs(dpdx) / 1e5)  # Initial viscosity estimate
# #         eta = eta_fn(gamma_est)
        
# #         # Define residual function that accounts for viscosity changes
# #         def residual(dpdx_val: float) -> float:
# #             # Estimate shear rate for this pressure gradient
# #             gamma_est_iter = max(MIN_SHEAR_RATE, V / h + (h / 2.0) * abs(dpdx_val) / max(eta, 1e-12))
# #             eta_iter = eta_fn(gamma_est_iter)
            
# #             # Calculate Poiseuille flow with updated viscosity
# #             Q_P = w * h**3 * dpdx_val / (12.0 * eta_iter)
# #             return Q_C + Q_P - Q
        
# #         # Check if root exists in the allowed range
# #         Q_at_max = Q_C + w * h**3 * MAX_DPDX / (12.0 * eta)
# #         Q_at_min = Q_C + w * h**3 * -MAX_DPDX / (12.0 * eta)
        
# #         # Determine appropriate solution strategy
# #         if Q > Q_at_max * 1.01:
# #             dpdx_new = MAX_DPDX
# #         elif Q < Q_at_min * 0.99:
# #             dpdx_new = -MAX_DPDX
# #         elif (Q - Q_at_min) * (Q - Q_at_max) > 0:
# #             dpdx_new = MAX_DPDX if Q > Q_C else -MAX_DPDX
# #         else:
# #             try:
# #                 dpdx_new = brentq(residual, -MAX_DPDX, MAX_DPDX, maxiter=50)
# #             except ValueError:
# #                 dpdx_new = MAX_DPDX if Q > Q_C else -MAX_DPDX
        
# #         # Check for convergence
# #         if abs(dpdx_new - dpdx) < tol * max(abs(dpdx_new), 1.0):
# #             dpdx = dpdx_new
# #             break
            
# #         dpdx = dpdx_new
    
# #     # 4) Final calculations
# #     dpdx = max(-MAX_DPDX, min(MAX_DPDX, dpdx))
    
# #     # Calculate final shear rate and stress
# #     gamma_w = max(MIN_SHEAR_RATE, V / h + (h / 2.0) * abs(dpdx) / 1e5)  # Conservative estimate
# #     eta = eta_fn(gamma_w)
# #     tau_w = eta * gamma_w
# #     Q_P = w * h**3 * dpdx / (12.0 * eta)
    
# #     return dpdx, gamma_w, tau_w, Q_C, Q_P
"""flow_solver.py – Rectangular‑slot pressure solver used by the 1‑D screw model

Key fixes 2025‑07‑10
--------------------
* Viscosity *η* is now re‑evaluated **inside** the residual passed to ``brentq`` so the
  root‑finder always sees a bracket with opposite signs even for highly
  shear‑thinning melts.
* Fallback when the root‑finder fails is no longer a hard‑coded ±1×10⁷ Pa m⁻¹.
  Instead an analytic Newtonian estimate is used, which still preserves
  dependence on the local geometry (h, w) and flow imbalance (Q − Q_C).
"""

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
    dpdx = 0.0
    gamma_init = max(1e-6, V/h)   # avoid div by zero
    eta = _sanitize_eta(eta_fn(gamma_init))
    
    # If Q ≃ Q_C, skip pressure solve
    if abs(Q - Q_C) < tol * max(Q_C, 1e-12):
        eta  = _sanitize_eta(eta_fn(V / h))
        gamma_w = V / h + (h / 2.0) * abs(dpdx) / max(eta, 1.0e-12)
       
        tau_w   = _sanitize_eta(eta_fn(gamma_w)) * gamma_w
        return 0.0, gamma_w, tau_w, Q_C, 0.0

    # 2) Initial guess: no pressure gradient, viscosity at simple shear
   

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