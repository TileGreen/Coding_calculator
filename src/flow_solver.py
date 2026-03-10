from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class FlowSolveResult:
    dpdx: float
    status: str
    residual_low: float
    residual_high: float
    q_low: float
    q_high: float
    q_target: float
    hit_bound: bool
    bracket_low: float
    bracket_high: float


# ───────────── Global safety caps (tune as needed) ─────────────

MAX_VISCOSITY: float = 1e5      # Pa·s
MIN_VISCOSITY: float = 1e2      # Pa·s
MIN_SHEAR_RATE: float = 0.1     # 1/s
MAX_DPDX: float = 15e6          # Pa/m


# ─────────────────── Helpers ────────────────────

def _sanitize_eta(val: float | np.ndarray) -> float:
    """
    Cast viscosity to a plain float and enforce global bounds.
    """
    eta = float(val)

    if not np.isfinite(eta) or eta <= 0.0:
        eta = MIN_VISCOSITY

    eta = max(MIN_VISCOSITY, min(MAX_VISCOSITY, eta))
    return eta


# ─────────────────── Core slot solver ────────────────────

def solve_dpdx_for_slot(
    h: float,
    w: float,
    V: float,
    Q: float,
    eta_fn: Callable[[float], float],
    max_iter: int = 8,
    tol: float = 1e-4,
) -> Tuple[float, float, float, float, float, str, float, float]:
    """
    Solve for the pressure gradient (dp/dx) required to achieve a target
    volumetric flow rate through a rectangular slot channel.

    Numerical strategy
    ------------------
    A semi-iterative outer loop updates viscosity from an estimated wall
    shear rate, while the inner solve uses Brent's method for dp/dx at the
    current viscosity.

    Returns
    -------
    tuple
        (dpdx, gamma_w, tau_w, Q_C, Q_P, status, Q_min, Q_max)

    Status meanings
    ---------------
    pure_drag
        Target flow is essentially equal to the Couette-only contribution.
    root_found
        A valid root was bracketed and solved.
    target_above_window
        Requested flow exceeds the current achievable window at ±MAX_DPDX.
    target_below_window
        Requested flow is below the current achievable window at ±MAX_DPDX.
    fallback
        Brent failed unexpectedly and a saturated dp/dx was returned.
    """

    # 1) Pure drag (Couette) capacity
    Q_C = w * h * V / 2.0

    if abs(Q - Q_C) < tol * max(abs(Q_C), 1e-12):
        gamma_w = max(MIN_SHEAR_RATE, V / h)
        eta = _sanitize_eta(eta_fn(gamma_w))
        tau_w = eta * gamma_w
        return (0.0, gamma_w, tau_w, Q_C, 0.0, "pure_drag", Q_C, Q_C)

    # 2) Initial guess
    dpdx = 0.0
    gamma_est = max(MIN_SHEAR_RATE, V / h)
    eta = _sanitize_eta(eta_fn(gamma_est))

    # Initialize so they are always defined even if max_iter is changed oddly.
    status = "fallback"
    Q_min = Q_C - w * h**3 * MAX_DPDX / (12.0 * eta)
    Q_max = Q_C + w * h**3 * MAX_DPDX / (12.0 * eta)

    # 3) Outer iteration
    for _ in range(max_iter):
        gamma_est = max(
            MIN_SHEAR_RATE,
            V / h + (h / 2.0) * abs(dpdx) / max(eta, 1e-12),
        )
        eta = _sanitize_eta(eta_fn(gamma_est))

        # Achievable flow window for the current viscosity estimate.
        Q_max = Q_C + w * h**3 * MAX_DPDX / (12.0 * eta)
        Q_min = Q_C - w * h**3 * MAX_DPDX / (12.0 * eta)

        if Q > Q_max * 1.01:
            dpdx_new = MAX_DPDX
            status = "target_above_window"

        elif Q < Q_min * 0.99:
            dpdx_new = -MAX_DPDX
            status = "target_below_window"

        else:

            def residual(dpdx_val: float) -> float:
                q_p = w * h**3 * dpdx_val / (12.0 * eta)
                return Q_C + q_p - Q

            try:
                dpdx_new = brentq(residual, -MAX_DPDX, MAX_DPDX, maxiter=100)
                status = "root_found"
            except ValueError as e:
                logger.warning(
                    f"[flow_solver] brentq failed ({e}); using saturated dp/dx"
                )
                dpdx_new = MAX_DPDX if Q > Q_C else -MAX_DPDX
                status = "fallback"

        if status == "root_found" and abs(dpdx_new - dpdx) < tol * max(abs(dpdx_new), 1.0):
            dpdx = dpdx_new
            break

        dpdx = dpdx_new

    # 4) Final shear / stress using the final clamped dpdx
    dpdx = max(-MAX_DPDX, min(MAX_DPDX, dpdx))

    gamma_w = max(
        MIN_SHEAR_RATE,
        V / h + (h / 2.0) * abs(dpdx) / max(eta, 1e-12),
    )
    eta = _sanitize_eta(eta_fn(gamma_w))
    tau_w = eta * gamma_w
    Q_P = w * h**3 * dpdx / (12.0 * eta)

    if abs(dpdx) >= 0.99 * MAX_DPDX:
        logger.warning(
            f"[flow_solver] Pressure gradient at limit: {dpdx / 1e6:.2f} MPa/m ({status})"
        )

    return (dpdx, gamma_w, tau_w, Q_C, Q_P, status, Q_min, Q_max)


# ─────────────────── Generic bracket-aware solver ────────────────────

def solve_zone_dpdx(
    q_target: float,
    flow_at_dpdx: Callable[[float], float],
    dpdx_min: float = -15e6,
    dpdx_max: float = 15e6,
    debug: bool = False,
) -> FlowSolveResult:
    """
    Solve for pressure gradient dp/dx such that flow_at_dpdx(dpdx) = q_target.

    This is a generic helper for any caller that can provide a callable
    `flow_at_dpdx(dpdx)`.
    """

    def residual(dpdx_val: float) -> float:
        return flow_at_dpdx(dpdx_val) - q_target

    q_low = flow_at_dpdx(dpdx_min)
    q_high = flow_at_dpdx(dpdx_max)
    r_low = q_low - q_target
    r_high = q_high - q_target

    if math.isclose(r_low, 0.0, abs_tol=1e-12):
        if debug:
            print(
                f"[flow_solver] lower bound exact hit: "
                f"dpdx={dpdx_min / 1e6:.2f} MPa/m, q_low={q_low:.6g}, q_target={q_target:.6g}"
            )
        return FlowSolveResult(
            dpdx=dpdx_min,
            status="lower_bound_hit",
            residual_low=r_low,
            residual_high=r_high,
            q_low=q_low,
            q_high=q_high,
            q_target=q_target,
            hit_bound=True,
            bracket_low=dpdx_min,
            bracket_high=dpdx_max,
        )

    if math.isclose(r_high, 0.0, abs_tol=1e-12):
        if debug:
            print(
                f"[flow_solver] upper bound exact hit: "
                f"dpdx={dpdx_max / 1e6:.2f} MPa/m, q_high={q_high:.6g}, q_target={q_target:.6g}"
            )
        return FlowSolveResult(
            dpdx=dpdx_max,
            status="upper_bound_hit",
            residual_low=r_low,
            residual_high=r_high,
            q_low=q_low,
            q_high=q_high,
            q_target=q_target,
            hit_bound=True,
            bracket_low=dpdx_min,
            bracket_high=dpdx_max,
        )

    if r_low * r_high > 0.0:
        if abs(r_low) <= abs(r_high):
            dpdx_fallback = dpdx_min
        else:
            dpdx_fallback = dpdx_max

        if r_low > 0.0 and r_high > 0.0:
            status = "target_below_window"
        elif r_low < 0.0 and r_high < 0.0:
            status = "target_above_window"
        else:
            status = "no_bracket"

        if debug:
            q_min = min(q_low, q_high)
            q_max = max(q_low, q_high)
            print(
                "[flow_solver] no bracket: "
                f"q_target={q_target:.6g}, "
                f"q_window=[{q_min:.6g}, {q_max:.6g}], "
                f"r_low={r_low:.6g}, r_high={r_high:.6g}, "
                f"fallback_dpdx={dpdx_fallback / 1e6:.2f} MPa/m, "
                f"status={status}"
            )

        return FlowSolveResult(
            dpdx=dpdx_fallback,
            status=status,
            residual_low=r_low,
            residual_high=r_high,
            q_low=q_low,
            q_high=q_high,
            q_target=q_target,
            hit_bound=True,
            bracket_low=dpdx_min,
            bracket_high=dpdx_max,
        )

    dpdx_sol = brentq(residual, dpdx_min, dpdx_max)

    if debug:
        print(
            f"[flow_solver] root found: dpdx={dpdx_sol / 1e6:.2f} MPa/m, "
            f"q_target={q_target:.6g}, q_low={q_low:.6g}, q_high={q_high:.6g}"
        )

    return FlowSolveResult(
        dpdx=dpdx_sol,
        status="root_found",
        residual_low=r_low,
        residual_high=r_high,
        q_low=q_low,
        q_high=q_high,
        q_target=q_target,
        hit_bound=False,
        bracket_low=dpdx_min,
        bracket_high=dpdx_max,
    )
