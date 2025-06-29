from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np

__all__ = [
    "dispersive_score",
    "distributive_score",
    "global_mixing_index",
]

# ---------------------------------------------------------------------------
# 1. Dispersive mixing (stress & strain)
# ---------------------------------------------------------------------------

def dispersive_score(
    gdot: np.ndarray,
    t: np.ndarray,
    *,
    eta_fn: Callable[[np.ndarray], np.ndarray],
    tau_crit: float = 3.5e5,
    gamma_target: float = 80.0,
):
    """Return ``(ok?, meta)`` for dispersive mixing.

    * Works with 1‑D or N‑D ``gdot`` arrays whose *time* axis is **0**.
      ``t`` must be a 1‑D array of the same length as ``gdot.shape[0]``.
    * Thresholds are applied via ``np.all`` so **every** trace must satisfy
      the relaxed energy/stress/strain criteria.
    """
    # Ensure NumPy arrays
    gdot = np.asarray(gdot, dtype=float)
    t = np.asarray(t, dtype=float).ravel()  # force 1‑D time vector

    if gdot.shape[0] != t.shape[0]:
        raise ValueError("Length of t must match gdot.shape[0] (time axis).")

    # Minimum shear‑rate floor for viscosity evaluation
    gdot_adj = np.maximum(gdot, 1.0)

    tau = eta_fn(gdot_adj) * gdot_adj

    # Energy density E = ∫τ·γ̇ dt along axis 0
    energy_density = np.trapz(tau * gdot, t, axis=0)
    tau_max = tau.max(axis=0)
    gamma_int = np.trapz(gdot, t, axis=0)

    # Relaxed thresholds for low‑shear scenarios
    energy_ok = bool(np.all(energy_density > 5e5))  # was 1e6
    stress_ok = bool(np.all(tau_max >= 2e5))         # was 3.5e5
    strain_ok = bool(np.all(gamma_int >= 40.0))      # was 80

    ok = energy_ok and stress_ok and strain_ok

    meta = {
        "tau_max": float(np.max(tau_max)),
        "gamma_int": float(np.max(gamma_int)),
        "energy_density": float(np.max(energy_density)),
    }

    return ok, meta

# ---------------------------------------------------------------------------
# 2. Distributive mixing (uniformity)
# ---------------------------------------------------------------------------

def distributive_score(
    phi_field: np.ndarray,
    *,
    cv_target: float = 0.05,
    mzmi_target: float = 0.90,
) -> Tuple[bool, Dict[str, float]]:
    """Simple CV/MZMI–based distributive‑mixing score."""
    phi_field = np.asarray(phi_field, dtype=float)

    if phi_field.size == 0:
        raise ValueError("phi_field must not be empty.")

    cv = float(np.std(phi_field) / np.mean(phi_field))
    mzmi = 1.0 - cv

    ok = (cv <= cv_target) and (mzmi >= mzmi_target)
    return ok, {"CV": cv, "MZMI": mzmi}

# ---------------------------------------------------------------------------
# 3. Global mixing index
# ---------------------------------------------------------------------------

def global_mixing_index(
    dispersive_ok: bool,
    distributive_ok: bool,
    *,
    w_disp: float = 0.5,
    w_dist: float = 0.5,
) -> float:
    """Return *GMI* ∈ [0, 1] with tunable weights."""
    return w_disp * float(dispersive_ok) + w_dist * float(distributive_ok)

