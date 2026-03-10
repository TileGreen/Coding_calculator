from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np


__all__ = [
   "DISPERSIVE_PRESETS",
    "dispersive_score",
    "distributive_score",
    "global_mixing_index",
]
# ---------------------------------------------------------------------------
# Presets for dispersive-mixing thresholds
# ---------------------------------------------------------------------------

DISPERSIVE_PRESETS: Dict[str, Dict[str, float]] = {
    # Older stricter reference levels
    "strict_reference": {
        "tau_crit": 3.5e5,
        "gamma_target": 80.0,
        "energy_target": 1.0e6,
    },
    # Relaxed thresholds for low-shear / borrowed-WPC scenarios
    "relaxed_wpc": {
        "tau_crit": 2.0e5,
        "gamma_target": 40.0,
        "energy_target": 5.0e5,
    },
    # Middle-ground exploratory mode for SPC screening
    "spc_exploratory": {
        "tau_crit": 2.5e5,
        "gamma_target": 60.0,
        "energy_target": 7.5e5,
    },
}

# ---------------------------------------------------------------------------
# 1. Dispersive mixing (stress & strain)
# ---------------------------------------------------------------------------
def dispersive_score(
    gdot: np.ndarray,
    t: np.ndarray,
    *,
    eta_fn: Callable[[np.ndarray], np.ndarray],
    tau_crit: Optional[float] = None,
    gamma_target: Optional[float] = None,
    energy_target: Optional[float] = None,
    preset: Optional[str] = None,
    debug: bool = True,
):
    """Return ``(ok?, meta)`` for dispersive mixing."""
    gdot = np.asarray(gdot, dtype=float)
    t = np.asarray(t, dtype=float).ravel()

    if gdot.shape[0] != t.shape[0]:
        raise ValueError("Length of t must match gdot.shape[0] (time axis).")
    # Sort by increasing time if needed
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    gdot = gdot[sort_idx, ...]
    if debug:
        
        print("t increasing:", np.all(np.diff(t) > 0))
        print("gdot min/max:", np.min(gdot), np.max(gdot))

    preset_meta = {}
    if preset is not None:
        if preset not in DISPERSIVE_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(DISPERSIVE_PRESETS)}"
            )
        preset_meta = DISPERSIVE_PRESETS[preset]

    tau_crit = float(tau_crit if tau_crit is not None else preset_meta.get("tau_crit", 3.5e5))
    gamma_target = float(gamma_target if gamma_target is not None else preset_meta.get("gamma_target", 80.0))
    energy_target = float(energy_target if energy_target is not None else preset_meta.get("energy_target", 1.0e6))
  

    #gdot_adj = np.maximum(gdot, 1.0)
    gdot_mag = np.abs(gdot)
    gdot_eval = np.maximum(gdot_mag, 1.0)

    tau = eta_fn(gdot_eval) * gdot_eval

    energy_density = np.trapz(tau * gdot_mag, t, axis=0)
    tau_max = tau.max(axis=0)
    gamma_int = np.trapz(gdot_mag, t, axis=0)

    energy_ok = bool(np.all(energy_density >= energy_target))
    stress_ok = bool(np.all(tau_max >= tau_crit))
    strain_ok = bool(np.all(gamma_int >= gamma_target))

    ok = energy_ok and stress_ok and strain_ok

    meta = {
        "preset_used": preset if preset is not None else "custom",
        "tau_crit_used": tau_crit,
        "gamma_target_used": gamma_target,
        "energy_target_used": energy_target,
        "tau_max": float(np.max(tau_max)),
        "gamma_int": float(np.max(gamma_int)),
        "energy_density": float(np.max(energy_density)),
        "energy_ok": energy_ok,
        "stress_ok": stress_ok,
        "strain_ok": strain_ok,
    }

    if debug:
        print("\n=== Dispersive Mixing Meta ===")
        for k, v in meta.items():
            print(f"{k}: {v}")
        if not ok:
            failed = []
            if not stress_ok:
                failed.append("stress")
            if not strain_ok:
                failed.append("strain")
            if not energy_ok:
                failed.append("energy")
            print("failed_at:", ", ".join(failed))
        else:
            print("failed_at: none")

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

