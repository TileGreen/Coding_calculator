from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional
import numpy as np

__all__ = [
    "DISPERSIVE_PRESETS",
    "dispersive_score",
    "distributive_score",
    "global_mixing_index",
]

DISPERSIVE_PRESETS: Dict[str, Dict[str, float]] = {
    "strict_reference": {
        "tau_crit": 3.5e5,
        "gamma_target": 80.0,
        "energy_target": 1.0e6,
    },
    "relaxed_wpc": {
        "tau_crit": 2.0e5,
        "gamma_target": 40.0,
        "energy_target": 5.0e5,
    },
    "spc_exploratory": {
        "tau_crit": 2.5e5,
        "gamma_target": 60.0,
        "energy_target": 7.5e5,
    },
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


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
    """
    Return (ok?, meta) for dispersive mixing.

    Notes
    -----
    Here `gdot` and `t` are treated as zone-average values:
      gdot[i] = representative shear rate in zone i
      t[i]    = residence time spent in zone i

    So:
      gamma_int     = sum(gdot_i * t_i)
      energy_density = sum(tau_i * gdot_i * t_i)
    """
    gdot = np.asarray(gdot, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    if gdot.shape[0] != t.shape[0]:
        raise ValueError("Length of t must match length of gdot.")

    if np.any(t < 0):
        raise ValueError("Residence times must be non-negative.")

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

    gdot_mag = np.abs(gdot)
    gdot_eval = np.maximum(gdot_mag, 1.0)

    tau = eta_fn(gdot_eval) * gdot_eval

    gamma_int = float(np.sum(gdot_mag * t))
    energy_density = float(np.sum(tau * gdot_mag * t))
    tau_max = float(np.max(tau))

    energy_ok = energy_density >= energy_target
    stress_ok = tau_max >= tau_crit
    strain_ok = gamma_int >= gamma_target

    stress_ratio = tau_max / tau_crit if tau_crit > 0 else np.nan
    strain_ratio = gamma_int / gamma_target if gamma_target > 0 else np.nan
    energy_ratio = energy_density / energy_target if energy_target > 0 else np.nan

    stress_score = _clip01(stress_ratio)
    strain_score = _clip01(strain_ratio)
    energy_score = _clip01(energy_ratio)

    dispersive_index = (stress_score + strain_score + energy_score) / 3.0
    ok = energy_ok and stress_ok and strain_ok

    meta = {
        "preset_used": preset if preset is not None else "custom",
        "tau_crit_used": tau_crit,
        "gamma_target_used": gamma_target,
        "energy_target_used": energy_target,
        "tau_max": tau_max,
        "gamma_int": gamma_int,
        "energy_density": energy_density,
        "energy_ok": bool(energy_ok),
        "stress_ok": bool(stress_ok),
        "strain_ok": bool(strain_ok),
        "stress_ratio": float(stress_ratio),
        "strain_ratio": float(strain_ratio),
        "energy_ratio": float(energy_ratio),
        "stress_score": stress_score,
        "strain_score": strain_score,
        "energy_score": energy_score,
        "dispersive_index": float(dispersive_index),
    }

    if debug:
        print("\n=== Dispersive Mixing Meta ===")
        for k, v in meta.items():
            print(f"{k}: {v}")

    return ok, meta


def distributive_score(
    phi_field: np.ndarray,
    *,
    cv_target: float = 0.05,
    mzmi_target: float = 0.90,
) -> Tuple[bool, Dict[str, float]]:
    """
    Simple CV/MZMI-based distributive mixing score.
    Returns both pass/fail and a continuous distributive_index in [0, 1].
    """
    phi_field = np.asarray(phi_field, dtype=float).ravel()

    if phi_field.size == 0:
        raise ValueError("phi_field must not be empty.")

    mean_val = np.mean(phi_field)
    if np.isclose(mean_val, 0.0):
        raise ValueError("Mean exposure is too close to zero for CV calculation.")

    cv = float(np.std(phi_field) / mean_val)
    mzmi = float(max(0.0, 1.0 - cv))

    ok = (cv <= cv_target) and (mzmi >= mzmi_target)

    cv_score = 1.0 if cv <= cv_target else _clip01(cv_target / cv)
    mzmi_score = _clip01(mzmi / mzmi_target) if mzmi_target > 0 else 0.0
    distributive_index = 0.5 * (cv_score + mzmi_score)

    return ok, {
        "CV": cv,
        "MZMI": mzmi,
        "cv_score": cv_score,
        "mzmi_score": mzmi_score,
        "distributive_index": float(distributive_index),
    }


def global_mixing_index(
    dispersive_value: float,
    distributive_value: float,
    *,
    w_disp: float = 0.5,
    w_dist: float = 0.5,
) -> float:
    """
    Return continuous GMI in [0, 1].
    Inputs can be bools or floats.
    """
    disp = _clip01(float(dispersive_value))
    dist = _clip01(float(distributive_value))
    return w_disp * disp + w_dist * dist
