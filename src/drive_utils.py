# -*- coding: utf-8 -*-
"""
drive_utils.py – Gear‑train & motor helpers
==========================================

This module is **backwards‑compatible** with every legacy snippet but adds a
few conveniences that newer orchestration scripts rely on.

Public API (stable)
-------------------
- `motor_from_screw()` – accepts either the original positional signature
  **or** a keyword style that starts from throughput (kg/h).
- `select_motor()` – pick a standard IEC motor rating given required torque
  *or* a dict returned by `motor_from_screw`.

Both helpers stay import‑safe for older code.  If a caller only ever used the
original positional call, nothing breaks.
"""
from __future__ import annotations

from math import pi
from typing import Dict, Union

from config import MotorSizingInputs

__all__ = [
    "motor_from_screw",
    "select_motor",
]

# ----------------------------------------------------------------------------
# Constants & defaults
# ----------------------------------------------------------------------------

_KV_DEFAULT = MotorSizingInputs().kv            # kW per kg·h⁻¹ thumb‑rule
_GEAR_RATIO_DEFAULT = 5.0
_GEAR_EFF_DEFAULT = 0.95
_SERVICE_FACTOR_DEFAULT = 1.25                # common design margin

# Standard IEC power ratings [kW] (4‑pole 50 Hz)
_IEC_RATINGS_KW = [
    0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
    7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75,
    90, 110, 132, 160, 200, 250, 315,
]

# Very coarse mapping power → frame (sufficient for high‑level sizing)
_FRAME_MAP = {
    0.37: "71", 0.55: "80", 0.75: "80",
    1.1: "90S", 1.5: "90L", 2.2: "100L", 3.0: "100L",
    4.0: "112M", 5.5: "132S", 7.5: "132M", 11: "160M", 15: "160L",
    18.5: "180M", 22: "180L", 30: "200L", 37: "225S", 45: "225M",
    55: "250M", 75: "280S", 90: "280M", 110: "280M", 132: "315S",
    160: "315M", 200: "315L", 250: "355M", 315: "355L",
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _torque_from_throughput(throughput_kg_h: float, rpm_screw: float, kv: float = _KV_DEFAULT) -> float:
    """Return screw torque **[N·m]** via empirical :math:`P = k_v \dot m`."""
    P_kW = kv * throughput_kg_h
    P_W = P_kW * 1_000.0
    return P_W * 60.0 / (2 * pi * rpm_screw)


# ----------------------------------------------------------------------------
# Public: motor_from_screw (kept name!)
# ----------------------------------------------------------------------------

def motor_from_screw(*args, **kwargs) -> Dict[str, float]:
    """Translate screw requirements → motor‑side specs.

    Two calling styles:
    1. **Legacy positional**: (torque_screw, rpm_screw, gear_ratio [, gear_eff])
    2. **Keyword throughput**: throughput_kg_h=…, rpm=… [, gear_ratio, gear_eff]
    """

    # --- Positional legacy form ------------------------------------------------
    if args:
        if len(args) < 3:
            raise TypeError("motor_from_screw positional form needs ≥3 args: torque_screw, rpm_screw, gear_ratio")

        torque_screw, rpm_screw, gear_ratio, *rest = args
        gear_eff = rest[0] if rest else kwargs.get("gear_eff", _GEAR_EFF_DEFAULT)

    else:
        # --- Keyword throughput form ------------------------------------------
        throughput = kwargs.pop("throughput_kg_h", None)
        torque_screw = kwargs.pop("torque_screw", None)
        rpm_screw = kwargs.pop("rpm_screw", kwargs.pop("rpm", None))
        gear_ratio = kwargs.pop("gear_ratio", _GEAR_RATIO_DEFAULT)
        gear_eff = kwargs.pop("gear_eff", _GEAR_EFF_DEFAULT)
        kv = kwargs.pop("kv", _KV_DEFAULT)
        # Accept & ignore diameter_mm for caller convenience
        _ = kwargs.pop("diameter_mm", None)

        if torque_screw is None:
            if throughput is None or rpm_screw is None:
                raise TypeError("Provide either torque_screw or (throughput_kg_h & rpm)")
            torque_screw = _torque_from_throughput(throughput, rpm_screw, kv)

    if gear_ratio <= 0:
        raise ValueError("gear_ratio must be > 0")
    if not (0 < gear_eff <= 1):
        raise ValueError("gear_eff must satisfy 0 < η ≤ 1")

    # Reduction train to motor shaft
    T_motor = torque_screw / (gear_ratio * gear_eff)
    rpm_motor = rpm_screw * gear_ratio

    # Mechanical power on screw side
    P_mech_W = 2 * pi * torque_screw * rpm_screw / 60.0

    return {
        "T_motor": T_motor,
        "rpm_motor": rpm_motor,
        "P_mech_W": P_mech_W,
        "T_screw": torque_screw,
        "rpm_screw": rpm_screw,
        "gear_ratio": gear_ratio,
        "gear_eff": gear_eff,
    }


# ----------------------------------------------------------------------------
# Public: select_motor
# ----------------------------------------------------------------------------

def _pick_iec_rating(power_kW: float) -> float:
    """Return the first IEC rating ≥ *power_kW*."""
    for rating in _IEC_RATINGS_KW:
        if power_kW <= rating:
            return rating
    return _IEC_RATINGS_KW[-1]


def select_motor(torque: Union[float, Dict[str, float]], *, rpm: float, service_factor: float = _SERVICE_FACTOR_DEFAULT) -> Dict[str, Union[float, str]]:
    """Choose an IEC motor size.

    Parameters
    ----------
    torque : float | dict
        Either the **required shaft torque** [N·m] **or** the dict returned by
        :func:`motor_from_screw`.
    rpm : float
        Output‑shaft speed [rev/min] at which *torque* is required.
    service_factor : float, optional
        Multiplier to add design margin (typically 1.15–1.4).

    Returns
    -------
    dict with keys  `rated_power_kW`, `required_power_kW`, `frame_size`,
    `torque_Nm`, `rpm`.
    """
    # Extract numerical torque if caller passed the full spec
    if isinstance(torque, dict):
        torque_Nm = torque.get("T_motor") or torque.get("T_screw")
        if torque_Nm is None:
            raise ValueError("Torque dict lacks 'T_motor'/'T_screw' key")
    else:
        torque_Nm = float(torque)

    torque_Nm *= service_factor  # apply design margin

    # Power requirement at given speed
    P_req_W = 2 * pi * torque_Nm * rpm / 60.0
    P_req_kW = P_req_W / 1_000.0

    # Select standard rating
    rating_kW = _pick_iec_rating(P_req_kW)
    frame = _FRAME_MAP.get(rating_kW, "−")

    return {
        "required_power_kW": round(P_req_kW, 2),
        "rated_power_kW": rating_kW,
        "torque_Nm": round(torque_Nm, 1),
        "rpm": rpm,
        "service_factor": service_factor,
        "frame_size": frame,
    }


# -----------------------------------------------------------------------------
# Basic smoke‑test when run standalone
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    spec = motor_from_screw(throughput_kg_h=240, rpm=60)
    print("motor_from_screw→", spec)
    sel = select_motor(spec, rpm=spec["rpm_motor"])
    print("select_motor→", sel)
