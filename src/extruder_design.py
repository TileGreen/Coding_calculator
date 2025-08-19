"""
extruder.py  –  Motor‑side sizing & power utilities
===================================================
Drop this module alongside *screw_design.py* and *drive_utils.py*.
It converts screw‑side throughput → power → torque, then maps that
through a gearbox (via :pyfunc:`drive_utils.motor_from_screw`).

Typical call from *main_integration.py*::

    from config import MotorSizingInputs
    import extruder

    motor_out = extruder.size_motor(Q, MotorSizingInputs(screw_rpm=20),
                                    gear_ratio=6.3, gear_eff=0.94)
    print(motor_out)

The helper keeps **all equations in one place** so you can later refine
empirical constants (k_v, service factors, etc.) without touching the
driver script.
"""

from __future__ import annotations

from dataclasses import asdict
from math import pi
from typing import Dict

from config import MotorSizingInputs  # type: ignore
from drive_utils import motor_from_screw

__all__ = ["screw_power", "size_motor"]

# -----------------------------------------------------------------------------
# 1.  Basic thermomechanical power estimate
# -----------------------------------------------------------------------------

def screw_power(throughput_kg_hr: float, k_v: float) -> float:
    """Return mechanical power **[kW]** required by the screw.

    Parameters
    ----------
    throughput_kg_hr : float
        Mass flow rate at die exit in **kg·h⁻¹**.
    k_v : float
        Empirical coefficient :math:`k_v` [kW per kg·h⁻¹].
        Rule‑of‑thumb values (single screw): 0.035–0.055.

    Notes
    -----
    :math:`P = k_v\,\dot m` is *strictly empirical* – refine once you
    have measured motor load curves or detailed FEM.
    """
    return k_v * throughput_kg_hr


# -----------------------------------------------------------------------------
# 2.  Complete motor/gearbox sizing pipeline
# -----------------------------------------------------------------------------

def size_motor(
    throughput_kg_hr: float,
    inp: MotorSizingInputs | None = None,
    *,
    gear_ratio: float = 5.0,
    gear_eff: float = 0.95,
    service_factor: float | None = None,
) -> Dict[str, float]:
    """Transform screw throughput into **motor‑side** requirements.

    Parameters
    ----------
    throughput_kg_hr : float
        Die mass flow in kg/h.
    inp : MotorSizingInputs, optional
        Dataclass collecting default constants (k_v, screw rpm, etc.).
    gear_ratio : float, optional
        Gearbox reduction :math:`G` (output/input). *Default = 5.0*.
    gear_eff : float, optional
        Gearbox efficiency (0 < η ≤ 1). *Default = 0.95*.
    service_factor : float, optional
        Extra multiplier for shock, duty, temperature. If *None*,
        the value in *inp* (``inp.safety_factor``) is used.

    Returns
    -------
    dict
        power_kW        – screw‑side mech. power  [kW]
        T_screw         – torque required at screw [N·m]
        rpm_screw       – process set‑point [rev/min]
        T_motor         – continuous motor torque [N·m]
        rpm_motor       – nominal motor speed [rev/min]
        P_mech          – same as *power_kW* but in W (for bookkeeping)
        T_motor_service – torque incl. service factor  [N·m]
        gear_ratio      – echoed
        gear_eff        – echoed
    """

    if inp is None:
        inp = MotorSizingInputs()

    if service_factor is None:
        service_factor = inp.safety_factor

    # 1) Screw‑side power (empirical)
    P_kW = screw_power(throughput_kg_hr, inp.kv)

    # 2) Convert to screw torque via process rpm
    rpm_screw = inp.screw_rpm
    P_W = P_kW * 1_000.0
    T_screw = P_W * 60.0 / (2 * pi * rpm_screw)

    # 3) Gearbox → motor
    drive = motor_from_screw(
        torque_screw=T_screw,
        rpm_screw=rpm_screw,
        gear_ratio=gear_ratio,
        gear_eff=gear_eff,
    )

    # 4) Apply service factor
    T_motor_sf = drive["T_motor"] * service_factor

    return {
        "power_kW": P_kW,
        "rpm_screw": rpm_screw,
        "T_screw": T_screw,
        "gear_ratio": gear_ratio,
        "gear_eff": gear_eff,
        **drive,
        "T_motor_service": T_motor_sf,
        **asdict(inp),  # keep original inputs for traceability
    }


# -----------------------------------------------------------------------------
# 3.  Self‑test when run as script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    _Q = 730  # kg/h – user target
    spec = size_motor(_Q, MotorSizingInputs(screw_rpm=25), gear_ratio=7.5, gear_eff=0.93)

    print("\nExtruder motor sizing self‑test – 730 kg/h\n" + "-" * 42)
    for k, v in spec.items():
        if k.startswith("T_"):
            unit = "N·m"
        elif k.startswith("rpm"):
            unit = "rpm"
        elif k.startswith("power") or k == "P_mech":
            unit = "kW" if k == "power_kW" else "W"
        else:
            continue
        print(f"{k:17s}: {v:,.2f} {unit}")