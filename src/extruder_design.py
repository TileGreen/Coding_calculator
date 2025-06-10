"""
extruder_design.py
Motor-and-drive sizing utilities.
"""
import math
from typing import Dict
from config import MotorSizingInputs

# ──────────────────────────────────────────────────────────────────────────────
def size_motor(throughput_kg_h: float,
               inp: MotorSizingInputs = MotorSizingInputs()) -> Dict[str, float]:
    """
    Return dict with:
        required_diameter_mm
        shaft_torque_Nm
        motor_power_kW
    """
    kg_s = throughput_kg_h / 3600.0

    # Rearranged volumetric-flow formula  Q = kv · D³ · n
    D_m = ((kg_s * 60.0) /
           (inp.melt_density_kg_m3 * inp.kv * inp.screw_rpm)) ** (1.0 / 3.0)
    D_cm = D_m * 100.0

    shaft_torque = inp.specific_torque_Nm_cm3 * (math.pi / 4.0) * D_cm**3
    power_kW = (2.0 * math.pi * inp.screw_rpm / 60.0 * shaft_torque) / 1000.0

    return {
        "required_diameter_mm": round(D_m * 1000.0, 1),
        "shaft_torque_Nm": round(shaft_torque * inp.safety_factor, 0),
        "motor_power_kW": round(power_kW * inp.safety_factor, 1),
    }
