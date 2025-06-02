# Extruder screw & motor quick‑sizer
#
# Change the inputs in the cell below and re‑run to explore different
# throughputs, screw rpm, melt density, etc.
# -----------------------------------------------------------------------------
import math

def screw_motor_calc(throughput_kg_h: float,
                     melt_density_kg_m3: float = 1500,
                     kv: float = 0.045,
                     screw_rpm: float = 80,
                     specific_torque_Nm_cm3: float = 12,
                     safety_factor: float = 1.15):
    """
    Simple single‑screw sizing helper.

    Parameters
    ----------
    throughput_kg_h : target mass rate            [kg/h]
    melt_density_kg_m3 : composite melt density   [kg/m³]
    kv : volumetric displacement coefficient      [–]  (≈0.04–0.05 typical)
    screw_rpm : planned running speed             [rev/min]
    specific_torque_Nm_cm3 : empirical value      [N·m/cm³]
    safety_factor : multiplier on power & torque

    Returns
    -------
    dict with diameter_mm, shaft_torque_Nm, motor_power_kW
    """
    # 1. Solve for screw diameter that meets mass rate at given rpm
    kg_s = throughput_kg_h / 3600
    D_m = ((kg_s * 60) / (melt_density_kg_m3 * kv * screw_rpm)) ** (1.0 / 3.0)
    D_mm = D_m * 1000

    # 2. Shaft torque from empirical specific‑torque rating
    D_cm = D_m * 100   # convert to cm
    shaft_torque_Nm = specific_torque_Nm_cm3 * (math.pi / 4) * D_cm ** 3

    # 3. Mechanical power at screw rpm
    motor_power_kW = 2 * math.pi * screw_rpm / 60 * shaft_torque_Nm / 1000

    # 4. Apply safety factor
    data = {
        "required_diameter_mm": round(D_mm, 1),
        "shaft_torque_Nm": round(shaft_torque_Nm * safety_factor, 0),
        "motor_power_kW": round(motor_power_kW * safety_factor, 1),
    }
    return data


# ---- Example: 730 kg/h on PE‑sand at 80 rpm -------------------------------
example = screw_motor_calc(
    throughput_kg_h=244,
    melt_density_kg_m3=1500,
    kv=0.045,
    screw_rpm=80,
    specific_torque_Nm_cm3=12,
    safety_factor=1.15
)

print(example)
