#!/usr/bin/env python3
"""
main.py  –  Integration driver
Run after placing all modules (config, screw_design, extruder_design,
viscosity_curve_v3) in the same folder plus data/wpc_viscosity.csv.
"""
from config import ScrewInputs, MotorSizingInputs
import screw_design
import extruder_design
import rheological_utils as rh     # ← new source of rheology data

# ─────────────────────────────────────────────────────────────
def main():
    # 1. Screw geometry / throughput
    screw_in  = ScrewInputs(D_mm=125.0, screw_speed_rpm=80.0)
    screw_out = screw_design.design(screw_in)
    Q         = screw_out["throughput_kg_hr"]

    print(f"[Screw design] Throughput  : {Q:.0f} kg/h")
    print(f"               Helix angle: {screw_out['theta_deg']:.1f}°")
    print(f"               Meter depth: {screw_out['h_m_mm']:.2f} mm")

    # 2. Motor sizing
    motor_in  = MotorSizingInputs(screw_rpm=screw_in.screw_speed_rpm)
    motor_out = extruder_design.size_motor(Q, motor_in)

    print("\n[Motor sizing]")
    for k, v in motor_out.items():
        print(f"  {k.replace('_',' ').title():22s}: {v}")

    # 3. Rheology sanity check @195 °C
    g_ref = 100.0
    T_ref = 195.0
    eta_pe  = rh.eta_poly(g_ref, T=T_ref)
    eta_spc = rh.eta_spc(g_ref, T=T_ref)
   
    print(f"\n[Rheology] η_PE ({g_ref:.0f} s⁻¹, {T_ref} °C)  ≈ {eta_pe:,.0f} Pa·s")
    print(f"           η_SPC({g_ref:.0f} s⁻¹, {T_ref} °C) ≈ {eta_spc:,.0f} Pa·s")

if __name__ == "__main__":
    main()
