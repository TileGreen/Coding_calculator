# -*- coding: utf-8 -*-
"""
Main integration script: throughput, fill-factor, screw geometry, geometry screening, and drivetrain sizing
"""
import argparse
from pathlib import Path
from config import load_candidates, ScrewInputs
import drive_utils
import geometry_screening
from screw_design import design
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diameter", "-D", type=float, default=125,
                        help="Screw diameter D [mm] (default 125)")
    parser.add_argument("--LD", type=float, default=17.0,
                        help="Screw length-to-diameter ratio L/D (default 17)")
    parser.add_argument("--compression-ratio", "--CR", type=float, default=3.0,
                        help="Channel compression ratio h_f/h_m (default 3.0)")
    parser.add_argument("--rpm", type=float, default=25,
                        help="Nominal screw speed [rev/min] (default 25)")
    parser.add_argument("--feed-rate", type=float, default=None,
                        help="Feeder-limited throughput [kg/h]; omit for flood-fed")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write JSON summary to this file path")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Compute theoretical geometry and throughput
    baseline = ScrewInputs(
        D_mm=args.diameter,
        L_over_D=args.LD,
        compression_ratio=args.compression_ratio,
        screw_speed_rpm=args.rpm
    )
    sd_geom = design(baseline)
    if "throughput_max_kg_hr" not in sd_geom or sd_geom["throughput_max_kg_hr"] is None:
        raise RuntimeError("design() must return 'throughput_max_kg_hr'")
    Q_max = sd_geom["throughput_max_kg_hr"]

    # 2) Starve-feeding
    if args.feed_rate is not None:
        Q_actual = min(args.feed_rate, Q_max)
        phi_eff = Q_actual / Q_max
    else:
        Q_actual = Q_max
        phi_eff = sd_geom.get("fill_factor_eff", 1.0)

    sd_results = {
        "input_diameter_mm": args.diameter,
        "input_LD": args.LD,
        "input_compression_ratio": args.compression_ratio,
        "input_rpm": args.rpm,
        "input_feed_rate": args.feed_rate,
        "geometry": sd_geom,
        "throughput_max_kg_hr": Q_max,
        "fill_factor_eff": phi_eff,
        "throughput_actual_kg_hr": Q_actual,
    }

    # 2.5) Geometry-screening
    candidates = load_candidates(Path("candidates.yaml"))
    sd_results['top_geometries'] = geometry_screening.screen_geometries(
        candidates, Q_kg_h=Q_actual, rpm=args.rpm
    )[:3]

    # 3) Drive-train sizing
    torque_data = drive_utils.motor_from_screw(
        throughput_kg_h=Q_actual,
        rpm=args.rpm,
        diameter_mm=args.diameter
    )
    # Extract numeric torque if possible
    if isinstance(torque_data, dict):
        # assume first numeric value in dict
        vals = [v for v in torque_data.values() if isinstance(v, (int, float))]
        torque_val = vals[0] if vals else torque_data
    else:
        torque_val = torque_data
    motor_spec = drive_utils.select_motor(torque_val, rpm=args.rpm)
    sd_results.update({
        "required_torque": torque_val,
        "selected_motor": motor_spec,
    })

    # Print Outputs
    print("=== Input Parameters ===")
    print(f"Diameter: {args.diameter} mm")
    print(f"L/D: {args.LD}")
    print(f"CR: {args.compression_ratio}")
    print(f"RPM: {args.rpm}")
    print(f"Feed rate: {args.feed_rate if args.feed_rate is not None else 'flood-fed'} kg/h\n")

    print("=== Geometry & Throughput ===")
    print(f"Helix angle: {sd_geom['theta_deg']:.1f}°")
    print(f"Feed depth: {sd_geom.get('h_f_mm', sd_geom.get('h_f', 0)):.2f} mm, Meter depth: {sd_geom.get('h_m_mm', sd_geom.get('h_m', 0)):.2f} mm")
    print(f"Max throughput: {Q_max:.1f} kg/h\n")

    print("=== Throughput Adjustment ===")
    print(f"Fill factor eff.: {phi_eff:.2f}")
    print(f"Actual throughput: {Q_actual:.1f} kg/h\n")

    print("=== Top Geometries ===")
    for i, geom in enumerate(sd_results['top_geometries'], 1):
        print(f"{i}. {geom['name']} - score {geom['score']:.3e}")

    print("\n=== Drive-Train ===")
    # print torque safely
    if isinstance(torque_val, (int, float)):
        print(f"Required torque: {torque_val:.2f} Nm")
    else:
        print(f"Required torque: {torque_val}")
    print(f"Selected motor spec: {motor_spec}\n")

    # Write JSON if requested
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(sd_results, f, indent=2)
        print(f"Written to {args.out}")

if __name__ == '__main__':
    main()
