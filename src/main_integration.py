# main_integration.py  – ADDED PRESSURE DEBUG
# -----------------------------------------------------------------------------
# CLI entry-point for sand-plastic screw candidate screening:
#   • Parse command-line args
#   • Load & override raw YAML candidate dicts
#   • Reference geometry & throughput
#   • Run geometry screening & mixing diagnostics
#   • Optional drive-train sizing and JSON output
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint

import yaml

from config import ScrewInputs
from screw_design import design
from geometry_screening import screen_geometries
import drive_utils  # optional: comment out if unavailable

# -----------------------------------------------------------------------------
# 0) Argument parsing
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("sand-plastic screw candidate screener")

    # Candidates YAML (optional positional)
    p.add_argument(
        "candidates",
        nargs="?",
        type=Path,
        default=Path("candidates.yaml"),
        help="Path to candidates.yaml (default: ./candidates.yaml)",
    )

    # Geometry overrides
    p.add_argument("--diameter", "-D", type=float, default=125.0,
                   help="Screw diameter D_mm [mm] (default 125)")
    p.add_argument("--LD", type=float, default=17.0,
                   help="Length-to-diameter ratio L_over_D (default 17)")
    p.add_argument("--CR", "--compression-ratio", type=float, default=3.0,
                   help="Channel compression_ratio h_f/h_m (default 3)")

    # Operating conditions
    p.add_argument("--rpm", "-n", type=float, default=90,
                   help="Screw speed [rev/min] (default 90)")
    p.add_argument("--feed-rate", type=float, default=90,
                   help="Throughput [kg/h]; omit for flood-fed")

    # Output control
    p.add_argument("--top", type=int, default=5,
                   help="Number of top candidates to show (default 5)")
    p.add_argument("--out", type=Path, default=None,
                   help="Write full JSON results to this path, if given")

    return p.parse_args()

# -----------------------------------------------------------------------------
# 1) Main driver
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    print("\n=== CLI arguments ===")
    pprint(vars(args))

    # ------------------------------------------------------------------------
    # 2) Reference geometry & max throughput
    # ------------------------------------------------------------------------
    ref_input = ScrewInputs(
        D_mm=args.diameter,
        L_over_D=args.LD,
        compression_ratio=args.CR,
        screw_speed_rpm=args.rpm,
    )
    ref_geom = design(ref_input)
    Q_max = ref_geom.get("throughput_max_kg_hr", float("inf"))
    Q_act = args.feed_rate if args.feed_rate is not None else Q_max
    Q_act = min(Q_act, Q_max)

    # ------------------------------------------------------------------------
    # 3) Load raw candidate dicts & override
    # ------------------------------------------------------------------------
    cand_path = args.candidates
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {cand_path}")
    raw = yaml.safe_load(cand_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{cand_path} must contain a list of candidates")

    dict_cands: list[dict] = []
    for idx, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ValueError("Each candidate entry must be a dict.")
        d = entry.copy()
        d.pop("name", None)
        d.setdefault("D_mm", args.diameter)
        d.setdefault("L_over_D", args.LD)
        d.setdefault("compression_ratio", args.CR)
        dict_cands.append(d)

    # ------------------------------------------------------------------------
    # 4) Run screening
    # ------------------------------------------------------------------------
    ranked = screen_geometries(
        dict_cands,
        rpm=args.rpm,
        throughput_kg_h=Q_act,
        rho_melt=1_200.0,
    )

    # ------------------------------------------------------------------------
    # 5) Pressure-gradient debug table
    # ------------------------------------------------------------------------
    hdr1 = (
        "Name    dpdx_feed[Pa/m] dpdx_comp[Pa/m] dpdx_metr[Pa/m]"
    )
    print("\n=== Pressure-Gradient Debug ===")
    print(hdr1)
    print("-" * len(hdr1))
    for r in ranked:
        print(
            f"{r['name']:<8} "
            f"{r['dpdx_feed_Pa_m']:16.2e} "
            f"{r['dpdx_comp_Pa_m']:16.2e} "
            f"{r['dpdx_metr_Pa_m']:16.2e}"
        )

    # ------------------------------------------------------------------------
    # 6) Depth & rheology debug table
    # ------------------------------------------------------------------------
    hdr2 = "Name   h_f[mm]  h_c[mm]  h_m[mm]  compFill  tau_max[Pa]  gamma_int"
    print("\n=== Depth & Rheology Debug ===")
    print(hdr2)
    print("-" * len(hdr2))
    for r in ranked:
        print(
            f"{r['name']:<6} "
            f"{r['h_f_mm']:8.2f} "
            f"{r['h_c_mm']:8.2f} "
            f"{r['h_m_mm']:8.2f} "
            f"{r['comp_fill']:8.2f} "
            f"{r['tau_max_Pa']:12.2e} "
            f"{r['gamma_int']:12.1f}"
        )

    # ------------------------------------------------------------------------
    # 7) Summary of top candidates
    # ------------------------------------------------------------------------
    print("\n=== Top Candidates ===")
    for i, r in enumerate(ranked[: args.top], 1):
        print(
            f"{i}. {r['name']} – GMI {r['GMI']:.3f}, "
            f"τ_max {r['tau_max_Pa']/1e6:.2f} MPa, "
            f"γ_int {r['gamma_int']:.1f}"
        )

    # ------------------------------------------------------------------------
    # 8) Drive-train sizing
    # ------------------------------------------------------------------------
    try:
        motor_data = drive_utils.motor_from_screw(
            throughput_kg_h=Q_act,
            rpm=args.rpm,
            gear_ratio=4.0,      # your real gearbox ratio
            gear_eff=0.95,       # typical efficiency
        )
        torque_screw = float(motor_data["T_screw"])
        motor_spec   = drive_utils.select_motor(motor_data, rpm=args.rpm)

        print("\n=== Drive-Train Sizing ===")
        print(f"Screw torque: {torque_screw:.1f} Nm")
        print(f"Motor selection: {motor_spec}")
    except Exception as ex:
        print(f"[drive_utils] skipped motor sizing: {ex}")

    # ------------------------------------------------------------------------
    # 9) JSON output
    # ------------------------------------------------------------------------
    if args.out:
        out_data = {
            "reference_geometry": ref_geom,
            "Q_actual_kg_h": Q_act,
            "top_candidates": ranked[: args.top],
            "drive_train": {"torque_Nm": torque_screw, "motor_spec": motor_spec},
        }
        args.out.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved full results → {args.out}")


if __name__ == "__main__":
    main()