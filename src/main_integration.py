# main_integration.py
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

from rheological_utils import get_rheology_model
from config import ScrewInputs
from screw_design import design
from geometry_screening import screen_geometries
import drive_utils


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

    p.add_argument(
        "--rheo-model",
        type=str,
        default="wpc_surrogate",
        choices=["wpc_surrogate", "simple_powerlaw", "measured_capillary"],
        help="Rheology model to use",
    )

    p.add_argument(
        "--rheo-temp",
        type=float,
        default=195.0,
        help="Rheology temperature [C]",
    )

    p.add_argument(
        "--rheo-data",
        type=str,
        default=None,
        help="CSV/XLSX path for rheology data",
    )

    p.add_argument("--wpc-wood-wt", type=float, default=50.0)
    p.add_argument("--spc-sand-wt", type=float, default=50.0)

    p.add_argument("--pl-K", type=float, default=15000.0)
    p.add_argument("--pl-n", type=float, default=0.35)

    # Geometry overrides
    p.add_argument(
        "--diameter",
        "-D",
        type=float,
        default=85,
        help="Screw diameter D_mm [mm] (default 85)",
    )
    p.add_argument(
        "--LD",
        type=float,
        default=44.0,
        help="Length-to-diameter ratio L_over_D (default 44)",
    )
    p.add_argument(
        "--CR",
        "--compression-ratio",
        type=float,
        default=3,
        help="Channel compression_ratio h_f/h_m (default 3)",
    )

    # Operating conditions
    p.add_argument(
        "--rpm",
        "-n",
        type=float,
        default=18,
        help="Screw speed [rev/min] (default 30)",
    )
    p.add_argument(
        "--feed-rate",
        type=float,
        default=300,
        help="Throughput [kg/h]; omit for flood-fed",
    )

    # Output control
    p.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top candidates to show (default 5)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full JSON results to this path, if given",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# -----------------------------------------------------------------------------
# 1) Main driver
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    print("\n=== CLI arguments ===")
    pprint(vars(args))

    # -------------------------------------------------------------------------
    # 2) Reference geometry & requested throughput
    # -------------------------------------------------------------------------
    ref_input = ScrewInputs(
        D_mm=args.diameter,
        L_over_D=args.LD,
        compression_ratio=args.CR,
        screw_speed_rpm=args.rpm,
    )

    ref_geom = design(ref_input)
    Q_ref_max = ref_geom.get("throughput_max_kg_hr", float("inf"))
    Q_req = args.feed_rate if args.feed_rate is not None else Q_ref_max

    # -------------------------------------------------------------------------
    # 3) Load candidate dicts & apply global overrides
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4) Rheology + screening
    # -------------------------------------------------------------------------
    rheo_model = get_rheology_model(
        model=args.rheo_model,
        role="spc",
        T=args.rheo_temp,
        data_path=args.rheo_data,
        wpc_wood_wt=args.wpc_wood_wt,
        spc_sand_wt=args.spc_sand_wt,
        K=args.pl_K,
        n=args.pl_n,
    )

    ranked = screen_geometries(
        dict_cands,
        rpm=args.rpm,
        throughput_kg_h=Q_req,
        rho_melt=1200.0,
        eta_fn=rheo_model.eta,
    )

    if not ranked:
        raise RuntimeError("screen_geometries returned no candidates.")

    # -------------------------------------------------------------------------
    # 5) Pressure-gradient debug table
    # -------------------------------------------------------------------------
    hdr1 = "Name    dpdx_feed[Pa/m] dpdx_comp[Pa/m] dpdx_metr[Pa/m]"
    print("\n=== Pressure-Gradient Debug ===")
    print(hdr1)
    print("-" * len(hdr1))

    for r in ranked:
        print(
            f"{r['name']:<8} "
            f"{_safe_float(r.get('dpdx_feed_Pa_m')):16.2e} "
            f"{_safe_float(r.get('dpdx_comp_Pa_m')):16.2e} "
            f"{_safe_float(r.get('dpdx_metr_Pa_m')):16.2e}"
        )

    # -------------------------------------------------------------------------
    # 6) Depth & rheology debug table
    # -------------------------------------------------------------------------
    hdr2 = "Name   h_f[mm]  h_c[mm]  h_m[mm]  compFill  tau_max[Pa]  gamma_int"
    print("\n=== Depth & Rheology Debug ===")
    print(hdr2)
    print("-" * len(hdr2))

    for r in ranked:
        print(
            f"{r['name']:<6} "
            f"{_safe_float(r.get('h_f_mm')):8.2f} "
            f"{_safe_float(r.get('h_c_mm')):8.2f} "
            f"{_safe_float(r.get('h_m_mm')):8.2f} "
            f"{_safe_float(r.get('comp_fill')):8.2f} "
            f"{_safe_float(r.get('tau_max_Pa')):12.2e} "
            f"{_safe_float(r.get('gamma_int')):12.1f}"
        )

    # -------------------------------------------------------------------------
    # 7) Summary of top candidates
    # -------------------------------------------------------------------------
    print("\n=== Top Candidates ===")
    for i, r in enumerate(ranked[: args.top], 1):
        disp_index = _safe_float(r.get("dispersive_index"), default=-1.0)
        dist_index = _safe_float(r.get("distributive_index"), default=-1.0)

        has_continuous_indices = disp_index >= 0.0 and dist_index >= 0.0

        if has_continuous_indices:
            print(
                f"{i}. {r['name']} – "
                f"GMI { _safe_float(r.get('GMI')):.3f}, "
                f"Gdisp {disp_index:.3f}, "
                f"Gdist {dist_index:.3f}, "
                f"τ_max { _safe_float(r.get('tau_max_Pa')) / 1e6:.2f} MPa, "
                f"γ_int { _safe_float(r.get('gamma_int')):.1f}, "
                f"P_rheo { _safe_float(r.get('P_rheo_kW')):.2f} kW, "
                f"T_rheo { _safe_float(r.get('T_rheo_Nm')):.1f} Nm"
            )
        else:
            print(
                f"{i}. {r['name']} – "
                f"GMI { _safe_float(r.get('GMI')):.3f}, "
                f"τ_max { _safe_float(r.get('tau_max_Pa')) / 1e6:.2f} MPa, "
                f"γ_int { _safe_float(r.get('gamma_int')):.1f}, "
                f"P_rheo { _safe_float(r.get('P_rheo_kW')):.2f} kW, "
                f"T_rheo { _safe_float(r.get('T_rheo_Nm')):.1f} Nm"
            )

    # -------------------------------------------------------------------------
    # 8) Drive-train sizing
    # -------------------------------------------------------------------------
    torque_screw = None
    motor_spec = None
    drive_summary = {}

    try:
        best = ranked[0]

        # 8.1 Empirical estimate from throughput
        motor_data_emp = drive_utils.motor_from_screw(
            throughput_kg_h=Q_req,
            rpm=args.rpm,
            gear_ratio=4.0,
            gear_eff=0.95,
        )

        T_emp_Nm = float(motor_data_emp["T_screw"])
        P_emp_kW = float(motor_data_emp["P_screw_W"]) / 1000.0

        # 8.2 Rheology-based estimate from screening results
        # If geometry_screening.py is not yet updated, these default to 0
        T_rheo_Nm = _safe_float(best.get("T_rheo_Nm"), default=0.0)
        P_rheo_kW = _safe_float(best.get("P_rheo_kW"), default=0.0)

        # Optional diagnostic ratio
        torque_ratio = float("inf") if T_emp_Nm <= 0 else T_rheo_Nm / T_emp_Nm

        # 8.3 Conservative design basis
        T_design_Nm = max(T_emp_Nm, T_rheo_Nm)
        P_design_kW = max(P_emp_kW, P_rheo_kW)

        # 8.4 Rebuild motor spec using DESIGN screw torque
        motor_spec_design = drive_utils.motor_from_screw(
            torque_screw=T_design_Nm,
            rpm_screw=args.rpm,
            gear_ratio=4.0,
            gear_eff=0.95,
        )

        motor_selection = drive_utils.select_motor(motor_spec_design)

        print("\n=== Drive-Train Sizing ===")
        print(f"Empirical screw torque : {T_emp_Nm:.1f} Nm")
        print(f"Empirical screw power  : {P_emp_kW:.2f} kW")
        print(f"Rheology torque        : {T_rheo_Nm:.1f} Nm")
        print(f"Rheology power         : {P_rheo_kW:.2f} kW")
        print(f"Design screw torque    : {T_design_Nm:.1f} Nm")
        print(f"Design screw power     : {P_design_kW:.2f} kW")

        if torque_ratio == float("inf"):
            print("Torque ratio (rheo/emp): inf")
        else:
            print(f"Torque ratio (rheo/emp): {torque_ratio:.2f}")

        print(f"Motor selection        : {motor_selection}")

        torque_screw = T_design_Nm
        motor_spec = motor_selection

        drive_summary = {
            "T_emp_Nm": T_emp_Nm,
            "P_emp_kW": P_emp_kW,
            "T_rheo_Nm": T_rheo_Nm,
            "P_rheo_kW": P_rheo_kW,
            "T_design_Nm": T_design_Nm,
            "P_design_kW": P_design_kW,
            "torque_ratio_rheo_to_emp": torque_ratio,
            "motor_spec_design": motor_spec_design,
            "motor_selection": motor_selection,
        }

    except Exception as ex:
        print(f"[drive_utils] skipped motor sizing: {ex}")

    # -------------------------------------------------------------------------
    # 9) JSON output
    # -------------------------------------------------------------------------
    if args.out:
        out_data = {
            "reference_geometry": ref_geom,
            "rheology": rheo_model.meta,
            "Q_requested_kg_h": Q_req,
            "top_candidates": ranked[: args.top],
            "drive_train": {
                "torque_Nm": torque_screw,
                "motor_spec": motor_spec,
                "summary": drive_summary,
            },
        }

        args.out.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved full results → {args.out}")


if __name__ == "__main__":
    main()
