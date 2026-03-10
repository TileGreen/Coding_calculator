
# geometry_screening.py
# -----------------------------------------------------------------------------
# Candidate-screening helper for sand-plastic screw geometries
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import List, Dict, Union
import numpy as np

from config import ScrewInputs
from screw_design import design
from rheological_utils import eta_spc
from mixing_matrix import (
    DISPERSIVE_PRESETS,
    dispersive_score,
    distributive_score,
    global_mixing_index,
)
from flow_solver import solve_dpdx_for_slot

__all__ = [
    "screen_geometries",
]

# -----------------------------------------------------------------------------
# 0) Helpers
# -----------------------------------------------------------------------------

def kg_h_to_m3_s(Q_kg_h: float, rho: float = 1_000.0) -> float:
    """Convert mass flow [kg/h] to volumetric flow [m³/s] given density [kg/m³]."""
    return Q_kg_h / rho / 3_600.0


def get_comp_fill(fill: float, h_mm: float, D_mm: float) -> float:
    """Compression-zone fill heuristic based on fill × depth.

    * Allow up to 1.5× nominal fill in shallow channels.
    * Clip to ≤95% when depth > 0.15 × D to avoid over-pressurising.
    """
    base = min(fill * 1.5, 1.0)
    result = min(base, 0.95) if h_mm > 0.15 * D_mm else base
    # Enforce minimum 50% fill for compression zone
    return max(result, 0.5)

# -----------------------------------------------------------------------------
# 1) Main routine
# -----------------------------------------------------------------------------

def screen_geometries(
    candidates: List[Union[Dict, ScrewInputs]],
    *,
    rpm: float,
    throughput_kg_h: float,
    rho_melt: float = 1_200.0,
    dispersive_preset: str = "relaxed_wpc",
) -> List[Dict]:
    """Enrich candidates with flow & mixing metrics and return a ranked list.

    Notes
    -----
    - `rpm` is enforced as the operating speed for all candidates during screening.
    - Candidates whose requested throughput exceeds their drag-based capacity are
      still evaluated, but are flagged as overloaded.
    """

    if rpm <= 0:
        raise ValueError("rpm must be > 0")

    if throughput_kg_h <= 0:
        raise ValueError("throughput_kg_h must be > 0")

    if rho_melt <= 0:
        raise ValueError("rho_melt must be > 0")

    Q_req_kg_h = throughput_kg_h
    results: List[Dict] = []

    for idx, cand in enumerate(candidates, 1):
        # 1) Normalize candidate input
        if isinstance(cand, ScrewInputs):
            inp = cand
            cand_map = vars(cand).copy()
        else:
            inp = ScrewInputs(**cand)
            cand_map = dict(cand)

        cand_map.setdefault("name", f"cand_{idx}")

        # Enforce common screening rpm for fair comparison
        inp.screw_speed_rpm = rpm
        cand_map["screening_rpm"] = rpm

        # 2) Base geometry
        geom = design(inp)
        Q_max = float(geom.get("throughput_max_kg_hr", np.nan))

        # Validate Q_max
        if not np.isfinite(Q_max) or Q_max <= 0:
            cand_map["screen_status"] = "invalid_Qmax"
            cand_map["Q_max_kg_h"] = Q_max
            results.append({
                **cand_map,
                **geom,
                "fill_eff_raw": np.nan,
                "fill_eff": np.nan,
                "overloaded": True,
                "GMI": -np.inf,
                "gamma_int": -np.inf,
            })
            continue

        fill_eff_raw = Q_req_kg_h / Q_max
        overloaded = fill_eff_raw > 1.0

        # Clip for geometric fill usage, but evaluate physics only at the
        # candidate-achievable flow so weak geometries are not over-penalized.
        fill_eff = min(max(fill_eff_raw, 0.0), 1.0)
        Q_eval_kg_h = min(Q_req_kg_h, Q_max)
        Q_eval_m3_s = kg_h_to_m3_s(Q_eval_kg_h, rho=rho_melt)

        D_m = inp.D_mm / 1_000.0
        V_surf = np.pi * D_m * rpm / 60.0
        w_slot = np.pi * D_m                     # [m]

        # Raw depths [mm]
        h_f_mm = geom["h_f_mm"]
        h_m_mm = geom["h_m_mm"]
        h_c_mm = 0.5 * (h_f_mm + h_m_mm)

        # 3) Effective depths with actual fill factors
        MIN_DEPTH = 5e-3                           # 0.1 mm [m]
        # Feed zone uses actual fill
        h_f_eff_m = max((h_f_mm * fill_eff) / 1_000.0, MIN_DEPTH)
        # Compression zone uses 1.5× fill_eff but capped at 1.0
        comp_fill = get_comp_fill(fill_eff, h_c_mm, inp.D_mm)
        h_c_eff_m = max((h_c_mm * comp_fill) / 1_000.0, MIN_DEPTH)
        # Metering zone always full channel
        h_m_eff_m = max(h_m_mm / 1_000.0, MIN_DEPTH)

        # 4) Axial velocities & residence times
        A_feed = 0.9 * w_slot * h_f_eff_m
        A_comp = 0.9 * w_slot * h_c_eff_m
        A_metr = 0.9 * w_slot * h_m_eff_m

        V_feed = Q_eval_m3_s / A_feed
        V_comp = Q_eval_m3_s / A_comp
        V_metr = Q_eval_m3_s / A_metr

        geom["t_feed_s"] = (inp.L_feed / 1_000.0) / V_feed
        geom["t_comp_s"] = (inp.L_compression / 1_000.0) / V_comp
        geom["t_metr_s"] = (inp.L_meter / 1_000.0) / V_metr

        # 5) Solve Couette–Poiseuille flow
        (dpdx_feed, gamma_feed, tau_feed,Q_C_feed,Q_P_feed,status_feed,Qmin_feed,Qmax_feed,) = solve_dpdx_for_slot( h_f_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_spc)
        (dpdx_comp, gamma_comp, tau_comp,Q_C_comp,Q_P_comp,status_comp,Qmin_comp,Qmax_comp,) = solve_dpdx_for_slot( h_f_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_spc)
        (dpdx_metr, gamma_metr, tau_metr,Q_C_metr,Q_P_metr,status_metr,Qmin_metr,Qmax_metr,) = solve_dpdx_for_slot( h_f_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_spc)

        # 6) Actual mass-based fill factors
        mass_C_feed = Q_C_feed * rho_melt * 3_600.0
        mass_C_comp = Q_C_comp * rho_melt * 3_600.0
        real_feed_fill = Q_eval_kg_h / mass_C_feed
        real_comp_fill = Q_eval_kg_h / mass_C_comp

        # 7) Mixing metrics
        tau_max = max(tau_feed, tau_comp, tau_metr)
        gamma_int = gamma_feed + gamma_comp + gamma_metr

        disp_ok, disp_meta = dispersive_score(
            np.array([gamma_feed, gamma_comp, gamma_metr]),
            np.array([geom["t_feed_s"], geom["t_comp_s"], geom["t_metr_s"]]),
            eta_fn=eta_spc,
            preset=dispersive_preset
            #tau_crit=3.5e5,
            #gamma_target=80.0,
            #energy_target=5.0e5
        )

        exposure = np.array([gamma_feed, gamma_comp, gamma_metr]) * np.array([
            geom["t_feed_s"], geom["t_comp_s"], geom["t_metr_s"]
        ])
        dist_ok, dist_meta = distributive_score(exposure, cv_target=0.08)

        gmi = global_mixing_index(disp_ok, dist_ok)

        # 8) Aggregate diagnostics
      
        results.append({
            **cand_map,

            "k_f": inp.k_feed_depth if inp.k_feed_depth is not None else 0.15,

            "h_f_mm": h_f_mm,
            "h_c_mm": h_c_mm,
            "h_m_mm": h_m_mm,

            "fill_eff": fill_eff,
            "comp_fill": comp_fill,

        **geom,

        # ───────────────── Pressure gradients ─────────────────
        "dpdx_feed_Pa_m": dpdx_feed,
        "dpdx_comp_Pa_m": dpdx_comp,
        "dpdx_metr_Pa_m": dpdx_metr,

        # solver status
        "status_feed": status_feed,
        "status_comp": status_comp,
        "status_metr": status_metr,

        # flow windows
        "Qmin_feed_m3_s": Qmin_feed,
        "Qmax_feed_m3_s": Qmax_feed,
        "Qmin_comp_m3_s": Qmin_comp,
        "Qmax_comp_m3_s": Qmax_comp,
        "Qmin_metr_m3_s": Qmin_metr,
        "Qmax_metr_m3_s": Qmax_metr,

        # ───────────────── Rheology ─────────────────
        "gamma_feed_1_s": gamma_feed,
        "gamma_comp_1_s": gamma_comp,
        "gamma_metr_1_s": gamma_metr,

        "tau_feed_Pa": tau_feed,
        "tau_comp_Pa": tau_comp,
        "tau_metr_Pa": tau_metr,

        # ───────────────── Couette / Pressure flows ─────────────────
        "Q_C_feed_m3_s": Q_C_feed,
        "Q_P_feed_m3_s": Q_P_feed,

        "Q_C_comp_m3_s": Q_C_comp,
        "Q_P_comp_m3_s": Q_P_comp,

        "Q_C_metr_m3_s": Q_C_metr,
        "Q_P_metr_m3_s": Q_P_metr,

        # ───────────────── Fill metrics ─────────────────
        "real_feed_fill": real_feed_fill,
        "real_comp_fill": real_comp_fill,

        # ───────────────── Mixing metrics ─────────────────
        "tau_max_Pa": tau_max,
        "gamma_int": gamma_int,
        "energy_density": disp_meta.get("energy_density"),

        "CV_exposure": dist_meta["CV"],
        "MZMI": dist_meta["MZMI"],

        "GMI": gmi,

        # ───────────────── Throughput ─────────────────
        "Q_max_kg_h": Q_max,
        "Q_req_kg_h": Q_req_kg_h,
        "Q_eval_kg_h": Q_eval_kg_h,

        "capacity_shortfall_kg_h": max(Q_req_kg_h - Q_eval_kg_h, 0.0),

        "overload_ratio": fill_eff_raw,
        "fill_eff_raw": fill_eff_raw,
        "fill_eff": fill_eff,

        "overloaded": overloaded,

        # ───────────────── Dispersive model info ─────────────────
        "dispersive_preset": disp_meta.get("preset_used"),
        "tau_crit_used": disp_meta.get("tau_crit_used"),
        "gamma_target_used": disp_meta.get("gamma_target_used"),
        "energy_target_used": disp_meta.get("energy_target_used"),

        "disp_energy_ok": disp_meta.get("energy_ok"),
        "disp_stress_ok": disp_meta.get("stress_ok"),
        "disp_strain_ok": disp_meta.get("strain_ok"),
})
    # 9) Sort and return
    return sorted(
    results,
    key=lambda r: (
        0 if r.get("overloaded", True) else 1,
        r.get("GMI", -np.inf),
        r.get("gamma_int", -np.inf),
    ),
    reverse=True,
)
