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
    return min(base, 0.95) if h_mm > 0.15 * D_mm else base

# -----------------------------------------------------------------------------
# 1) Main routine
# -----------------------------------------------------------------------------

def screen_geometries(
    candidates: List[Union[Dict, ScrewInputs]],
    *,
    rpm: float,
    throughput_kg_h: float,
    rho_melt: float = 1_200.0,
) -> List[Dict]:
    """Enrich *candidates* with flow & mixing metrics and return a ranked list."""

    Q_m3_s = kg_h_to_m3_s(throughput_kg_h, rho=rho_melt)
    results: List[Dict] = []

    for idx, cand in enumerate(candidates, 1):
        # 1) Prepare input object and mapping
        if isinstance(cand, ScrewInputs):
            inp = cand
            cand_map = vars(cand).copy()
        else:
            inp = ScrewInputs(**cand)
            cand_map = dict(cand)
        cand_map.setdefault("name", f"cand_{idx}")

        # 2) Base geometry and kinematics
        geom = design(inp)
        Q_max = geom.get("throughput_max_kg_hr", np.nan)
        # Actual fill factor based on desired throughput
        fill_eff = float(throughput_kg_h) / float(Q_max)

        D_m = inp.D_mm / 1_000.0                     # [m]
        V_surf = np.pi * D_m * rpm / 60.0            # [m/s]
        w_slot = np.pi * D_m                         # [m]

        # Raw depths [mm]
        h_f_mm = geom["h_f_mm"]
        h_m_mm = geom["h_m_mm"]
        h_c_mm = 0.5 * (h_f_mm + h_m_mm)

        # 3) Effective depths with actual fill factors
        MIN_DEPTH = 1.0e-4                            # 0.1 mm [m]
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

        V_feed = Q_m3_s / A_feed
        V_comp = Q_m3_s / A_comp
        V_metr = Q_m3_s / A_metr

        geom["t_feed_s"] = (inp.L_feed / 1_000.0) / V_feed
        geom["t_comp_s"] = (inp.L_compression / 1_000.0) / V_comp
        geom["t_metr_s"] = (inp.L_meter / 1_000.0) / V_metr

        # 5) Solve Couette–Poiseuille flow
        dpdx_feed, gamma_feed, tau_feed, Q_C_feed, Q_P_feed = solve_dpdx_for_slot(
            h_f_eff_m, w_slot, V_surf, Q_m3_s, eta_spc
        )
        dpdx_comp, gamma_comp, tau_comp, Q_C_comp, Q_P_comp = solve_dpdx_for_slot(
            h_c_eff_m, w_slot, V_surf, Q_m3_s, eta_spc
        )
        dpdx_metr, gamma_metr, tau_metr, Q_C_metr, Q_P_metr = solve_dpdx_for_slot(
            h_m_eff_m, w_slot, V_surf, Q_m3_s, eta_spc
        )

        # 6) Actual mass-based fill factors
        mass_C_feed = Q_C_feed * rho_melt * 3_600.0
        mass_C_comp = Q_C_comp * rho_melt * 3_600.0
        real_feed_fill = throughput_kg_h / mass_C_feed
        real_comp_fill = throughput_kg_h / mass_C_comp

        # 7) Mixing metrics
        tau_max = max(tau_feed, tau_comp, tau_metr)
        gamma_int = gamma_feed + gamma_comp + gamma_metr

        disp_ok, disp_meta = dispersive_score(
            np.array([gamma_feed, gamma_comp, gamma_metr]),
            np.array([geom["t_feed_s"], geom["t_comp_s"], geom["t_metr_s"]]),
            eta_fn=eta_spc,
            tau_crit=3.5e5,
            gamma_target=80.0,
        )

        exposure = np.array([gamma_feed, gamma_comp, gamma_metr]) * np.array([
            geom["t_feed_s"], geom["t_comp_s"], geom["t_metr_s"]
        ])
        dist_ok, dist_meta = distributive_score(exposure, cv_target=0.08)

        gmi = global_mixing_index(disp_ok, dist_ok)

        # 8) Aggregate diagnostics
        results.append({
            **cand_map,
            "h_f_mm": h_f_mm,
            "h_c_mm": h_c_mm,
            "h_m_mm": h_m_mm,
            "fill_eff": fill_eff,
            "comp_fill": comp_fill,
            **geom,
            "dpdx_feed_Pa_m": dpdx_feed,
            "dpdx_comp_Pa_m": dpdx_comp,
            "dpdx_metr_Pa_m": dpdx_metr,
            "gamma_feed_1_s": gamma_feed,
            "gamma_comp_1_s": gamma_comp,
            "gamma_metr_1_s": gamma_metr,
            "tau_feed_Pa": tau_feed,
            "tau_comp_Pa": tau_comp,
            "tau_metr_Pa": tau_metr,
            "Q_C_feed_m3_s": Q_C_feed,
            "Q_P_feed_m3_s": Q_P_feed,
            "Q_C_comp_m3_s": Q_C_comp,
            "Q_P_comp_m3_s": Q_P_comp,
            "Q_C_metr_m3_s": Q_C_metr,
            "Q_P_metr_m3_s": Q_P_metr,
            "real_feed_fill": real_feed_fill,
            "real_comp_fill": real_comp_fill,
            "tau_max_Pa": tau_max,
            "gamma_int": gamma_int,
            "energy_density": disp_meta.get("energy_density"),
            "CV_exposure": dist_meta["CV"],
            "MZMI": dist_meta["MZMI"],
            "GMI": gmi,
        })

    # 9) Sort and return
    return sorted(results, key=lambda r: (r["GMI"], r["gamma_int"]), reverse=True)
