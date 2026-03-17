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
# Helpers
# -----------------------------------------------------------------------------

def kg_h_to_m3_s(Q_kg_h: float, rho: float = 1_000.0) -> float:
    """Convert mass flow [kg/h] to volumetric flow [m³/s]."""
    return Q_kg_h / rho / 3600.0


def get_comp_fill(fill: float, h_mm: float, D_mm: float) -> float:
    base = min(fill * 1.5, 1.0)
    result = min(base, 0.95) if h_mm > 0.15 * D_mm else base
    return max(result, 0.5)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def screen_geometries(
    candidates: List[Union[Dict, ScrewInputs]],
    *,
    rpm: float,
    eta_fn=None,
    throughput_kg_h: float,
    rho_melt: float = 1200.0,
    dispersive_preset: str = "relaxed_wpc",
) -> List[Dict]:
    if eta_fn is None:
        eta_fn = eta_spc

    if rpm <= 0:
        raise ValueError("rpm must be > 0")

    if throughput_kg_h <= 0:
        raise ValueError("throughput_kg_h must be > 0")

    Q_req_kg_h = throughput_kg_h
    results: List[Dict] = []

    for idx, cand in enumerate(candidates, 1):

        # ---------------------------------------------------------------------
        # Candidate parsing
        # ---------------------------------------------------------------------

        if isinstance(cand, ScrewInputs):
            inp = cand
            cand_map = vars(cand).copy()
            depth_override_hf = None
            depth_override_hm = None
        else:
            cand_map = dict(cand)
            depth_override_hf = cand_map.pop("h_f_mm", None)
            depth_override_hm = cand_map.pop("h_m_mm", None)
            inp = ScrewInputs(**cand_map)

        cand_map.setdefault("name", f"cand_{idx}")

        inp.screw_speed_rpm = rpm
        cand_map["screening_rpm"] = rpm

        # ---------------------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------------------

        geom = design(
            inp,
            h_f_mm=depth_override_hf,
            h_m_mm=depth_override_hm,
        )

        Q_max = geom["throughput_max_kg_hr"]

        if not np.isfinite(Q_max) or Q_max <= 0:
            results.append({
                **cand_map,
                **geom,
                "screen_status": "invalid_Qmax",
                "GMI": -np.inf,
                "gamma_int": -np.inf,
            })
            continue

        # ---------------------------------------------------------------------
        # Throughput handling
        # ---------------------------------------------------------------------

        fill_eff_raw = Q_req_kg_h / Q_max
        overloaded = fill_eff_raw > 1.0

        fill_eff = min(max(fill_eff_raw, 0.0), 1.0)
        Q_eval_kg_h = min(Q_req_kg_h, Q_max)

        Q_eval_m3_s = kg_h_to_m3_s(Q_eval_kg_h, rho=rho_melt)

        # ---------------------------------------------------------------------
        # Geometry parameters
        # ---------------------------------------------------------------------

        D_m = inp.D_mm / 1000.0
        V_surf = np.pi * D_m * rpm / 60.0
        w_slot = np.pi * D_m

        h_f_mm = geom["h_f_mm"]
        h_m_mm = geom["h_m_mm"]
        h_c_mm = 0.5 * (h_f_mm + h_m_mm)

        # Effective depths

        MIN_DEPTH = 5e-4

        h_f_eff_m = max((h_f_mm * fill_eff) / 1000.0, MIN_DEPTH)

        comp_fill = get_comp_fill(fill_eff, h_c_mm, inp.D_mm)
        h_c_eff_m = max((h_c_mm * comp_fill) / 1000.0, MIN_DEPTH)

        h_m_eff_m = max(h_m_mm / 1000.0, MIN_DEPTH)

        # ---------------------------------------------------------------------
        # Velocities
        # ---------------------------------------------------------------------

        A_feed = 0.9 * w_slot * h_f_eff_m
        A_comp = 0.9 * w_slot * h_c_eff_m
        A_metr = 0.9 * w_slot * h_m_eff_m

        V_feed = Q_eval_m3_s / A_feed
        V_comp = Q_eval_m3_s / A_comp
        V_metr = Q_eval_m3_s / A_metr

        geom["t_feed_s"] = (inp.L_feed / 1000.0) / V_feed
        geom["t_comp_s"] = (inp.L_compression / 1000.0) / V_comp
        geom["t_metr_s"] = (inp.L_meter / 1000.0) / V_metr

        # ---------------------------------------------------------------------
        # Flow solver
        # ---------------------------------------------------------------------

        dpdx_feed, gamma_feed, tau_feed, Q_C_feed, Q_P_feed, status_feed, Qmin_feed, Qmax_feed = solve_dpdx_for_slot(
            h_f_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_fn
        )

        dpdx_comp, gamma_comp, tau_comp, Q_C_comp, Q_P_comp, status_comp, Qmin_comp, Qmax_comp = solve_dpdx_for_slot(
            h_c_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_fn
        )

        dpdx_metr, gamma_metr, tau_metr, Q_C_metr, Q_P_metr, status_metr, Qmin_metr, Qmax_metr = solve_dpdx_for_slot(
            h_m_eff_m, w_slot, V_surf, Q_eval_m3_s, eta_fn
        )
        # ---------------------------------------------------------------------
        # Actual mass-based fill factors
        # ---------------------------------------------------------------------

        mass_C_feed = Q_C_feed * rho_melt * 3600.0
        mass_C_comp = Q_C_comp * rho_melt * 3600.0

        # Avoid divide-by-zero
        real_feed_fill = Q_eval_kg_h / mass_C_feed if mass_C_feed > 0 else np.nan
        real_comp_fill = Q_eval_kg_h / mass_C_comp if mass_C_comp > 0 else np.nan

        # ---------------------------------------------------------------------
        # Mixing
        # ---------------------------------------------------------------------

         

        tau_max = max(tau_feed, tau_comp, tau_metr)

        gamma_int = (
            gamma_feed * geom["t_feed_s"]
            + gamma_comp * geom["t_comp_s"]
            + gamma_metr * geom["t_metr_s"]
        )

        gdot_vec = np.array([gamma_feed, gamma_comp, gamma_metr], dtype=float)
        t_vec = np.array(
            [geom["t_feed_s"], geom["t_comp_s"], geom["t_metr_s"]],
            dtype=float
        )

        disp_ok, disp_meta = dispersive_score(
            gdot_vec,
            t_vec,
            eta_fn=eta_fn,
            preset=dispersive_preset,
            debug=False,
        )

        exposure = gdot_vec * t_vec
        dist_ok, dist_meta = distributive_score(exposure)

        disp_index = disp_meta["dispersive_index"]
        dist_index = dist_meta["distributive_index"]
        gmi = global_mixing_index(disp_index, dist_index)

        # ---------------------------------------------------------------------
        # Rheology-based pressure / power
        # ---------------------------------------------------------------------

        L_feed_m = inp.L_feed / 1000.0
        L_comp_m = inp.L_compression / 1000.0
        L_metr_m = inp.L_meter / 1000.0

        dP_feed = abs(dpdx_feed) * L_feed_m
        dP_comp = abs(dpdx_comp) * L_comp_m
        dP_metr = abs(dpdx_metr) * L_metr_m
        dP_total = dP_feed + dP_comp + dP_metr

        P_rheo_W = dP_total * Q_eval_m3_s

        omega_screw_rad_s = 2.0 * np.pi * rpm / 60.0
        T_rheo_Nm = P_rheo_W / omega_screw_rad_s if omega_screw_rad_s > 0 else np.nan

        # ---------------------------------------------------------------------
        # Collect result
        # ---------------------------------------------------------------------

        results.append({
            **cand_map,
            **geom,

            "k_f": inp.k_feed_depth if inp.k_feed_depth is not None else 0.15,
            "depth_source": geom.get("depth_source"),
            "k_f_used": geom.get("k_f_used"),

            "h_f_mm": h_f_mm,
            "h_c_mm": h_c_mm,
            "h_m_mm": h_m_mm,

            "h_f_eff_mm": h_f_eff_m * 1000.0,
            "h_c_eff_mm": h_c_eff_m * 1000.0,
            "h_m_eff_mm": h_m_eff_m * 1000.0,

            "fill_eff": fill_eff,
            "comp_fill": comp_fill,

            # Pressure gradients
            "dpdx_feed_Pa_m": dpdx_feed,
            "dpdx_comp_Pa_m": dpdx_comp,
            "dpdx_metr_Pa_m": dpdx_metr,
            # Continuous mixing indices
            "dispersive_index": disp_index,
            "distributive_index": dist_index,

            # Pressure-drop by zone
            "dP_feed_Pa": dP_feed,
            "dP_comp_Pa": dP_comp,
            "dP_metr_Pa": dP_metr,
            "dP_total_Pa": dP_total,

            # Rheology-based power / torque
            "Q_eval_m3_s": Q_eval_m3_s,
            "omega_screw_rad_s": omega_screw_rad_s,
            "P_rheo_W": P_rheo_W,
            "P_rheo_kW": P_rheo_W / 1000.0,
            "T_rheo_Nm": T_rheo_Nm,

            # Solver status
            "status_feed": status_feed,
            "status_comp": status_comp,
            "status_metr": status_metr,

            # Flow windows
            "Qmin_feed_m3_s": Qmin_feed,
            "Qmax_feed_m3_s": Qmax_feed,
            "Qmin_comp_m3_s": Qmin_comp,
            "Qmax_comp_m3_s": Qmax_comp,
            "Qmin_metr_m3_s": Qmin_metr,
            "Qmax_metr_m3_s": Qmax_metr,

            # Rheology
            "gamma_feed_1_s": gamma_feed,
            "gamma_comp_1_s": gamma_comp,
            "gamma_metr_1_s": gamma_metr,

            "tau_feed_Pa": tau_feed,
            "tau_comp_Pa": tau_comp,
            "tau_metr_Pa": tau_metr,

            # Couette / pressure flows
            "Q_C_feed_m3_s": Q_C_feed,
            "Q_P_feed_m3_s": Q_P_feed,
            "Q_C_comp_m3_s": Q_C_comp,
            "Q_P_comp_m3_s": Q_P_comp,
            "Q_C_metr_m3_s": Q_C_metr,
            "Q_P_metr_m3_s": Q_P_metr,

            # Fill metrics
            "real_feed_fill": real_feed_fill,
            "real_comp_fill": real_comp_fill,

            # Mixing
            "tau_max_Pa": tau_max,
            "gamma_int": gamma_int,
            "energy_density": disp_meta.get("energy_density"),
            "CV_exposure": dist_meta["CV"],
            "MZMI": dist_meta["MZMI"],
            "GMI": gmi,

            # Throughput
            "Q_max_kg_h": Q_max,
            "Q_req_kg_h": Q_req_kg_h,
            "Q_eval_kg_h": Q_eval_kg_h,
            "capacity_shortfall_kg_h": max(Q_req_kg_h - Q_eval_kg_h, 0.0),
            "overload_ratio": fill_eff_raw,
            "fill_eff_raw": fill_eff_raw,
            "overloaded": overloaded,

            # Dispersive thresholds
            "dispersive_preset": disp_meta.get("preset_used"),
            "tau_crit_used": disp_meta.get("tau_crit_used"),
            "gamma_target_used": disp_meta.get("gamma_target_used"),
            "energy_target_used": disp_meta.get("energy_target_used"),
            "disp_energy_ok": disp_meta.get("energy_ok"),
            "disp_stress_ok": disp_meta.get("stress_ok"),
            "disp_strain_ok": disp_meta.get("strain_ok"),
        })

    # -------------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------------

    return sorted(
        results,
        key=lambda r: (
            0 if r.get("overloaded", True) else 1,
            r.get("GMI", -np.inf),
            r.get("gamma_int", -np.inf),
        ),
        reverse=True,
    )
