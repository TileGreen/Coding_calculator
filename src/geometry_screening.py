# -*- coding: utf-8 -*-
"""
geometry_screening.py

Screen dozens of candidate geometries via a 1D continuum model to estimate local shear rates,
residence times, and produce a composite ranking score.
"""
from typing import List, Dict
import numpy as np
from config import ScrewInputs
from screw_design import design
from extruder_design import size_motor
from rheological_utils import eta_spc


def kg_h_to_m3_s(Q_kg_h: float, rho: float = 1000.0) -> float:
    """
    Convert mass flow rate [kg/h] to volumetric flow rate [m^3/s], given density [kg/m^3].
    """
    return Q_kg_h / rho / 3600.0

def compute_residence_bins(shear_rates: List[float], times: List[float], bins: int=10) -> np.ndarray:
    # weight residence probability in each bin proportional to shear* time
    weights = np.array(shear_rates)*np.array(times)
    # normalize across zones
    probs = weights/weights.sum()
    # assign uniform within each bin segment
    p_j = np.repeat(probs/bins, bins)
    return p_j
def compute_entropy(p_j: np.ndarray) -> float:
    p = p_j[p_j>0]
    return -np.sum(p*np.log(p))


def zone_shear_rate(Q: float, D_mm: float, h_mm: float) -> float:
    """
    Estimate average shear rate in a screw channel zone.
    Q: volumetric flow rate [m^3/s]
    D_mm: screw diameter [mm]
    h_mm: channel depth [mm]
    Returns shear rate [1/s].
    """
    R = D_mm / 2e3  # radius in meters
    A = 2 * np.pi * R * (h_mm / 1e3)
    U = Q / A
    return U / (h_mm / 1e3)


def zone_residence_time(L_mm: float, Q: float, D_mm: float, h_mm: float) -> float:
    """
    Estimate residence time in a screw channel zone.
    L_mm: zone length [mm]
    Q: volumetric flow rate [m^3/s]
    D_mm: screw diameter [mm]
    h_mm: channel depth [mm]
    Returns time [s].
    """
    R = D_mm / 2e3
    A = 2 * np.pi * R * (h_mm / 1e3)
    U = Q / A
    L = L_mm / 1e3
    return L / U


def screen_geometries(
    candidates: List[ScrewInputs],
    Q_kg_h: float,
    rpm: float
) -> List[Dict[str, float]]:
    """
    Screen a list of ScrewInputs to estimate shear rates and residence times.

    Returns a list of dicts with keys:
      - 'name': candidate identifier
      - 'theta_deg': optimum helix angle [deg]
      - 'gamma_feed': shear rate in feed zone [1/s]
      - 'gamma_meter': shear rate in metering zone [1/s]
      - 't_feed': residence time in feed zone [s]
      - 't_meter': residence time in metering zone [s]
      - 'score': composite ranking metric (higher is better)
    """
    # Convert to volumetric flow
    rho = candidates[0].bulk_density
    Q_m3_s = kg_h_to_m3_s(Q_kg_h, rho)
    results = []

    for inp in candidates:
        geom = design(inp)
        # depths from geometry
        h_f = geom.get('h_f_mm', geom.get('h_f'))
        h_m = geom.get('h_m_mm', geom.get('h_m'))
        # zone lengths
        L_f, L_m = inp.L_feed, inp.L_meter

        gamma_feed = zone_shear_rate(Q_m3_s, inp.D_mm, h_f)
        gamma_meter = zone_shear_rate(Q_m3_s, inp.D_mm, h_m)
        t_feed = zone_residence_time(L_f, Q_m3_s, inp.D_mm, h_f)
        t_meter = zone_residence_time(L_m, Q_m3_s, inp.D_mm, h_m)

        # composite score: sum of shear rates over total residence time
        score = (gamma_feed + gamma_meter) / (t_feed + t_meter)

        results.append({
            'name': getattr(inp, 'name', f"D{inp.D_mm}_CR{inp.compression_ratio}"),
            'theta_deg': geom['theta_deg'],
            'gamma_feed': gamma_feed,
            'gamma_meter': gamma_meter,
            't_feed': t_feed,
            't_meter': t_meter,
            'score': score
        })

    # sort descending by score
    return sorted(results, key=lambda r: r['score'], reverse=True)
