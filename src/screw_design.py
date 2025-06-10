"""
screw_design.py
Pure helpers for single-screw geometry & throughput.
Nothing executes on import; call design().
"""
from math import pi
from config import ScrewInputs
from typing import Dict

# ──────────────────────────────────────────────────────────────────────────────
def _optimum_helix(n: float) -> float:
    """Tadmor & Klein correlation for optimum helix angle [deg]."""
    return max(5.0, 27.0 - 22.0 * n)

def _channel_depths(D_mm: float, k_feed: float, CR: float):
    h_f = k_feed * D_mm       # feed-zone depth
    h_m = h_f / CR            # metering depth
    return h_f, h_m
# ──────────────────────────────────────────────────────────────────────────────
def design(inp: ScrewInputs = ScrewInputs(), *, k_feed_depth: float = 0.15) -> Dict[str, float]:
    """
    Compute basic screw layout + projected throughput.

    Returns
    -------
    dict with keys:
        theta_deg           optimum helix angle
        h_f_mm, h_m_mm      channel depths
        throughput_kg_hr    calculated Q
    """
    D_m = inp.D_mm / 1000.0
    theta_deg = _optimum_helix(inp.power_law_index_n)
    h_f_mm, h_m_mm = _channel_depths(inp.D_mm, k_feed_depth, inp.compression_ratio)

    # simple open-channel estimate:
    channel_area_m2 = pi * D_m * (h_m_mm / 1000.0)
    Q_kg_s = (inp.bulk_density * channel_area_m2 *
              inp.screw_speed_rpm / 60.0 * inp.fill_factor)
    Q_kg_hr = Q_kg_s * 3600.0

    return {
        "theta_deg": theta_deg,
        "h_f_mm": h_f_mm,
        "h_m_mm": h_m_mm,
        "throughput_kg_hr": Q_kg_hr,
    }
