"""
screw_design.py
Pure helpers for single-screw geometry & throughput.
Nothing executes on import; call design().
"""
from math import pi
from config import ScrewInputs
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _optimum_helix(n: float) -> float:
    """Tadmor & Klein correlation for optimum helix angle [deg]."""
    return max(5.0, 27.0 - 22.0 * n)


def _channel_depths(D_mm: float, k_feed: float, CR: float):
    h_f = k_feed * D_mm
    h_m = h_f / CR
    return h_f, h_m


def design(
    inp: ScrewInputs = ScrewInputs(),
    *,
    k_feed_depth: float | None = None,
    h_f_mm: float | None = None,
    h_m_mm: float | None = None,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute basic screw layout + projected throughput.

    Depth precedence
    ----------------
    1) explicit h_f_mm / h_m_mm
    2) k_feed_depth (argument or inp.k_feed_depth)
    3) default k_f = 0.15

    Rules
    -----
    - If only h_f_mm is given, h_m_mm = h_f_mm / CR
    - If only h_m_mm is given, h_f_mm = h_m_mm * CR
    """

    feed_rate_kg_hr: float | None = None
    D_m = inp.D_mm / 1000.0
    theta_deg = _optimum_helix(inp.power_law_index_n)

    # ---- Resolve depths
    explicit_hf = h_f_mm is not None
    explicit_hm = h_m_mm is not None

    if explicit_hf or explicit_hm:
        if h_f_mm is None:
            h_f_mm = float(h_m_mm) * float(inp.compression_ratio)
        elif h_m_mm is None:
            h_m_mm = float(h_f_mm) / float(inp.compression_ratio)

        depth_source = "direct_depth"
        k_f = float(h_f_mm) / float(inp.D_mm)
    else:
        k_f = (
            inp.k_feed_depth
            if inp.k_feed_depth is not None
            else (0.15 if k_feed_depth is None else k_feed_depth)
        )
        h_f_mm, h_m_mm = _channel_depths(inp.D_mm, k_f, inp.compression_ratio)
        depth_source = "k_feed_depth"

    h_f_mm = float(h_f_mm)
    h_m_mm = float(h_m_mm)

    # ---- Validation
    if h_f_mm <= 0 or h_m_mm <= 0:
        raise ValueError("Channel depths must be > 0.")
    if h_f_mm < h_m_mm:
        raise ValueError("Expected h_f_mm >= h_m_mm for compression screw geometry.")
    if inp.compression_ratio <= 0:
        raise ValueError("compression_ratio must be > 0.")

    # ---- Simple open-channel throughput estimate
    channel_area_m2 = pi * D_m * (h_m_mm / 1000.0)
    Q_kg_s = (
        inp.bulk_density
        * channel_area_m2
        * inp.screw_speed_rpm / 60.0
        * inp.fill_factor
    )
    Q_kg_hr = Q_kg_s * 3600.0

    if feed_rate_kg_hr is not None:
        phi_eff = min(feed_rate_kg_hr / Q_kg_hr, 1.0)
        Q_actual_hr = phi_eff * Q_kg_hr
    else:
        phi_eff = inp.fill_factor
        Q_actual_hr = Q_kg_hr

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.debug(
        f"depth_source={depth_source}  k_f={k_f:.3f}  "
        f"h_f={h_f_mm:.2f} mm  h_m={h_m_mm:.2f} mm"
    )

    return {
        "theta_deg": theta_deg,
        "h_f_mm": h_f_mm,
        "h_m_mm": h_m_mm,
        "k_f_used": k_f,
        "depth_source": depth_source,
        "throughput_max_kg_hr": Q_kg_hr,
        "fill_factor_eff": phi_eff,
        "throughput_actual_kg_hr": Q_actual_hr,
    }
