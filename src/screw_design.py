import pandas as pd
import numpy as np
import math
#from ace_tools import display_dataframe_to_user

# ─────────────────────────────────────────────
# USER‑CONFIGURABLE INPUTS  (edit these values)
# ─────────────────────────────────────────────
D_mm = 125.0                 # Barrel / screw diameter [mm]
L_over_D = 25                # Total L/D of screw
compression_ratio = 3.0      # h_feed : h_meter
power_law_index_n = 0.4      # From rheology (n ≈ 0.4 for HDPE)
screw_speed_rpm = 60         # Design screw speed [rev/min]
bulk_density = 1150          # Approx. PE‑sand bulk density [kg/m³]
fill_factor = 0.6            # Solids fill fraction in feed section

# Zonal length fractions  (feed, transition, metering)
zone_frac = {"Feed": 0.30, "Transition": 0.48, "Metering": 0.22}

# Empirical constants
k_feed_depth = 0.15          # h_feed = k * D

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def optimum_helix_angle(n: float) -> float:
    """
    Simple linear fit to Rauwendaal Fig 5.4:
    θ_opt ≈ 27° − 22·n   [valid 0 ≤ n ≤ 1]
    """
    return max(5.0, 27.0 - 22.0 * n)

def channel_depths(D: float, k_f: float, CR: float):
    h_f = k_f * D
    h_m = h_f / CR
    return h_f, h_m

def screw_pitch(D: float, theta_deg: float):
    theta_rad = math.radians(theta_deg)
    return math.pi * D * math.tan(theta_rad)

def volumetric_throughput(rho: float, D: float, h_m: float,
                          N_rpm: float, phi_fill: float):
    N_rps = N_rpm / 60.0
    channel_area = math.pi * D * h_m  # very crude single‑flight estimate
    return rho * channel_area * N_rps * phi_fill   # [kg/s]

# ─────────────────────────────────────────────
# CORE CALCULATIONS
# ─────────────────────────────────────────────
theta = optimum_helix_angle(power_law_index_n)
h_feed, h_meter = channel_depths(D_mm, k_feed_depth, compression_ratio)
pitch_all = screw_pitch(D_mm, theta)
Q_kg_s = volumetric_throughput(bulk_density, D_mm/1000, h_meter/1000,
                               screw_speed_rpm, fill_factor)
Q_kg_hr = Q_kg_s * 3600

# Build a zone‑wise DataFrame
rows = []
cumulative_LD = 0.0
for zone, frac in zone_frac.items():
    length_LD = L_over_D * frac
    start_LD = cumulative_LD
    end_LD = cumulative_LD + length_LD
    depth = h_feed if zone == "Feed" else (
        h_feed - (h_feed - h_meter) * min(1.0, (end_LD - start_LD) / length_LD)
    )
    rows.append({
        "Zone": zone,
        "Start (L/D)": round(start_LD, 2),
        "End (L/D)": round(end_LD, 2),
        "Channel depth h [mm]": round(depth, 2),
        "Pitch p [mm]": round(pitch_all, 1),
        "Helix angle θ [deg]": round(theta, 1)
    })
    cumulative_LD += length_LD

df = pd.DataFrame(rows)

# Display results
#display_dataframe_to_user("Screw Design Results", df)

print(f"\nEstimated metering‑zone depth h_m = {h_meter:.2f} mm")
print(f"Optimum helix angle θ_opt ≈ {theta:.1f}°  → pitch ≈ {pitch_all:.1f} mm")
print(f"Crude throughput estimate @ {screw_speed_rpm} rpm: "
      f"{Q_kg_hr:.0f} kg/h")

