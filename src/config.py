# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:44:03 2025

Central configuration for screw and motor inputs, plus candidate loader.
"""
from dataclasses import dataclass
from pathlib import Path
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Screw geometry and process inputs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScrewInputs:
    D_mm: float = 90.0              # Screw outer diameter [mm]
    L_over_D: float = 25.0          # Total length-to-diameter ratio
    compression_ratio: float = 3.0  # Compression ratio (h_f/h_m)
    power_law_index_n: float = 0.4   # Viscosity power-law index (n)
    screw_speed_rpm: float = 15.0    # Screw speed [rpm]
    bulk_density: float = 1150.0     # Pellet bulk density [kg/m^3]
    fill_factor: float = 0.60        # Effective fill fraction
    k_feed_depth: float | None = None   # ← add this line
    # Zone length fractions (must sum to 1.0)
    feed_zone_frac: float = 0.30         # Feed-zone fraction of L
    compression_zone_frac: float = 0.20  # Compression-zone fraction of L
    meter_zone_frac: float = 0.50        # Metering-zone fraction of L

    @property
    def barrel_length_mm(self) -> float:
        """Compute total barrel length [mm] from L/D and diameter."""
        return self.L_over_D * self.D_mm

    @property
    def L_feed(self) -> float:
        """Feed-zone length [mm]."""
        return self.barrel_length_mm * self.feed_zone_frac

    @property
    def L_compression(self) -> float:
        """Compression-zone length [mm]."""
        return self.barrel_length_mm * self.compression_zone_frac

    @property
    def L_meter(self) -> float:
        """Metering-zone length [mm]."""
        return self.barrel_length_mm * self.meter_zone_frac

# ──────────────────────────────────────────────────────────────────────────────
# Motor sizing inputs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MotorSizingInputs:
    kv: float = 0.045                 # Motor constant [V/(rad/s)]
    screw_rpm: float = 80.0           # Design screw speed [rpm]
    melt_density_kg_m3: float = 1500.0
    specific_torque_Nm_cm3: float = 12.0
    safety_factor: float = 1.15

# ──────────────────────────────────────────────────────────────────────────────
# Utility: load candidates for screening from a YAML file
# ──────────────────────────────────────────────────────────────────────────────
def load_candidates(path: Path) -> list[ScrewInputs]:
    """
    Load a list of ScrewInputs from a YAML file. Each entry should match the
    ScrewInputs fields.

    Example YAML:
      - D_mm: 90
        L_over_D: 25
        compression_ratio: 3.0
        feed_zone_frac: 0.25
        ...
    """
    data = yaml.safe_load(path.read_text())
    candidates = []
    for entry in data:
        # Only pass known fields; ignore extras
        params = {k: v for k, v in entry.items() if k in ScrewInputs.__dataclass_fields__}
        candidates.append(ScrewInputs(**params))
    return candidates
