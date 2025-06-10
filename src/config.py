# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:44:03 2025

@author: hp
"""

from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# Central place for every default value – edit once, propagate everywhere.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScrewInputs:
    D_mm: float = 125.0            # screw OD
    L_over_D: float = 15.6
    compression_ratio: float = 3.0
    power_law_index_n: float = 0.4 # viscosity model
    screw_speed_rpm: float = 80.0
    bulk_density: float = 1150.0   # pellets, kg·m⁻³
    fill_factor: float = 0.60

@dataclass
class MotorSizingInputs:
    kv: float = 0.045
    screw_rpm: float = 80.0
    melt_density_kg_m3: float = 1500.0
    specific_torque_Nm_cm3: float = 12.0
    safety_factor: float = 1.15
