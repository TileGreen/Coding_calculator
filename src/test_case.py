# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:08:16 2025

@author: hp
"""

from screw_design import design
from config import ScrewInputs
from geometry_screening import zone_shear_rate_drag
from mixing_matrix import distributive_score
import numpy as np

# Choose a known candidate (manually or via index)
test_case = ScrewInputs(D_mm=125, L_over_D=17.0, compression_ratio=3.0)
rpm = 90

# Design geometry to extract h_f and h_m
geom = design(test_case)
h_f, h_m = geom["h_f_mm"], geom["h_m_mm"]

# Compute shear rates
g_feed = zone_shear_rate_drag(test_case.D_mm, h_f, rpm)
g_metr = zone_shear_rate_drag(test_case.D_mm, h_m, rpm)

phi_field = np.array([g_feed, g_metr])
ok, meta = distributive_score(phi_field, cv_target=0.05, mzmi_target=0.90)

print(f"phi_field = {phi_field}")
print(f"→ CV = {meta['CV']:.3f}, MZMI = {meta['MZMI']:.3f}, OK? = {ok}")
