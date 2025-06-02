#!/usr/bin/env python3
"""
viscosity_curve_v3.py · 2025-06-02
---------------------------------------------------------------
Generate viscosity curves for:
  • neat polyethylene (Carreau–Yasuda) at any melt temperature
  • sand–plastic composite (50 wt %) at the same temperature

Outputs
-------
  PE_viscosity_<T>C.png
  SPC_prediction_<T>C.png
  PE_SPC_viscosity_<T>C.csv
"""

# ─────────────────── imports ─────────────────────────────────
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ─────────────────── constants ───────────────────────────────
DATA_CSV = Path(__file__).parent.parent / "data" / "wpc_viscosity.csv"     # γ̇ , η_wpc  @195 °C
RHO_PE     = 0.92                            # g cm-3
RHO_WOOD   = 1.35                            # g cm-3
RHO_SAND   = 2.60                            # g cm-3
WT_FRAC_WPC = 50.0                           # wt % wood
# --- neat-PE info -----------------------------------------------------------
MFI_PE     = 14.5        # g / 10 min   ← update if you have a better value
E_A        = 35e3        # J mol-1      Arrhenius activation energy (PE)
# --- filler model (Krieger–Dougherty) ---------------------------------------
ETA_INT_SAND = 3.0       # intrinsic viscosity  (≈2.5–3.5 spherical; 3.5–5 angular)
PHI_M_SAND   = 0.62      # maximum packing fraction (≈0.57–0.63)
# ---------------------------------------------------------------------------

R_GAS = 8.314            # J mol-1 K-1

# ─────────────────── helper functions ───────────────────────
def wt2phi(wt, rho_f, rho_m):
    """Convert weight-% filler to volume fraction."""
    w = wt / 100.0
    return (w / rho_f) / ((w / rho_f) + ((1 - w) / rho_m))

def kd(phi, eta_int, phi_m):
    """Krieger–Dougherty viscosity ratio η_r(φ)."""
    base = 1.0 - phi / phi_m
    return np.where(base > 1e-12, base ** (-eta_int * phi_m), np.inf)

def carreau_yasuda(g, eta0, etainf, lam, n, a):
    return etainf + (eta0 - etainf) * (1 + (lam * g) ** a) ** ((n - 1) / a)

def mfi_to_eta0(mfi):
    """Bagley–Tordella-type correlation for PE at 190-200 °C."""
    return 10 ** (4.6 - 0.5 * np.log10(mfi))  # Pa s

def arrhenius_shift(T, Tref):
    return np.exp(-E_A / R_GAS * (1 / (T + 273.15) - 1 / (Tref + 273.15)))

def fit_cy_fixed_eta0(g, eta, eta0_target):
    """
    Fit Carreau–Yasuda with bounds tied to a target η0.
    Returns (η0, η∞, λ, n, a)
    """
    low  = [0.3 * eta0_target,     1,   1e-4, 0.10, 0.5]
    high = [50.0 * eta0_target, 5e3, 1.0e4, 1.00, 5.0]

    # crude guesses
    p0 = [eta0_target,
          max(eta[-1] * 0.5, 1.1 * low[1]),     # η∞
          1.0 / g[np.argmin(eta)],              # λ
          0.3,                                  # n
          2.0]                                  # a
    p0 = np.clip(p0, low, high)

    popt, _ = opt.curve_fit(carreau_yasuda, g, eta,
                            p0=p0, bounds=(low, high), maxfev=30000)
    return popt

# ─────────────────── CLI argument ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=float, default=195.0,
                help="Melt temperature [°C] (default 195)")
T_tar = ap.parse_args().T
T_ref = 195.0

# ─────────────────── data load ───────────────────────────────
df = pd.read_csv(DATA_CSV, header=None, names=["gdot", "eta_wpc"])
phi_wpc = wt2phi(WT_FRAC_WPC, RHO_WOOD, RHO_PE)

# ─────────────────── stage ① : polymer fit (@195 °C) ────────
eta0_MFI = mfi_to_eta0(MFI_PE)
print(f"[anchor] η0 from MFI={MFI_PE:.1f} → {eta0_MFI:.2e} Pa·s")

# rough KD factor just to remove the obvious filler boost
eta_poly_guess = df["eta_wpc"] / kd(phi_wpc, 3.0, max(phi_wpc + 0.05, 0.55))
cy_195 = fit_cy_fixed_eta0(df["gdot"].values, eta_poly_guess.values, eta0_MFI)
eta0_195, etainf, lam_195, n_cy, a_cy = cy_195
poly_195 = lambda g: carreau_yasuda(np.asarray(g), *cy_195)

print(f"[CY-fit]  η0={eta0_195:.2e}  λ={lam_195:.2e}  n={n_cy:.2f}  a={a_cy:.1f}")

# ─────────────────── stage ② : KD fit for WPC ───────────────
def resid_kd(p):
    return np.log(df["eta_wpc"]) - np.log(poly_195(df["gdot"]) * kd(phi_wpc, p[0], p[1]))

eta_int_wpc, phi_m_wpc = opt.leastsq(resid_kd, x0=[6.0, 0.58])[0]
print(f"[KD-fit]  WPC  η_int={eta_int_wpc:.2f}  φ_m={phi_m_wpc:.3f}")

# polymer viscosity at 195 °C after KD refinement
eta_poly_195 = df["eta_wpc"] / kd(phi_wpc, eta_int_wpc, phi_m_wpc)

# ─────────────────── temperature shift ──────────────────────
aT = arrhenius_shift(T_tar, T_ref)
eta0_T, lam_T = eta0_195 * aT, lam_195 * aT
cy_T = (eta0_T, etainf, lam_T, n_cy, a_cy)
poly_T = lambda g: carreau_yasuda(np.asarray(g), *cy_T)
print(f"[CY-shift] → {T_tar:.0f} °C  η0={eta0_T:.2e}  λ={lam_T:.2e}")

# ─────────────────── SPC prediction ─────────────────────────
eta_r_spc = kd(phi_wpc, ETA_INT_SAND, PHI_M_SAND)
df["eta_poly_T"] = poly_T(df["gdot"])
df["eta_spc_T"]  = eta_r_spc * df["eta_poly_T"]

# ─────────────────── export & plots ─────────────────────────
csv_name = f"PE_SPC_viscosity_{T_tar:.0f}C.csv"
df[["gdot", "eta_poly_T", "eta_spc_T"]].to_csv(csv_name, index=False)

# (a) neat PE curve
plt.figure(figsize=(5, 3.5))
plt.loglog(df["gdot"], df["eta_poly_T"], 'k-',
           label=f"PE @ {T_tar:.0f} °C")
plt.xlabel("Shear rate [1 / s]")
plt.ylabel("Viscosity [Pa·s]")
plt.title("Neat-PE viscosity (Carreau–Yasuda)")
plt.grid(which="both", ls="--", alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"PE_viscosity_{T_tar:.0f}C.png", dpi=300)

# (b) WPC vs SPC
plt.figure(figsize=(5.8, 3.5))
plt.loglog(df["gdot"], df["eta_wpc"],  'o-', ms=5,
           label="WPC 50 wt % wood @195 °C")
plt.loglog(df["gdot"], df["eta_spc_T"], 's--', ms=5,
           label=f"SPC (φ={phi_wpc:.2f}) @ {T_tar:.0f} °C")
plt.xlabel("Shear rate [1 / s]")
plt.ylabel("Viscosity [Pa·s]")
plt.title(f"WPC vs predicted SPC @ {T_tar:.0f} °C")
plt.grid(which="both", ls="--", alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"SPC_prediction_{T_tar:.0f}C.png", dpi=300)

print(f"[OK] written: {csv_name}, PNGs for {T_tar:.0f} °C")
