#!/usr/bin/env python3
"""
viscosity_curve_v3.py
---------------------------------------------------------------
• Library functions:
      eta_poly(gdot, T=195)  → η of neat PE  (Pa·s)
      eta_spc(gdot, T=195)   → η of 50 wt % sand composite (Pa·s)

• Stand-alone CLI:
      $ python -m viscosity_curve_v3 --T 210
  writes PE/ SPC PNGs and a CSV into the working directory.
"""

# ───────────────────────── imports ──────────────────────────
from pathlib import Path
from functools import lru_cache
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ──────────────────── file / path handling ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # …/Coding_calculator
DATA_DIR     = PROJECT_ROOT / "data"
DATA_CSV     = DATA_DIR / "wpc_viscosity.csv"

# ───────────────────────── constants ───────────────────────
RHO_PE        = 0.92    # g·cm⁻³
RHO_WOOD      = 1.35
RHO_SAND      = 2.60
WT_FRAC_WPC   = 50.0    # wt % wood

MFI_PE        = 14.5    # g/10 min  (update if needed)
E_A           = 35e3    # J·mol⁻¹   (Arrhenius activation)

ETA_INT_SAND  = 3.0
PHI_M_SAND    = 0.62

R_GAS         = 8.314   # J·mol⁻¹·K⁻¹

# ───────────────────── helper functions ────────────────────
def wt2phi(wt, rho_f, rho_m):
    w = wt / 100.0
    return (w / rho_f) / ((w / rho_f) + ((1 - w) / rho_m))

def kd(phi, eta_int, phi_m):
    base = 1.0 - phi / phi_m
    return np.where(base > 1e-12, base**(-eta_int * phi_m), np.inf)

def carreau_yasuda(g, eta0, eta_inf, lam, n, a):
    """Safe Carreau–Yasuda with η∞ in the fit."""
    g   = np.asarray(g, dtype=float)
    lam = np.abs(lam) + 1e-30      # ensure λ ≥ 0
    with np.errstate(invalid="ignore", over="ignore"):
        return eta_inf + (eta0 - eta_inf) * (1.0 + (lam * g)**a)**((n - 1.0) / a)


def mfi_to_eta0(mfi):
    return 10**(4.6 - 0.5 * np.log10(mfi))

def arrhenius_shift(T, Tref):
    return np.exp(-E_A / R_GAS * (1/(T+273.15) - 1/(Tref+273.15)))

# ─────────── bounded CY fit anchored to η0_target ───────────
def fit_cy_fixed_eta0(g, eta, eta0_target):
    """
    Bounded CY fit: returns (η0, η∞, λ, n, a)
    """
    # η0, η∞, λ, n, a
    low  = [0.3*eta0_target,     1.0, 1e-6, 0.10, 0.5]
    high = [50.*eta0_target,  5e3,   1e3,  1.00, 5.0]

    p0 = [eta0_target,
          max(eta[-1]*0.5, 1.1*low[1]),      # η∞  near tail
          1.0 / g[np.argmin(eta)],           # λ   ~ 1/γ̇_min
          0.3,                               # n
          2.0]                               # a
    p0 = np.clip(p0, low, high)

    popt, _ = opt.curve_fit(carreau_yasuda, g, eta,
                            p0=p0, bounds=(low, high), maxfev=30000)
    return popt


# ─────────────── build PE & SPC models (cached) ─────────────
@lru_cache(maxsize=None)
def _build_models(T: float):
    df      = pd.read_csv(DATA_CSV, header=None, names=["gdot", "eta_wpc"])
    phi_wpc = wt2phi(WT_FRAC_WPC, RHO_WOOD, RHO_PE)

    # stage 1: PE fit @195 °C
    eta0_MFI  = mfi_to_eta0(MFI_PE)
    eta_guess = df["eta_wpc"] / kd(phi_wpc, 3.0, max(phi_wpc+0.05, 0.55))
    
    cy_195    = fit_cy_fixed_eta0(df["gdot"].values,
                                  eta_guess.values, eta0_MFI)
    eta0_195, eta_inf, lam_195, n_cy, a_cy = cy_195
    poly_195 = lambda g: carreau_yasuda(np.asarray(g), *cy_195)

    # stage 2: KD refinement (wood composite)
    def resid_kd(p):
        return (np.log(df["eta_wpc"])
                - np.log(poly_195(df["gdot"])
                         * kd(phi_wpc, p[0], p[1])))
    eta_int_wpc, phi_m_wpc = opt.leastsq(resid_kd, x0=[6.0, 0.58])[0]

    # stage 3: temperature shift
    aT     = arrhenius_shift(T, 195.0)
    eta0_T = eta0_195 * aT
    lam_T  = lam_195  * aT
    cy_T   = (eta0_T, eta_inf, lam_T, n_cy, a_cy)
    poly_T = lambda g: carreau_yasuda(np.asarray(g), *cy_T)

    # stage 4: SPC prediction
    eta_r_spc = kd(phi_wpc, ETA_INT_SAND, PHI_M_SAND)
    spc_T     = lambda g: eta_r_spc * poly_T(np.asarray(g))

    return poly_T, spc_T

# ───────────────────── public API ───────────────────────────
def eta_poly(gdot, *, T: float = 195.0):
    """Neat-PE viscosity (Pa·s)."""
    poly, _ = _build_models(float(T))
    return poly(gdot)

def eta_spc(gdot, *, T: float = 195.0):
    """Viscosity of 50 wt % sand composite (Pa·s)."""
    _, spc = _build_models(float(T))
    return spc(gdot)

# ───────────── CLI: generate PNG + CSV curves ───────────────
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=195.0,
                    help="Melt temperature for output curves [°C]")
    T_tar = ap.parse_args().T

    df = pd.read_csv(DATA_CSV, header=None, names=["gdot", "eta_wpc"])
    df["eta_poly_T"] = eta_poly(df["gdot"], T=T_tar)
    df["eta_spc_T"]  = eta_spc(df["gdot"],  T=T_tar)
    phi_wpc = wt2phi(WT_FRAC_WPC, RHO_WOOD, RHO_PE)

    # CSV
    csv_name = f"PE_SPC_viscosity_{T_tar:.0f}C.csv"
    df[["gdot", "eta_poly_T", "eta_spc_T"]].to_csv(csv_name, index=False)

    # (a) PE curve
    plt.figure(figsize=(5, 3.5))
    plt.loglog(df["gdot"], df["eta_poly_T"], 'k-', label=f"PE @ {T_tar:.0f} °C")
    plt.xlabel("Shear rate [1/s]")
    plt.ylabel("Viscosity [Pa·s]")
    plt.title("Neat-PE viscosity")
    plt.grid(True, ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PE_viscosity_{T_tar:.0f}C.png", dpi=300)

    # (b) WPC vs SPC
    plt.figure(figsize=(5.8, 3.5))
    plt.loglog(df["gdot"], df["eta_wpc"],  'o-', ms=5,
               label="WPC 50 wt % wood @195 °C")
    plt.loglog(df["gdot"], df["eta_spc_T"], 's--', ms=5,
               label=f"SPC (φ={phi_wpc:.2f}) @ {T_tar:.0f} °C")
    plt.xlabel("Shear rate [1/s]")
    plt.ylabel("Viscosity [Pa·s]")
    plt.title(f"WPC vs predicted SPC @ {T_tar:.0f} °C")
    plt.grid(True, ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"SPC_prediction_{T_tar:.0f}C.png", dpi=300)

    print(f"[OK] {csv_name} + PNGs written for {T_tar:.0f} °C")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":  # only runs when executed directly
    _cli()
