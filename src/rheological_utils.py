from __future__ import annotations

"""
rheological_utils.py
--------------------
Flexible rheology interface for early-stage screw/extrusion design.

Why this structure?
- We do not yet have capillary-rheometer data for the SPC system.
- Therefore the rheology layer must be explicit about model source and confidence.
- The rest of the codebase should depend on one stable interface only.

Supported model families:
1) wpc_surrogate
   Infer neat-polymer Carreau-Yasuda curve from measured WPC data,
   then predict SPC by replacing the filler crowding factor.
2) simple_powerlaw
   Fast fallback model for screening/debug.
3) measured_capillary
   Placeholder ready for future direct SPC measurements.

Backward compatibility:
- eta_poly(...)
- eta_wpc(...)
- eta_spc(...)
- get_curve_data(...)

The default public behaviour remains compatible with existing callers that
expect eta_spc(gdot, T=...) to return viscosity in Pa.s.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_CSV = DATA_DIR / "wpc_viscosity.csv"

RHO_PE = 0.92
RHO_WOOD = 1.35
RHO_SAND = 2.60

WPC_WOOD_WT_DEFAULT = 50.0
SPC_SAND_WT_DEFAULT = 50.0

MFI_PE = 14.5
E_A = 35e3
R_GAS = 8.314

ETA_INT_SAND = 3.0
PHI_M_SAND = 0.62

MAX_VISCOSITY = 1.0e6
MIN_GUARD_SHEAR = 1.0e-3
LOW_SHEAR_EXTRAP_FACTOR = 0.2

DEFAULT_MODEL = "wpc_surrogate"


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------
@dataclass
class RheologyModel:
    name: str
    family: str
    eta_fn: Callable[[np.ndarray], np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    def eta(self, gdot):
        g = np.asarray(gdot, dtype=float)
        eta = self.eta_fn(g)
        eta = np.asarray(eta, dtype=float)
        eta = np.minimum(np.maximum(eta, 1.0e-12), MAX_VISCOSITY)
        return float(eta) if np.isscalar(gdot) else eta


# -----------------------------------------------------------------------------
# Primitive rheology helpers
# -----------------------------------------------------------------------------
def _safe_array(x):
    return np.asarray(x, dtype=float)


def wt2phi(wt_percent: float, rho_filler: float, rho_matrix: float) -> float:
    w = float(wt_percent) / 100.0
    return (w / rho_filler) / ((w / rho_filler) + ((1.0 - w) / rho_matrix))


def kd(phi, eta_int, phi_m):
    base = 1.0 - np.asarray(phi, dtype=float) / float(phi_m)
    base = np.maximum(base, 1.0e-6)
    return base ** (-float(eta_int) * float(phi_m))


def carreau_yasuda(gdot, eta0, eta_inf, lam, n, a):
    g = _safe_array(gdot)
    lam = abs(float(lam)) + 1.0e-30
    with np.errstate(over="ignore", invalid="ignore"):
        eta = eta_inf + (eta0 - eta_inf) * (1.0 + (lam * g) ** a) ** ((n - 1.0) / a)
    return eta


def power_law(gdot, K, n):
    g = np.maximum(_safe_array(gdot), MIN_GUARD_SHEAR)
    return float(K) * g ** (float(n) - 1.0)


def mfi_to_eta0(mfi: float) -> float:
    return 10 ** (4.6 - 0.5 * np.log10(mfi))


def arrhenius_shift(T: float, Tref: float) -> float:
    return np.exp(-E_A / R_GAS * (1.0 / (T + 273.15) - 1.0 / (Tref + 273.15)))


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        if key in {"gdot", "shearrate", "shearrate1/s", "gamma", "gammadot", "shearrate(s-1)"}:
            rename_map[c] = "gdot"
        elif key in {"eta", "viscosity", "etawpc", "wpc", "viscositypa·s", "viscositypas", "viscosity(pa.s)"}:
            rename_map[c] = "eta_wpc"
        elif key in {"etaspc", "spcviscosity"}:
            rename_map[c] = "eta_spc"

    out = df.rename(columns=rename_map)
    if "gdot" not in out.columns:
        if out.shape[1] < 2:
            raise ValueError("Could not identify shear-rate column.")
        out = out.copy()
        out.columns = ["gdot", *list(out.columns[1:])]

    if "eta_wpc" not in out.columns and "eta_spc" not in out.columns:
        if out.shape[1] >= 2:
            out = out.iloc[:, :2].copy()
            out.columns = ["gdot", "eta_wpc"]
        else:
            raise ValueError("Could not identify viscosity column.")

    keep = [c for c in ["gdot", "eta_wpc", "eta_spc"] if c in out.columns]
    out = out[keep].copy()

    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["gdot"])
    for c in ["eta_wpc", "eta_spc"]:
        if c in out.columns:
            out = out.dropna(subset=[c])

    out = out.sort_values("gdot")
    pos_mask = out["gdot"] > 0.0
    if "eta_wpc" in out.columns:
        pos_mask &= out["eta_wpc"] > 0.0
    if "eta_spc" in out.columns:
        pos_mask &= out["eta_spc"] > 0.0
    out = out.loc[pos_mask]

    if out.empty:
        raise ValueError("Rheology dataset is empty after cleaning.")
    return out.reset_index(drop=True)


def read_rheology_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path is not None else DEFAULT_DATA_CSV
    if not path.exists():
        raise FileNotFoundError(f"Rheology data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
        return _normalize_columns(raw)

    try:
        raw = pd.read_csv(path)
        return _normalize_columns(raw)
    except Exception:
        raw = pd.read_csv(path, header=None)
        return _normalize_columns(raw)


# -----------------------------------------------------------------------------
# Fitting / calibration helpers
# -----------------------------------------------------------------------------
def fit_cy_bounded(g, eta, eta0_target):
    g = _safe_array(g)
    eta = _safe_array(eta)

    low = [0.2 * eta0_target, 1.0, 1.0e-6, 0.05, 0.5]
    high = [80.0 * eta0_target, 5.0e3, 1.0e4, 1.0, 5.0]

    p0 = [
        eta0_target,
        max(eta[-1] * 0.7, 1.1 * low[1]),
        1.0 / max(g[np.argmin(eta)], 1.0e-3),
        0.35,
        2.0,
    ]
    p0 = np.clip(p0, low, high)

    popt, _ = opt.curve_fit(
        carreau_yasuda,
        g,
        eta,
        p0=p0,
        bounds=(low, high),
        maxfev=50000,
    )
    return popt


def _patch_low_shear(g, eta_func, g_min_measured):
    g = _safe_array(g)
    eta = eta_func(g).copy()

    g_patch = max(LOW_SHEAR_EXTRAP_FACTOR * g_min_measured, MIN_GUARD_SHEAR)
    low = g < g_patch
    if np.any(low):
        g1 = max(g_patch, MIN_GUARD_SHEAR)
        g2 = max(min(g_min_measured, 10.0 * g1), 1.05 * g1)
        eta1 = float(eta_func(np.array([g1]))[0])
        eta2 = float(eta_func(np.array([g2]))[0])
        slope = np.log(eta2 / eta1) / np.log(g2 / g1)
        eta_low = eta1 * (np.maximum(g, MIN_GUARD_SHEAR) / g1) ** slope
        eta[low] = eta_low[low]

    return np.minimum(eta, MAX_VISCOSITY)


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build_wpc_surrogate_cached(cache_key):
    (
        data_path,
        T,
        wpc_wood_wt,
        spc_sand_wt,
        eta_int_sand,
        phi_m_sand,
    ) = cache_key

    df = read_rheology_data(data_path)
    if "eta_wpc" not in df.columns:
        raise ValueError("wpc_surrogate requires a dataset with WPC viscosity column.")

    g_meas = df["gdot"].to_numpy(dtype=float)
    eta_wpc_meas = df["eta_wpc"].to_numpy(dtype=float)

    phi_wpc = wt2phi(wpc_wood_wt, RHO_WOOD, RHO_PE)
    phi_spc = wt2phi(spc_sand_wt, RHO_SAND, RHO_PE)

    eta0_mfi = mfi_to_eta0(MFI_PE)
    kd_seed = kd(phi_wpc, 3.0, max(phi_wpc + 0.05, 0.55))
    eta_poly_guess = eta_wpc_meas / kd_seed

    cy_195 = fit_cy_bounded(g_meas, eta_poly_guess, eta0_mfi)
    eta0_195, eta_inf, lam_195, n_cy, a_cy = cy_195
    poly_195 = lambda g: carreau_yasuda(_safe_array(g), *cy_195)

    def resid_kd(p):
        eta_int_w, phi_m_w = p
        model = poly_195(g_meas) * kd(phi_wpc, eta_int_w, phi_m_w)
        return np.log(eta_wpc_meas) - np.log(np.maximum(model, 1.0e-12))

    kd_guess = np.array([6.0, 0.58])
    kd_fit, _ = opt.leastsq(resid_kd, x0=kd_guess)
    eta_int_wpc = float(max(kd_fit[0], 0.5))
    phi_m_wpc = float(np.clip(kd_fit[1], phi_wpc + 1.0e-3, 0.90))

    aT = arrhenius_shift(T, 195.0)
    cy_T = (eta0_195 * aT, eta_inf, lam_195 * aT, n_cy, a_cy)
    poly_T_raw = lambda g: carreau_yasuda(_safe_array(g), *cy_T)
    wpc_T_raw = lambda g: kd(phi_wpc, eta_int_wpc, phi_m_wpc) * poly_T_raw(_safe_array(g))
    spc_T_raw = lambda g: kd(phi_spc, eta_int_sand, phi_m_sand) * poly_T_raw(_safe_array(g))

    poly_T = lambda g: _patch_low_shear(g, poly_T_raw, float(g_meas.min()))
    wpc_T = lambda g: _patch_low_shear(g, wpc_T_raw, float(g_meas.min()))
    spc_T = lambda g: _patch_low_shear(g, spc_T_raw, float(g_meas.min()))

    meta = {
        "model_name": "wpc_surrogate",
        "model_family": "KD + Carreau-Yasuda",
        "data_source": str(data_path),
        "temperature_C": float(T),
        "is_surrogate": True,
        "calibration_basis": "WPC dataset back-calculated to neat polymer then remapped to sand",
        "phi_wpc": phi_wpc,
        "phi_spc": phi_spc,
        "eta_int_wpc": eta_int_wpc,
        "phi_m_wpc": phi_m_wpc,
        "eta_int_sand": float(eta_int_sand),
        "phi_m_sand": float(phi_m_sand),
        "cy_195": tuple(map(float, cy_195)),
        "cy_T": tuple(map(float, cy_T)),
        "g_min_measured": float(g_meas.min()),
        "g_max_measured": float(g_meas.max()),
    }
    return poly_T, wpc_T, spc_T, meta


def build_wpc_surrogate_model(
    *,
    T: float = 195.0,
    data_path: str | Path | None = None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    spc_sand_wt: float = SPC_SAND_WT_DEFAULT,
    eta_int_sand: float = ETA_INT_SAND,
    phi_m_sand: float = PHI_M_SAND,
) -> dict[str, RheologyModel]:
    path = Path(data_path).resolve() if data_path is not None else DEFAULT_DATA_CSV.resolve()
    key = (str(path), float(T), float(wpc_wood_wt), float(spc_sand_wt), float(eta_int_sand), float(phi_m_sand))
    poly_fn, wpc_fn, spc_fn, meta = _build_wpc_surrogate_cached(key)

    return {
        "poly": RheologyModel("poly_from_wpc", "wpc_surrogate", poly_fn, dict(meta, role="poly")),
        "wpc": RheologyModel("wpc_fit", "wpc_surrogate", wpc_fn, dict(meta, role="wpc")),
        "spc": RheologyModel("spc_from_wpc", "wpc_surrogate", spc_fn, dict(meta, role="spc")),
    }


def build_simple_powerlaw_model(
    *,
    K: float = 15000.0,
    n: float = 0.35,
    T: float = 195.0,
) -> RheologyModel:
    eta_fn = lambda g: power_law(g, K=K, n=n)
    return RheologyModel(
        name="simple_powerlaw",
        family="power_law",
        eta_fn=eta_fn,
        meta={
            "model_name": "simple_powerlaw",
            "model_family": "power_law",
            "data_source": None,
            "temperature_C": float(T),
            "is_surrogate": True,
            "calibration_basis": "manual fallback parameters",
            "K": float(K),
            "n": float(n),
        },
    )


def build_measured_capillary_model(
    *,
    data_path: str | Path,
    viscosity_column: str = "eta_spc",
    T: float = 195.0,
) -> RheologyModel:
    df = read_rheology_data(data_path)
    if viscosity_column not in df.columns:
        raise ValueError(f"Column '{viscosity_column}' not found in measured dataset.")

    g = df["gdot"].to_numpy(dtype=float)
    eta = df[viscosity_column].to_numpy(dtype=float)

    logg = np.log(g)
    loge = np.log(eta)

    def eta_interp(x):
        xx = np.maximum(_safe_array(x), MIN_GUARD_SHEAR)
        vals = np.interp(np.log(xx), logg, loge, left=loge[0], right=loge[-1])
        return np.exp(vals)

    return RheologyModel(
        name="measured_capillary",
        family="interpolated_measured",
        eta_fn=eta_interp,
        meta={
            "model_name": "measured_capillary",
            "model_family": "interpolated_measured",
            "data_source": str(Path(data_path).resolve()),
            "temperature_C": float(T),
            "is_surrogate": False,
            "calibration_basis": f"direct interpolation of column '{viscosity_column}'",
            "g_min_measured": float(g.min()),
            "g_max_measured": float(g.max()),
        },
    )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def get_rheology_model(
    *,
    model: str = DEFAULT_MODEL,
    role: str = "spc",
    T: float = 195.0,
    data_path: str | Path | None = None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    spc_sand_wt: float = SPC_SAND_WT_DEFAULT,
    eta_int_sand: float = ETA_INT_SAND,
    phi_m_sand: float = PHI_M_SAND,
    K: float = 15000.0,
    n: float = 0.35,
    viscosity_column: str = "eta_spc",
) -> RheologyModel:
    model = str(model).strip().lower()
    role = str(role).strip().lower()

    if model == "wpc_surrogate":
        models = build_wpc_surrogate_model(
            T=T,
            data_path=data_path,
            wpc_wood_wt=wpc_wood_wt,
            spc_sand_wt=spc_sand_wt,
            eta_int_sand=eta_int_sand,
            phi_m_sand=phi_m_sand,
        )
        if role not in models:
            raise ValueError(f"Role '{role}' not available for model '{model}'.")
        return models[role]

    if model == "simple_powerlaw":
        if role != "spc":
            raise ValueError("simple_powerlaw currently supports role='spc' only.")
        return build_simple_powerlaw_model(K=K, n=n, T=T)

    if model == "measured_capillary":
        if role != "spc":
            raise ValueError("measured_capillary currently supports role='spc' only.")
        if data_path is None:
            raise ValueError("measured_capillary requires data_path.")
        return build_measured_capillary_model(data_path=data_path, viscosity_column=viscosity_column, T=T)

    raise ValueError(f"Unknown rheology model: {model}")


# -----------------------------------------------------------------------------
# Backward-compatible public API
# -----------------------------------------------------------------------------
def eta_poly(
    gdot,
    *,
    T: float = 195.0,
    model: str = "wpc_surrogate",
    data_path=None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    **kwargs,
):
    mdl = get_rheology_model(
        model=model,
        role="poly",
        T=T,
        data_path=data_path,
        wpc_wood_wt=wpc_wood_wt,
        **kwargs,
    )
    return mdl.eta(gdot)


def eta_wpc(
    gdot,
    *,
    T: float = 195.0,
    model: str = "wpc_surrogate",
    data_path=None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    **kwargs,
):
    mdl = get_rheology_model(
        model=model,
        role="wpc",
        T=T,
        data_path=data_path,
        wpc_wood_wt=wpc_wood_wt,
        **kwargs,
    )
    return mdl.eta(gdot)


def eta_spc(
    gdot,
    *,
    T: float = 195.0,
    model: str = DEFAULT_MODEL,
    data_path=None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    spc_sand_wt: float = SPC_SAND_WT_DEFAULT,
    **kwargs,
):
    mdl = get_rheology_model(
        model=model,
        role="spc",
        T=T,
        data_path=data_path,
        wpc_wood_wt=wpc_wood_wt,
        spc_sand_wt=spc_sand_wt,
        **kwargs,
    )
    return mdl.eta(gdot)


def get_curve_data(
    *,
    T: float = 195.0,
    model: str = DEFAULT_MODEL,
    data_path=None,
    wpc_wood_wt: float = WPC_WOOD_WT_DEFAULT,
    spc_sand_wt: float = SPC_SAND_WT_DEFAULT,
    n_points: int = 120,
    gmin: float | None = None,
    gmax: float | None = None,
    **kwargs,
):
    if data_path is not None:
        df = read_rheology_data(data_path)
    else:
        df = read_rheology_data(None)

    if gmin is None:
        gmin = float(df["gdot"].min())
    if gmax is None:
        gmax = float(df["gdot"].max())

    g = np.logspace(np.log10(gmin), np.log10(gmax), int(n_points))

    out = {"gdot": g}
    if model == "wpc_surrogate":
        out["eta_poly"] = eta_poly(g, T=T, model=model, data_path=data_path, wpc_wood_wt=wpc_wood_wt, **kwargs)
        out["eta_wpc"] = eta_wpc(g, T=T, model=model, data_path=data_path, wpc_wood_wt=wpc_wood_wt, **kwargs)
    out["eta_spc"] = eta_spc(
        g,
        T=T,
        model=model,
        data_path=data_path,
        wpc_wood_wt=wpc_wood_wt,
        spc_sand_wt=spc_sand_wt,
        **kwargs,
    )
    return out


# -----------------------------------------------------------------------------
# CLI utility
# -----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=195.0)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--wpc-wood-wt", type=float, default=WPC_WOOD_WT_DEFAULT)
    ap.add_argument("--spc-sand-wt", type=float, default=SPC_SAND_WT_DEFAULT)
    ap.add_argument("--K", type=float, default=15000.0)
    ap.add_argument("--n", type=float, default=0.35)
    args = ap.parse_args()

    curves = get_curve_data(
        T=args.T,
        model=args.model,
        data_path=args.data,
        wpc_wood_wt=args.wpc_wood_wt,
        spc_sand_wt=args.spc_sand_wt,
        K=args.K,
        n=args.n,
    )
    df_out = pd.DataFrame(curves)

    csv_name = f"rheology_curves_{args.model}_{args.T:.0f}C.csv"
    df_out.to_csv(csv_name, index=False)

    plt.figure(figsize=(6.0, 4.0))
    if "eta_poly" in df_out.columns:
        plt.loglog(df_out["gdot"], df_out["eta_poly"], label=f"poly @ {args.T:.0f} C")
    if "eta_wpc" in df_out.columns:
        plt.loglog(df_out["gdot"], df_out["eta_wpc"], label=f"wpc-fit @ {args.T:.0f} C")
    plt.loglog(df_out["gdot"], df_out["eta_spc"], label=f"spc ({args.model}) @ {args.T:.0f} C")
    plt.xlabel("Shear rate [1/s]")
    plt.ylabel("Viscosity [Pa.s]")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_name = f"rheology_curves_{args.model}_{args.T:.0f}C.png"
    plt.savefig(fig_name, dpi=300)

    print(f"Saved: {csv_name}")
    print(f"Saved: {fig_name}")


if __name__ == "__main__":
    _cli()
