"""Generate a theoretical Planck 2018 best-fit Lambda-CDM CMB TT spectrum.

This script uses CAMB as the Boltzmann solver. Python configures CAMB,
retrieves the lensed scalar temperature spectrum, saves the numerical table,
and creates publication-quality figures. It does not use observational Planck
data and does not perform residual or significance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import camb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Cosmological configuration: approximate Planck 2018 best-fit flat Lambda-CDM
# =============================================================================

ELL_MAX = 2500

# Hubble constant today in km/s/Mpc.
H0 = 67.32

# Physical baryon density parameter, Omega_b h^2, dimensionless.
OMBH2 = 0.02238

# Physical cold dark matter density parameter, Omega_c h^2, dimensionless.
OMCH2 = 0.12011

# Thomson scattering optical depth to reionization, dimensionless.
TAU = 0.0543

# Scalar primordial curvature power amplitude at CAMB's pivot scale.
# CAMB expects As directly, not ln(10^10 As), so no conversion is applied.
AS = 2.1005e-9

# Scalar spectral index of primordial curvature perturbations, dimensionless.
NS = 0.96605

# Sum of neutrino masses in eV.
MNU = 0.06

# Spatial curvature density parameter, dimensionless. Zero means spatially flat.
OMK = 0.0

# CAMB's standard scalar primordial-power pivot scale, in Mpc^-1.
PIVOT_SCALAR = 0.05

# Effective number of relativistic neutrino species used by CAMB's default
# Planck-like setup. This is stated explicitly for reproducibility.
NUM_MASSIVE_NEUTRINOS = 1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
SPECTRUM_CSV = TABLE_DIR / "camb_planck2018_tt_spectrum.csv"
FULL_PNG = FIGURE_DIR / "camb_planck2018_tt_full.png"
FULL_PDF = FIGURE_DIR / "camb_planck2018_tt_full.pdf"
LOWELL_PNG = FIGURE_DIR / "camb_planck2018_tt_lowell.png"
LOWELL_PDF = FIGURE_DIR / "camb_planck2018_tt_lowell.pdf"


@dataclass(frozen=True)
class CosmologyConfig:
    """Container for the cosmological parameters used in the CAMB run."""

    ell_max: int = ELL_MAX
    h0: float = H0
    ombh2: float = OMBH2
    omch2: float = OMCH2
    tau: float = TAU
    scalar_amplitude: float = AS
    scalar_spectral_index: float = NS
    neutrino_mass_sum_ev: float = MNU
    curvature_omega_k: float = OMK
    pivot_scalar_mpc_inverse: float = PIVOT_SCALAR
    num_massive_neutrinos: int = NUM_MASSIVE_NEUTRINOS


def create_camb_parameters(config: CosmologyConfig) -> camb.CAMBparams:
    """Create CAMB parameters for a flat six-parameter Lambda-CDM spectrum."""

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=config.h0,
        ombh2=config.ombh2,
        omch2=config.omch2,
        omk=config.curvature_omega_k,
        mnu=config.neutrino_mass_sum_ev,
        tau=config.tau,
        num_massive_neutrinos=config.num_massive_neutrinos,
    )
    pars.InitPower.set_params(
        As=config.scalar_amplitude,
        ns=config.scalar_spectral_index,
        pivot_scalar=config.pivot_scalar_mpc_inverse,
    )
    pars.set_for_lmax(config.ell_max, lens_potential_accuracy=1)
    pars.WantTensors = False
    return pars


def calculate_tt_spectrum(
    pars: camb.CAMBparams, ell_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate lensed scalar TT spectra as Cl and Dl arrays in microkelvin^2.

    CAMB calculates the cosmology once with ``get_results``. The resulting
    object then returns the lensed scalar spectrum in both conventions:
    ``raw_cl=False`` gives D_ell = ell(ell + 1) C_ell / (2 pi), while
    ``raw_cl=True`` gives raw C_ell. Both are in microkelvin squared here.
    """

    results = camb.get_results(pars)

    lensed_dl = results.get_lensed_scalar_cls(
        lmax=ell_max,
        CMB_unit="muK",
        raw_cl=False,
    )

    lensed_cl = results.get_lensed_scalar_cls(
        lmax=ell_max,
        CMB_unit="muK",
        raw_cl=True,
    )

    ell = np.arange(lensed_dl.shape[0], dtype=int)
    dl_tt = lensed_dl[:, 0]
    cl_tt = lensed_cl[:, 0]

    return ell, cl_tt, dl_tt


def build_output_dataframe(
    ell: np.ndarray, cl_tt: np.ndarray, dl_tt: np.ndarray
) -> pd.DataFrame:
    """Build the scientific output table, excluding monopole and dipole."""

    mask = ell >= 2
    dataframe = pd.DataFrame(
        {
            "ell": ell[mask],
            "Cl_TT_uK2": cl_tt[mask],
            "Dl_TT_uK2": dl_tt[mask],
        }
    )
    return dataframe


def save_spectrum(dataframe: pd.DataFrame, csv_path: Path) -> None:
    """Save the spectrum table to CSV."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)


def create_full_spectrum_plot(dataframe: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Create the full theoretical TT spectrum figure for 2 <= ell <= 2500."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = dataframe[(dataframe["ell"] >= 2) & (dataframe["ell"] <= ELL_MAX)]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(plot_data["ell"], plot_data["Dl_TT_uK2"], color="#1f77b4", linewidth=1.7)
    ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]", fontsize=12)
    ax.set_title(
        "Theoretical Planck 2018 Best-Fit Lambda-CDM TT Spectrum Generated with CAMB",
        fontsize=13,
    )
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlim(2, ELL_MAX)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def create_low_ell_plot(dataframe: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    """Create the low-multipole theoretical TT figure for 2 <= ell <= 29."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    low_ell = dataframe[(dataframe["ell"] >= 2) & (dataframe["ell"] <= 29)]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(
        low_ell["ell"],
        low_ell["Dl_TT_uK2"],
        color="#2ca02c",
        marker="o",
        markersize=4.5,
        linewidth=1.4,
    )
    ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]", fontsize=12)
    ax.set_title(
        "Low-Multipole Theoretical CAMB TT Spectrum Only",
        fontsize=13,
    )
    ax.set_xticks(np.arange(2, 30, 3))
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlim(1.5, 29.5)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def cosmology_summary(config: CosmologyConfig) -> dict[str, Any]:
    """Return cosmological parameters in a printable dictionary."""

    return {
        "H0_km_s_Mpc": config.h0,
        "ombh2": config.ombh2,
        "omch2": config.omch2,
        "tau": config.tau,
        "As": config.scalar_amplitude,
        "ns": config.scalar_spectral_index,
        "mnu_eV": config.neutrino_mass_sum_ev,
        "omk": config.curvature_omega_k,
        "pivot_scalar_Mpc^-1": config.pivot_scalar_mpc_inverse,
        "ell_max": config.ell_max,
        "num_massive_neutrinos": config.num_massive_neutrinos,
    }


def main() -> None:
    """Run the full theoretical spectrum-generation workflow."""

    config = CosmologyConfig()
    pars = create_camb_parameters(config)
    ell, cl_tt, dl_tt = calculate_tt_spectrum(pars, config.ell_max)
    dataframe = build_output_dataframe(ell, cl_tt, dl_tt)

    save_spectrum(dataframe, SPECTRUM_CSV)
    create_full_spectrum_plot(dataframe, FULL_PNG, FULL_PDF)
    create_low_ell_plot(dataframe, LOWELL_PNG, LOWELL_PDF)

    print("\nCAMB theoretical TT spectrum generation complete")
    print(f"CAMB version: {camb.__version__}")
    print(f"ell range generated: {int(dataframe['ell'].min())} to {int(dataframe['ell'].max())}")
    print(f"number of rows saved: {len(dataframe)}")
    print(f"output CSV location: {SPECTRUM_CSV}")
    print("figure locations:")
    for figure_path in (FULL_PNG, FULL_PDF, LOWELL_PNG, LOWELL_PDF):
        print(f"  - {figure_path}")
    print("cosmological parameters used:")
    for name, value in cosmology_summary(config).items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
