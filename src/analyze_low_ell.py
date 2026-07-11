"""Compare Planck 2018 low-ell TT observations with CAMB theory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PLANCK_FILE = PROJECT_ROOT / "data" / "raw" / "COM_PowerSpect_CMB-TT-full_R3.01.txt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PLANCK_LOWELL = PROCESSED_DIR / "planck2018_tt_lowell.csv"
CAMB_SPECTRUM = PROJECT_ROOT / "outputs" / "tables" / "camb_planck2018_tt_spectrum.csv"
COMPARISON_CSV = PROJECT_ROOT / "outputs" / "tables" / "lowell_planck_camb_comparison.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

LOW_ELL_MIN = 2
LOW_ELL_MAX = 29
LOW_ELL_RANGE = np.arange(LOW_ELL_MIN, LOW_ELL_MAX + 1)

PLANCK_COLUMNS = [
    "ell",
    "planck_Dl_uK2",
    "planck_error_lower_uK2",
    "planck_error_upper_uK2",
]

FIGURE_PATHS = {
    "comparison_png": FIGURE_DIR / "planck_camb_lowell_comparison.png",
    "comparison_pdf": FIGURE_DIR / "planck_camb_lowell_comparison.pdf",
    "residuals_png": FIGURE_DIR / "planck_camb_lowell_residuals.png",
    "residuals_pdf": FIGURE_DIR / "planck_camb_lowell_residuals.pdf",
    "fractional_png": FIGURE_DIR / "planck_camb_lowell_fractional_difference.png",
    "fractional_pdf": FIGURE_DIR / "planck_camb_lowell_fractional_difference.pdf",
    "normalized_png": FIGURE_DIR / "planck_camb_lowell_normalized_residuals.png",
    "normalized_pdf": FIGURE_DIR / "planck_camb_lowell_normalized_residuals.pdf",
}


def require_file(path: Path, description: str) -> None:
    """Raise a helpful error if a required input file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def parse_planck_tt_file(raw_path: Path) -> pd.DataFrame:
    """Parse the official whitespace-separated Planck PR3 unbinned TT file."""

    require_file(raw_path, "official raw Planck TT file")
    dataframe = pd.read_csv(
        raw_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=PLANCK_COLUMNS,
        engine="python",
    )

    if dataframe.empty:
        raise ValueError(f"No data rows were parsed from {raw_path}")

    ell_values = dataframe["ell"].to_numpy(dtype=float)
    if not np.all(np.isfinite(ell_values)):
        raise ValueError("The Planck ell column contains non-finite values.")
    if not np.allclose(ell_values, np.round(ell_values), rtol=0.0, atol=1e-12):
        raise ValueError("The Planck ell column contains values that are not integers.")

    dataframe["ell"] = np.round(ell_values).astype(int)
    for column in PLANCK_COLUMNS[1:]:
        dataframe[column] = dataframe[column].astype(float)

    return dataframe


def extract_low_ell_planck(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract exactly ell = 2 through 29 from the Planck TT spectrum."""

    low_ell = dataframe[dataframe["ell"].isin(LOW_ELL_RANGE)].copy()
    low_ell = low_ell.sort_values("ell").reset_index(drop=True)

    if not np.array_equal(low_ell["ell"].to_numpy(), LOW_ELL_RANGE):
        raise ValueError("The Planck low-ell data do not contain exactly ell = 2 through 29.")

    return low_ell


def save_processed_planck_lowell(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned low-ell Planck rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def load_camb_lowell(camb_path: Path) -> pd.DataFrame:
    """Load CAMB theoretical D_ell values over ell = 2 through 29."""

    require_file(camb_path, "Stage 1 CAMB spectrum CSV")
    camb = pd.read_csv(camb_path)
    required = {"ell", "Dl_TT_uK2"}
    missing = required.difference(camb.columns)
    if missing:
        raise ValueError(f"CAMB CSV is missing required columns: {sorted(missing)}")

    camb_low = camb.loc[camb["ell"].isin(LOW_ELL_RANGE), ["ell", "Dl_TT_uK2"]].copy()
    camb_low["ell"] = camb_low["ell"].astype(int)
    camb_low = camb_low.rename(columns={"Dl_TT_uK2": "camb_Dl_uK2"})
    camb_low = camb_low.sort_values("ell").reset_index(drop=True)

    if not np.array_equal(camb_low["ell"].to_numpy(), LOW_ELL_RANGE):
        raise ValueError("The CAMB low-ell data do not contain exactly ell = 2 through 29.")

    return camb_low


def build_comparison_table(planck_low: pd.DataFrame, camb_low: pd.DataFrame) -> pd.DataFrame:
    """Merge Planck and CAMB data and calculate descriptive comparison columns."""

    comparison = pd.merge(planck_low, camb_low, on="ell", how="inner", validate="one_to_one")
    comparison = comparison.sort_values("ell").reset_index(drop=True)

    if not np.array_equal(comparison["ell"].to_numpy(), LOW_ELL_RANGE):
        raise ValueError("The merged table does not contain exactly ell = 2 through 29.")

    comparison["residual_Dl_uK2"] = comparison["planck_Dl_uK2"] - comparison["camb_Dl_uK2"]
    comparison["fractional_difference"] = (
        comparison["residual_Dl_uK2"] / comparison["camb_Dl_uK2"]
    )
    comparison["percent_difference"] = 100.0 * comparison["fractional_difference"]
    comparison["cosmic_variance_fraction"] = np.sqrt(2.0 / (2.0 * comparison["ell"] + 1.0))
    comparison["cosmic_variance_sigma_uK2"] = (
        comparison["camb_Dl_uK2"] * comparison["cosmic_variance_fraction"]
    )
    comparison["normalized_residual_cv"] = (
        comparison["residual_Dl_uK2"] / comparison["cosmic_variance_sigma_uK2"]
    )
    comparison["camb_cv_lower_uK2"] = (
        comparison["camb_Dl_uK2"] - comparison["cosmic_variance_sigma_uK2"]
    )
    comparison["camb_cv_upper_uK2"] = (
        comparison["camb_Dl_uK2"] + comparison["cosmic_variance_sigma_uK2"]
    )

    column_order = [
        "ell",
        "planck_Dl_uK2",
        "planck_error_lower_uK2",
        "planck_error_upper_uK2",
        "camb_Dl_uK2",
        "residual_Dl_uK2",
        "fractional_difference",
        "percent_difference",
        "cosmic_variance_fraction",
        "cosmic_variance_sigma_uK2",
        "normalized_residual_cv",
        "camb_cv_lower_uK2",
        "camb_cv_upper_uK2",
    ]
    return comparison[column_order]


def save_comparison_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the merged comparison table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def configure_axis_for_low_ell(ax: plt.Axes) -> None:
    """Apply shared low-ell x-axis formatting."""

    ax.set_xlim(1.5, 29.5)
    ax.set_xticks(np.arange(2, 30, 3))
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(r"Multipole $\ell$")


def save_figure(fig: plt.Figure, paths: Iterable[Path]) -> None:
    """Save a figure to all requested paths."""

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
    plt.close(fig)


def create_planck_camb_comparison_plot(dataframe: pd.DataFrame) -> None:
    """Plot Planck observations, CAMB theory, and ideal cosmic-variance band."""

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.fill_between(
        dataframe["ell"],
        dataframe["camb_cv_lower_uK2"],
        dataframe["camb_cv_upper_uK2"],
        color="#4c78a8",
        alpha=0.18,
        label="CAMB ideal full-sky cosmic variance (+/-1 sigma)",
    )
    ax.plot(
        dataframe["ell"],
        dataframe["camb_Dl_uK2"],
        color="#1f77b4",
        linewidth=1.8,
        label="CAMB lensed scalar TT theory",
    )
    yerr = np.vstack(
        [
            dataframe["planck_error_lower_uK2"].to_numpy(),
            dataframe["planck_error_upper_uK2"].to_numpy(),
        ]
    )
    ax.errorbar(
        dataframe["ell"],
        dataframe["planck_Dl_uK2"],
        yerr=yerr,
        fmt="o",
        color="#d62728",
        ecolor="#d62728",
        elinewidth=1.1,
        capsize=3.0,
        markersize=4.5,
        label="Planck 2018 PR3 observations with asymmetric 68% confidence intervals",
    )
    configure_axis_for_low_ell(ax)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
    ax.set_title("Planck 2018 PR3 TT Observations vs CAMB Theory, Low Multipoles")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, [FIGURE_PATHS["comparison_png"], FIGURE_PATHS["comparison_pdf"]])


def create_residual_plot(dataframe: pd.DataFrame) -> None:
    """Plot Planck minus CAMB residuals with ideal cosmic-variance limits."""

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.fill_between(
        dataframe["ell"],
        -dataframe["cosmic_variance_sigma_uK2"],
        dataframe["cosmic_variance_sigma_uK2"],
        color="#4c78a8",
        alpha=0.18,
        label="CAMB ideal full-sky cosmic variance around zero (+/-1 sigma)",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        dataframe["ell"],
        dataframe["residual_Dl_uK2"],
        color="#9467bd",
        marker="o",
        linewidth=1.4,
        markersize=4.5,
        label="Planck - CAMB residual",
    )
    configure_axis_for_low_ell(ax)
    ax.set_ylabel(r"Residual $D_\ell^{TT}$ [$\mu K^2$]")
    ax.set_title("Low-Multipole TT Residuals: Planck 2018 PR3 minus CAMB")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, [FIGURE_PATHS["residuals_png"], FIGURE_PATHS["residuals_pdf"]])


def create_fractional_difference_plot(dataframe: pd.DataFrame) -> None:
    """Plot percent differences between Planck and CAMB."""

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.plot(
        dataframe["ell"],
        dataframe["percent_difference"],
        color="#2ca02c",
        marker="o",
        linewidth=1.4,
        markersize=4.5,
    )
    configure_axis_for_low_ell(ax)
    ax.set_ylabel("Percent difference [%]")
    ax.set_title("Low-Multipole TT Percent Difference: (Planck - CAMB) / CAMB")
    fig.tight_layout()
    save_figure(fig, [FIGURE_PATHS["fractional_png"], FIGURE_PATHS["fractional_pdf"]])


def create_normalized_residual_plot(dataframe: pd.DataFrame) -> None:
    """Plot residuals normalized by ideal full-sky cosmic variance."""

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for value, linewidth in [(0, 1.1), (1, 0.8), (-1, 0.8), (2, 0.8), (-2, 0.8)]:
        ax.axhline(value, color="black", linestyle="-" if value == 0 else "--", linewidth=linewidth)
    ax.plot(
        dataframe["ell"],
        dataframe["normalized_residual_cv"],
        color="#ff7f0e",
        marker="o",
        linewidth=1.4,
        markersize=4.5,
    )
    configure_axis_for_low_ell(ax)
    ax.set_ylabel("Normalized residual relative to ideal cosmic variance")
    ax.set_title("Low-Multipole TT Residuals Normalized by Ideal Cosmic Variance")
    fig.tight_layout()
    save_figure(fig, [FIGURE_PATHS["normalized_png"], FIGURE_PATHS["normalized_pdf"]])


def create_all_figures(dataframe: pd.DataFrame) -> None:
    """Create all Stage 2 figures."""

    create_planck_camb_comparison_plot(dataframe)
    create_residual_plot(dataframe)
    create_fractional_difference_plot(dataframe)
    create_normalized_residual_plot(dataframe)


def print_summary(planck_rows_loaded: int, comparison: pd.DataFrame) -> None:
    """Print descriptive low-ell comparison summaries."""

    residual = comparison["residual_Dl_uK2"]
    fractional = comparison["fractional_difference"]
    below_count = int((comparison["residual_Dl_uK2"] < 0).sum())
    above_count = int((comparison["residual_Dl_uK2"] > 0).sum())
    largest_idx = comparison["normalized_residual_cv"].abs().idxmax()
    largest_row = comparison.loc[largest_idx]

    print("\nStage 2 low-ell Planck-CAMB analysis complete")
    print(f"Number of Planck rows loaded: {planck_rows_loaded}")
    print(
        "Multipole range analyzed: "
        f"{int(comparison['ell'].min())} to {int(comparison['ell'].max())}"
    )
    print(f"Number of rows matched with CAMB: {len(comparison)}")
    print(f"Mean residual: {residual.mean():.6g} microK^2")
    print(f"Median residual: {residual.median():.6g} microK^2")
    print(f"Mean fractional difference: {fractional.mean():.6g}")
    print(f"Number of observed multipoles below CAMB prediction: {below_count}")
    print(f"Number of observed multipoles above CAMB prediction: {above_count}")
    print(
        "Largest absolute normalized residual: "
        f"ell={int(largest_row['ell'])}, "
        f"value={largest_row['normalized_residual_cv']:.6g}"
    )
    print("Output files:")
    print(f"  - {PROCESSED_PLANCK_LOWELL}")
    print(f"  - {COMPARISON_CSV}")
    for figure_path in FIGURE_PATHS.values():
        print(f"  - {figure_path}")
    print(
        "Note: normalized_residual_cv is a convenient residual measured relative "
        "to ideal full-sky cosmic variance, not an official Planck significance."
    )


def main() -> None:
    """Run the Stage 2 low-ell comparison workflow."""

    planck_full = parse_planck_tt_file(RAW_PLANCK_FILE)
    planck_low = extract_low_ell_planck(planck_full)
    save_processed_planck_lowell(planck_low, PROCESSED_PLANCK_LOWELL)

    camb_low = load_camb_lowell(CAMB_SPECTRUM)
    comparison = build_comparison_table(planck_low, camb_low)
    save_comparison_table(comparison, COMPARISON_CSV)
    create_all_figures(comparison)
    print_summary(len(planck_full), comparison)


if __name__ == "__main__":
    main()
