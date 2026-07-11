"""Analyze precomputed CAMB spectra for primordial cutoff models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import camb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_low_ell import (
    LOW_ELL_RANGE,
    PROCESSED_PLANCK_LOWELL,
    RAW_PLANCK_FILE,
    extract_low_ell_planck,
    parse_planck_tt_file,
    save_processed_planck_lowell,
)
from cutoff_models import (
    AS,
    NS,
    PIVOT_SCALAR,
    CutoffFamily,
    CutoffModel,
    cutoff_function,
    model_cutoff_values,
    modified_primordial_power,
    standard_primordial_power,
)
from generate_cutoff_spectra import (
    ALPHA_VALUES,
    BASELINE_REPRODUCTION_CSV,
    KC_VALUES,
    RUN_METADATA_JSON,
    SHARP_CONVERGENCE_CSV,
    STANDARD_CAMB_CSV,
    INDEX_CSV,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TEXT_DIR = PROJECT_ROOT / "outputs" / "text"

SCAN_CSV = TABLE_DIR / "cutoff_model_scan.csv"
BEST_MODELS_CSV = TABLE_DIR / "cutoff_best_models.csv"
BEST_LOWELL_SPECTRA_CSV = TABLE_DIR / "cutoff_best_lowell_spectra.csv"
PRIMORDIAL_SAMPLES_CSV = TABLE_DIR / "cutoff_primordial_samples.csv"
METHODS_MD = TEXT_DIR / "cutoff_methods_and_results.md"

FIGURE_STEMS = [
    "cutoff_function_shapes",
    "best_primordial_power_spectra",
    "best_primordial_power_spectra_ratio",
    "planck_standard_cutoff_lowell_comparison",
    "cutoff_model_residual_comparison",
    "exponential_scan_heatmap",
    "rational_scan_heatmap",
    "cutoff_to_standard_TT_ratio",
    "cutoff_to_standard_TT_ratio_lowell",
]

LOW_ELL_MIN = 2
LOW_ELL_MAX = 29
ELL_MAX = 2500


def require_file(path: Path, description: str) -> None:
    """Raise a clear error when a required input is absent."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def ensure_planck_lowell() -> pd.DataFrame:
    """Load or regenerate the cleaned Planck low-ell table from the raw file."""

    if PROCESSED_PLANCK_LOWELL.exists():
        planck = pd.read_csv(PROCESSED_PLANCK_LOWELL)
    else:
        planck_full = parse_planck_tt_file(RAW_PLANCK_FILE)
        planck = extract_low_ell_planck(planck_full)
        save_processed_planck_lowell(planck, PROCESSED_PLANCK_LOWELL)
    expected = np.arange(LOW_ELL_MIN, LOW_ELL_MAX + 1)
    planck = planck.sort_values("ell").reset_index(drop=True)
    if not np.array_equal(planck["ell"].to_numpy(dtype=int), expected):
        raise ValueError("Planck low-ell table must contain exactly ell = 2 through 29.")
    return planck


def load_generation_index() -> pd.DataFrame:
    """Load the cutoff spectrum index written by the generator."""

    require_file(INDEX_CSV, "cutoff spectrum index")
    index = pd.read_csv(INDEX_CSV)
    required = {"model_id", "model_family", "spectrum_csv", "runtime_seconds"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"Cutoff spectrum index is missing columns: {sorted(missing)}")
    return index


def load_standard_spectrum() -> pd.DataFrame:
    """Load the original fixed Lambda-CDM CAMB spectrum."""

    require_file(STANDARD_CAMB_CSV, "standard CAMB spectrum")
    standard = pd.read_csv(STANDARD_CAMB_CSV)
    required = {"ell", "Cl_TT_uK2", "Dl_TT_uK2"}
    missing = required.difference(standard.columns)
    if missing:
        raise ValueError(f"Standard CAMB table is missing columns: {sorted(missing)}")
    return standard.sort_values("ell").reset_index(drop=True)


def read_spectrum(path: str | Path) -> pd.DataFrame:
    """Load one generated spectrum and validate its ell support."""

    spectrum_path = Path(path)
    require_file(spectrum_path, "generated cutoff spectrum")
    dataframe = pd.read_csv(spectrum_path)
    required = {"ell", "Cl_TT_uK2", "Dl_TT_uK2"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Spectrum {spectrum_path} is missing columns: {sorted(missing)}")
    dataframe = dataframe.sort_values("ell").reset_index(drop=True)
    expected = np.arange(2, ELL_MAX + 1)
    if not np.array_equal(dataframe["ell"].to_numpy(dtype=int), expected):
        raise ValueError(f"Spectrum {spectrum_path} must contain exactly ell = 2 through 2500.")
    values = dataframe[["Cl_TT_uK2", "Dl_TT_uK2"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (dataframe["Dl_TT_uK2"].to_numpy(dtype=float) > 0).all():
        raise ValueError(f"Spectrum {spectrum_path} contains invalid TT values.")
    return dataframe


def compute_band_variance(ell: np.ndarray, dl_values: np.ndarray) -> float:
    """Compute V = 1/2 sum[(2ell+1) D_ell / (ell(ell+1))]."""

    ell_float = ell.astype(float)
    weights = (2.0 * ell_float + 1.0) / (ell_float * (ell_float + 1.0))
    return float(0.5 * np.sum(weights * dl_values))


def compute_metrics(
    metadata: pd.Series,
    spectrum: pd.DataFrame,
    standard: pd.DataFrame,
    planck: pd.DataFrame,
    observed_band_variance: float,
    q_baseline: float,
    aic_baseline: float,
) -> dict[str, object]:
    """Calculate all requested comparison metrics for one model."""

    low = pd.merge(
        planck,
        spectrum[["ell", "Dl_TT_uK2"]].rename(columns={"Dl_TT_uK2": "model_Dl_uK2"}),
        on="ell",
        validate="one_to_one",
    )
    ell = low["ell"].to_numpy(dtype=int)
    planck_dl = low["planck_Dl_uK2"].to_numpy(dtype=float)
    model_dl = low["model_Dl_uK2"].to_numpy(dtype=float)
    residual = planck_dl - model_dl
    frac_residual = residual / model_dl
    selected_sigma = np.where(
        model_dl > planck_dl,
        low["planck_error_upper_uK2"].to_numpy(dtype=float),
        low["planck_error_lower_uK2"].to_numpy(dtype=float),
    )
    model_band_variance = compute_band_variance(ell, model_dl)
    r_low = observed_band_variance / model_band_variance
    q_value = float(np.sum((2.0 * ell + 1.0) * (planck_dl / model_dl + np.log(model_dl))))
    n_extra = int(metadata.get("n_extra_parameters", 0))
    aic = q_value + 2.0 * n_extra

    merged_high = pd.merge(
        standard[["ell", "Dl_TT_uK2"]].rename(columns={"Dl_TT_uK2": "standard_Dl_uK2"}),
        spectrum[["ell", "Dl_TT_uK2"]].rename(columns={"Dl_TT_uK2": "model_Dl_uK2"}),
        on="ell",
        validate="one_to_one",
    )
    high_30_100 = merged_high[(merged_high["ell"] >= 30) & (merged_high["ell"] <= 100)]
    high_30_2500 = merged_high[(merged_high["ell"] >= 30) & (merged_high["ell"] <= ELL_MAX)]
    frac_30_100 = (
        high_30_100["model_Dl_uK2"].to_numpy(dtype=float)
        / high_30_100["standard_Dl_uK2"].to_numpy(dtype=float)
        - 1.0
    )
    frac_30_2500 = (
        high_30_2500["model_Dl_uK2"].to_numpy(dtype=float)
        / high_30_2500["standard_Dl_uK2"].to_numpy(dtype=float)
        - 1.0
    )

    return {
        "model_family": str(metadata["model_family"]),
        "model_id": str(metadata["model_id"]),
        "kc_Mpc_inverse": metadata.get("kc_Mpc_inverse", np.nan),
        "alpha": metadata.get("alpha", np.nan),
        "pivot_normalized": bool(metadata.get("pivot_normalized", True)),
        "sharp_implementation": str(metadata.get("sharp_implementation", "not_applicable")),
        "sharp_approximation_alpha": metadata.get("sharp_approximation_alpha", np.nan),
        "spectrum_csv": str(metadata["spectrum_csv"]),
        "n_extra_parameters": n_extra,
        "observed_band_variance_uK2": observed_band_variance,
        "model_band_variance_uK2": model_band_variance,
        "R_low_observed_over_model": r_low,
        "percent_band_difference": 100.0 * (r_low - 1.0),
        "mean_fractional_residual": float(np.mean(frac_residual)),
        "mean_absolute_fractional_residual": float(np.mean(np.abs(frac_residual))),
        "rms_fractional_residual": float(np.sqrt(np.mean(np.square(frac_residual)))),
        "count_planck_below_model": int(np.sum(planck_dl < model_dl)),
        "count_planck_above_model": int(np.sum(planck_dl > model_dl)),
        "chi2_asymmetric_descriptive": float(np.sum(np.square(residual / selected_sigma))),
        "Q_ideal_fullsky": q_value,
        "Delta_Q_vs_baseline": q_value - q_baseline,
        "AIC_ideal": aic,
        "Delta_AIC_vs_baseline": aic - aic_baseline,
        "max_abs_fractional_change_ell_30_100": float(np.max(np.abs(frac_30_100))),
        "max_abs_fractional_change_ell_30_2500": float(np.max(np.abs(frac_30_2500))),
        "mean_abs_fractional_change_ell_30_2500": float(np.mean(np.abs(frac_30_2500))),
        "grid_boundary_warning": "",
        "camb_version": str(metadata.get("camb_version", camb.__version__)),
        "runtime_seconds": float(metadata.get("runtime_seconds", 0.0)),
    }


def add_boundary_warnings(scan: pd.DataFrame) -> pd.DataFrame:
    """Attach warnings to rows whose parameters lie on the tested grid edge."""

    scan = scan.copy()
    warnings: list[str] = []
    kc_min = float(np.min(KC_VALUES))
    kc_max = float(np.max(KC_VALUES))
    alpha_min = float(np.min(ALPHA_VALUES))
    alpha_max = float(np.max(ALPHA_VALUES))
    for _, row in scan.iterrows():
        parts: list[str] = []
        family = row["model_family"]
        kc = row["kc_Mpc_inverse"]
        alpha = row["alpha"]
        if family != CutoffFamily.STANDARD.value and pd.notna(kc):
            if np.isclose(float(kc), kc_min, rtol=0.0, atol=1e-15):
                parts.append("kc at tested minimum")
            if np.isclose(float(kc), kc_max, rtol=0.0, atol=1e-15):
                parts.append("kc at tested maximum")
        if family in {CutoffFamily.EXPONENTIAL.value, CutoffFamily.RATIONAL.value} and pd.notna(alpha):
            if np.isclose(float(alpha), alpha_min, rtol=0.0, atol=1e-15):
                parts.append("alpha at tested minimum")
            if np.isclose(float(alpha), alpha_max, rtol=0.0, atol=1e-15):
                parts.append("alpha at tested maximum")
        warnings.append("; ".join(parts))
    scan["grid_boundary_warning"] = warnings
    return scan


def select_best_models(scan: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Select the requested best-model rows."""

    baseline = scan.loc[scan["model_family"] == CutoffFamily.STANDARD.value].iloc[0]
    best_exp = scan.loc[scan["model_family"] == CutoffFamily.EXPONENTIAL.value].sort_values(
        "Q_ideal_fullsky"
    ).iloc[0]
    best_rat = scan.loc[scan["model_family"] == CutoffFamily.RATIONAL.value].sort_values(
        "Q_ideal_fullsky"
    ).iloc[0]
    best_sharp = scan.loc[scan["model_family"] == CutoffFamily.SHARP.value].sort_values(
        "Q_ideal_fullsky"
    ).iloc[0]
    cutoff_only = scan.loc[scan["model_family"] != CutoffFamily.STANDARD.value]
    best_cutoff_q = cutoff_only.sort_values("Q_ideal_fullsky").iloc[0]
    best_aic = scan.sort_values("AIC_ideal").iloc[0]

    selected = [
        ("standard baseline", baseline),
        ("best exponential model by minimum Q", best_exp),
        ("best rational model by minimum Q", best_rat),
        ("best sharp model by minimum Q", best_sharp),
        ("overall best cutoff model by minimum Q", best_cutoff_q),
        ("overall best model by minimum AIC", best_aic),
    ]
    rows = []
    for label, row in selected:
        output = row.to_dict()
        output["selection"] = label
        rows.append(output)
    best = pd.DataFrame(rows)
    first_columns = ["selection"] + [column for column in scan.columns if column in best.columns]
    key_rows = {
        "standard": baseline,
        "exponential": best_exp,
        "rational": best_rat,
        "sharp": best_sharp,
        "overall_q": best_cutoff_q,
        "overall_aic": best_aic,
    }
    return best[first_columns], key_rows


def spectrum_for_row(row: pd.Series) -> pd.DataFrame:
    """Load the spectrum associated with a scan row."""

    return read_spectrum(row["spectrum_csv"])


def build_best_lowell_spectra(planck: pd.DataFrame, best_rows: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a compact low-ell comparison table for the selected families."""

    output = planck.rename(
        columns={
            "planck_Dl_uK2": "Planck_Dl_uK2",
            "planck_error_lower_uK2": "Planck_error_lower_uK2",
            "planck_error_upper_uK2": "Planck_error_upper_uK2",
        }
    ).copy()
    names = {
        "standard": "standard_CAMB_Dl_uK2",
        "exponential": "best_exponential_Dl_uK2",
        "rational": "best_rational_Dl_uK2",
        "sharp": "best_sharp_Dl_uK2",
    }
    for key, column in names.items():
        spectrum = spectrum_for_row(best_rows[key])
        low = spectrum.loc[spectrum["ell"].isin(LOW_ELL_RANGE), ["ell", "Dl_TT_uK2"]].rename(
            columns={"Dl_TT_uK2": column}
        )
        output = pd.merge(output, low, on="ell", validate="one_to_one")
    return output


def row_to_model(row: pd.Series) -> CutoffModel:
    """Convert a scan row back into a CutoffModel."""

    family = CutoffFamily(str(row["model_family"]))
    kc = None if pd.isna(row["kc_Mpc_inverse"]) else float(row["kc_Mpc_inverse"])
    alpha = None if pd.isna(row["alpha"]) else float(row["alpha"])
    return CutoffModel(
        family=family,
        kc_mpc_inverse=kc,
        alpha=alpha,
        pivot_normalized=bool(row["pivot_normalized"]),
        sharp_implementation=str(row.get("sharp_implementation", "not_applicable")),
    )


def build_primordial_samples(best_rows: dict[str, pd.Series]) -> pd.DataFrame:
    """Save sampled primordial spectra for the standard and selected cutoffs."""

    k = np.logspace(-6, -1, 500)
    standard = standard_primordial_power(k)
    output = pd.DataFrame({"k_Mpc_inverse": k, "P0_k": standard})
    for key in ["exponential", "rational", "sharp"]:
        model = row_to_model(best_rows[key])
        output[f"{key}_F_k_over_kc"] = model_cutoff_values(k, model)
        output[f"{key}_Pnew_k"] = modified_primordial_power(k, model)
        output[f"{key}_Pnew_over_P0"] = output[f"{key}_Pnew_k"] / standard
    return output


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as PNG and PDF without touching older paper figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_cutoff_function_shapes() -> None:
    """Plot the cutoff functions before pivot normalization."""

    x = np.logspace(-4, 4, 800)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = {1.0: "#1f77b4", 2.0: "#2ca02c", 4.0: "#ff7f0e", 8.0: "#9467bd"}
    for alpha in ALPHA_VALUES:
        ax.plot(x, cutoff_function(CutoffFamily.EXPONENTIAL, x, float(alpha)), color=colors[float(alpha)], linestyle="-", label=fr"exp $\alpha={alpha:g}$")
        ax.plot(x, cutoff_function(CutoffFamily.RATIONAL, x, float(alpha)), color=colors[float(alpha)], linestyle="--", label=fr"rat $\alpha={alpha:g}$")
    ax.plot(x, cutoff_function(CutoffFamily.SHARP, x), color="black", linewidth=1.8, label="sharp step")
    ax.set_xscale("log")
    ax.set_xlabel(r"$x=k/k_c$")
    ax.set_ylabel(r"$F(x)$")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(ncol=2, fontsize=8.2)
    ax.set_title("Primordial Cutoff Function Shapes")
    fig.tight_layout()
    save_figure(fig, "cutoff_function_shapes")


def plot_best_primordial_power(samples: pd.DataFrame) -> None:
    """Plot best primordial spectra and ratios."""

    k = samples["k_Mpc_inverse"]
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.plot(k, samples["P0_k"], color="black", linewidth=1.8, label=r"standard $P_0(k)$")
    for key, label, color in [
        ("exponential", "best exponential", "#1f77b4"),
        ("rational", "best rational", "#2ca02c"),
        ("sharp", "best sharp", "#d62728"),
    ]:
        ax.plot(k, samples[f"{key}_Pnew_k"], color=color, linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_\mathcal{R}(k)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.set_title("Best Tested Primordial Power Spectra")
    fig.tight_layout()
    save_figure(fig, "best_primordial_power_spectra")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for key, label, color in [
        ("exponential", "best exponential", "#1f77b4"),
        ("rational", "best rational", "#2ca02c"),
        ("sharp", "best sharp", "#d62728"),
    ]:
        ax.plot(k, samples[f"{key}_Pnew_over_P0"], color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$P_\mathrm{new}(k)/P_0(k)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.set_title("Best Tested Primordial Power Ratios")
    fig.tight_layout()
    save_figure(fig, "best_primordial_power_spectra_ratio")


def plot_lowell_comparison(lowell: pd.DataFrame) -> None:
    """Plot Planck, standard CAMB, and selected cutoff models over low ell."""

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    yerr = np.vstack(
        [
            lowell["Planck_error_lower_uK2"].to_numpy(dtype=float),
            lowell["Planck_error_upper_uK2"].to_numpy(dtype=float),
        ]
    )
    ax.errorbar(
        lowell["ell"],
        lowell["Planck_Dl_uK2"],
        yerr=yerr,
        fmt="o",
        color="black",
        ecolor="0.25",
        capsize=3,
        markersize=4.5,
        label="Planck 2018 TT",
    )
    series = [
        ("standard_CAMB_Dl_uK2", r"standard $\Lambda$CDM", "#4c78a8"),
        ("best_exponential_Dl_uK2", "best exponential", "#1f77b4"),
        ("best_rational_Dl_uK2", "best rational", "#2ca02c"),
        ("best_sharp_Dl_uK2", "best sharp", "#d62728"),
    ]
    for column, label, color in series:
        ax.plot(lowell["ell"], lowell[column], marker="o", linewidth=1.5, markersize=3.5, label=label, color=color)
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
    ax.set_title(r"Planck and Best Tested Cutoff Models, low-$\ell$ TT")
    ax.set_xlim(1.5, 29.5)
    ax.set_xticks(np.arange(2, 30, 3))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "planck_standard_cutoff_lowell_comparison")


def plot_residual_comparison(lowell: pd.DataFrame) -> None:
    """Plot Planck-minus-model residuals for selected models."""

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0)
    series = [
        ("standard_CAMB_Dl_uK2", r"standard $\Lambda$CDM", "#4c78a8"),
        ("best_exponential_Dl_uK2", "best exponential", "#1f77b4"),
        ("best_rational_Dl_uK2", "best rational", "#2ca02c"),
        ("best_sharp_Dl_uK2", "best sharp", "#d62728"),
    ]
    for column, label, color in series:
        residual = lowell["Planck_Dl_uK2"] - lowell[column]
        ax.plot(lowell["ell"], residual, marker="o", linewidth=1.4, markersize=3.5, label=label, color=color)
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"Planck - model $D_\ell^{TT}$ [$\mu K^2$]")
    ax.set_title("Low-ell Residual Comparison")
    ax.set_xlim(1.5, 29.5)
    ax.set_xticks(np.arange(2, 30, 3))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "cutoff_model_residual_comparison")


def plot_heatmap(scan: pd.DataFrame, family: CutoffFamily, stem: str) -> None:
    """Plot Delta_Q over kc and alpha for one smooth family."""

    subset = scan.loc[scan["model_family"] == family.value].copy()
    pivot = subset.pivot(index="alpha", columns="kc_Mpc_inverse", values="Delta_Q_vs_baseline")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    xticks = np.arange(len(pivot.columns))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{value:.1e}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel(r"$k_c$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(fr"{family.value.title()} Scan: $\Delta Q$ vs Standard")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\Delta Q$")
    fig.tight_layout()
    save_figure(fig, stem)


def plot_tt_ratios(best_rows: dict[str, pd.Series]) -> None:
    """Plot D_ell cutoff / standard for selected models."""

    standard = spectrum_for_row(best_rows["standard"])
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig_low, ax_low = plt.subplots(figsize=(8.5, 4.8))
    for key, label, color in [
        ("exponential", "best exponential", "#1f77b4"),
        ("rational", "best rational", "#2ca02c"),
        ("sharp", "best sharp", "#d62728"),
    ]:
        spectrum = spectrum_for_row(best_rows[key])
        merged = pd.merge(
            standard[["ell", "Dl_TT_uK2"]].rename(columns={"Dl_TT_uK2": "standard"}),
            spectrum[["ell", "Dl_TT_uK2"]].rename(columns={"Dl_TT_uK2": "model"}),
            on="ell",
            validate="one_to_one",
        )
        ratio = merged["model"] / merged["standard"]
        ax.plot(merged["ell"], ratio, linewidth=1.2, label=label, color=color)
        low = merged[merged["ell"] <= 100]
        ax_low.plot(low["ell"], low["model"] / low["standard"], linewidth=1.4, label=label, color=color)
    for axis in [ax, ax_low]:
        axis.axhline(1.0, color="black", linewidth=1.0)
        axis.set_xlabel(r"Multipole $\ell$")
        axis.set_ylabel(r"$D_\ell^\mathrm{cutoff}/D_\ell^\mathrm{standard}$")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8.5)
    ax.set_xlim(2, ELL_MAX)
    ax.set_title("Cutoff-to-Standard TT Ratio")
    ax_low.set_xlim(2, 100)
    ax_low.set_title("Cutoff-to-Standard TT Ratio, Low-ell Zoom")
    fig.tight_layout()
    fig_low.tight_layout()
    save_figure(fig, "cutoff_to_standard_TT_ratio")
    save_figure(fig_low, "cutoff_to_standard_TT_ratio_lowell")


def create_all_figures(scan: pd.DataFrame, lowell: pd.DataFrame, samples: pd.DataFrame, best_rows: dict[str, pd.Series]) -> None:
    """Create all requested figures."""

    plot_cutoff_function_shapes()
    plot_best_primordial_power(samples)
    plot_lowell_comparison(lowell)
    plot_residual_comparison(lowell)
    plot_heatmap(scan, CutoffFamily.EXPONENTIAL, "exponential_scan_heatmap")
    plot_heatmap(scan, CutoffFamily.RATIONAL, "rational_scan_heatmap")
    plot_tt_ratios(best_rows)


def write_methods_summary(scan: pd.DataFrame, best: pd.DataFrame, run_metadata: dict[str, object]) -> None:
    """Write the requested human-readable methods and results summary."""

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    pivot_normalized = bool(run_metadata.get("pivot_normalized", True))
    mode = str(run_metadata.get("scan_mode", "unknown"))
    sharp_implementation = str(run_metadata.get("sharp_implementation", "unknown"))
    sharp_alpha = run_metadata.get("sharp_approximation_alpha", None)
    boundary = best.loc[best["grid_boundary_warning"].fillna("") != "", ["selection", "grid_boundary_warning"]]
    boundary_text = "None among the selected rows."
    if not boundary.empty:
        boundary_text = "\n".join(
            f"- {row.selection}: {row.grid_boundary_warning}" for row in boundary.itertuples(index=False)
        )
    selected_columns = [
        "selection",
        "model_family",
        "kc_Mpc_inverse",
        "alpha",
        "R_low_observed_over_model",
        "Delta_Q_vs_baseline",
        "Delta_AIC_vs_baseline",
        "max_abs_fractional_change_ell_30_2500",
        "grid_boundary_warning",
    ]
    table_rows = ["| " + " | ".join(selected_columns) + " |"]
    table_rows.append("| " + " | ".join(["---"] * len(selected_columns)) + " |")
    for _, row in best[selected_columns].iterrows():
        values = []
        for column in selected_columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        table_rows.append("| " + " | ".join(values) + " |")
    best_table_markdown = "\n".join(table_rows)
    lines = [
        "# Cutoff Methods and Results",
        "",
        "## Formulas",
        "",
        "The standard primordial spectrum is `P0(k) = As * (k / k_pivot) ** (ns - 1)`.",
        "The tested cutoff spectrum is `Pnew(k) = P0(k) * F(k/kc)`.",
        "For smooth cutoffs, the default run preserves `As` at the pivot by dividing by `F(k_pivot/kc)`.",
        "",
        "Cutoff functions:",
        "",
        "- Exponential: `F(x) = 1 - exp(-x**alpha)`",
        "- Rational: `F(x) = x**alpha / (1 + x**alpha)`",
        "- Sharp: `F(x) = 0` for `x < 1`, and `F(x) = 1` for `x >= 1`",
        "",
        "## Grid and CAMB Settings",
        "",
        f"- Scan mode: {mode}",
        f"- Pivot normalization enabled: {pivot_normalized}",
        f"- kc range: {float(np.min(KC_VALUES)):.12g} to {float(np.max(KC_VALUES)):.12g} Mpc^-1",
        f"- Number of full-grid kc values: {len(KC_VALUES)}",
        f"- Smooth alpha values: {', '.join(f'{value:g}' for value in ALPHA_VALUES)}",
        "- Cosmology: fixed Planck-2018-like flat Lambda-CDM parameters from `generate_camb_spectrum.py`",
        f"- CAMB version: {camb.__version__}",
        f"- Sharp implementation selected by convergence test: {sharp_implementation}",
        f"- Sharp approximation exponent used when applicable: {sharp_alpha}",
        "- Note: CAMB/HMCode timed out for the originally requested alpha = 100 sharp approximation at part of the full grid; the recorded alpha = 50 fallback preserves the standard NonLinear_lens CAMB setting and is labeled as a numerical sharp-step approximation.",
        "",
        "## Best Tested Rows",
        "",
        best_table_markdown,
        "",
        "## Boundary Warnings",
        "",
        boundary_text,
        "",
        "## Limitations",
        "",
        "This is a conditional comparison using fixed cosmological parameters, public Planck low-ell spectrum values, a simplified ideal full-sky likelihood proxy, and a finite pre-specified parameter grid. It is not the official Planck likelihood or model evidence. A smaller Q means only that a model improves the simplified fit among the tested grid; a definitive comparison requires the official Planck likelihood.",
    ]
    METHODS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(scan: pd.DataFrame, best: pd.DataFrame, total_runtime: float) -> None:
    """Print a concise terminal summary."""

    print("\nCutoff model analysis complete")
    baseline = scan.loc[scan["model_family"] == CutoffFamily.STANDARD.value].iloc[0]
    print(f"standard R_low: {baseline['R_low_observed_over_model']:.8g}")
    for family in [CutoffFamily.EXPONENTIAL, CutoffFamily.RATIONAL, CutoffFamily.SHARP]:
        row = scan.loc[scan["model_family"] == family.value].sort_values("Q_ideal_fullsky").iloc[0]
        print(
            f"best {family.value}: kc={row['kc_Mpc_inverse']:.8g}, "
            f"alpha={row['alpha'] if pd.notna(row['alpha']) else 'none'}, "
            f"Delta_Q={row['Delta_Q_vs_baseline']:.8g}, "
            f"Delta_AIC={row['Delta_AIC_vs_baseline']:.8g}, "
            f"R_low={row['R_low_observed_over_model']:.8g}"
        )
        if row["grid_boundary_warning"]:
            print(f"WARNING {family.value}: {row['grid_boundary_warning']}; tested range may need expansion.")
    best_q = best.loc[best["selection"] == "overall best cutoff model by minimum Q"].iloc[0]
    best_aic = best.loc[best["selection"] == "overall best model by minimum AIC"].iloc[0]
    print(f"overall best cutoff by Q: {best_q['model_family']} ({best_q['model_id']})")
    print(f"overall best by AIC: {best_aic['model_family']} ({best_aic['model_id']})")
    print(
        "high-ell preservation for best-Q cutoff: "
        f"max30-100={best_q['max_abs_fractional_change_ell_30_100']:.3e}, "
        f"max30-2500={best_q['max_abs_fractional_change_ell_30_2500']:.3e}, "
        f"mean30-2500={best_q['mean_abs_fractional_change_ell_30_2500']:.3e}"
    )
    print(f"CAMB version: {camb.__version__}")
    print(f"total_runtime_seconds: {total_runtime:.3f}")
    print("Output files:")
    for path in [
        SCAN_CSV,
        BEST_MODELS_CSV,
        BEST_LOWELL_SPECTRA_CSV,
        PRIMORDIAL_SAMPLES_CSV,
        METHODS_MD,
    ] + [FIGURE_DIR / f"{stem}.png" for stem in FIGURE_STEMS] + [FIGURE_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS]:
        print(f"  - {path}")


def main() -> None:
    """Run the cutoff model analysis."""

    start = time.perf_counter()
    require_file(BASELINE_REPRODUCTION_CSV, "baseline reproduction report")
    require_file(SHARP_CONVERGENCE_CSV, "sharp convergence report")
    planck = ensure_planck_lowell()
    standard = load_standard_spectrum()
    index = load_generation_index()
    run_metadata = json.loads(RUN_METADATA_JSON.read_text(encoding="utf-8")) if RUN_METADATA_JSON.exists() else {}

    ell = planck["ell"].to_numpy(dtype=int)
    observed_band_variance = compute_band_variance(ell, planck["planck_Dl_uK2"].to_numpy(dtype=float))
    standard_low = standard.loc[standard["ell"].isin(LOW_ELL_RANGE), ["ell", "Dl_TT_uK2"]]
    low_for_q = pd.merge(planck, standard_low, on="ell", validate="one_to_one")
    planck_dl = low_for_q["planck_Dl_uK2"].to_numpy(dtype=float)
    standard_dl = low_for_q["Dl_TT_uK2"].to_numpy(dtype=float)
    q_baseline = float(np.sum((2.0 * ell + 1.0) * (planck_dl / standard_dl + np.log(standard_dl))))
    aic_baseline = q_baseline

    rows: list[dict[str, object]] = []
    for _, metadata in index.iterrows():
        spectrum = standard if metadata["model_family"] == CutoffFamily.STANDARD.value else read_spectrum(metadata["spectrum_csv"])
        rows.append(
            compute_metrics(
                metadata,
                spectrum,
                standard,
                planck,
                observed_band_variance,
                q_baseline,
                aic_baseline,
            )
        )
    scan = add_boundary_warnings(pd.DataFrame(rows))
    scan = scan.sort_values(["model_family", "kc_Mpc_inverse", "alpha"], na_position="first").reset_index(drop=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(SCAN_CSV, index=False)

    best, best_rows = select_best_models(scan)
    best.to_csv(BEST_MODELS_CSV, index=False)
    lowell = build_best_lowell_spectra(planck, best_rows)
    lowell.to_csv(BEST_LOWELL_SPECTRA_CSV, index=False)
    samples = build_primordial_samples(best_rows)
    samples.to_csv(PRIMORDIAL_SAMPLES_CSV, index=False)
    create_all_figures(scan, lowell, samples, best_rows)
    write_methods_summary(scan, best, run_metadata)

    print_summary(scan, best, time.perf_counter() - start)


if __name__ == "__main__":
    main()
