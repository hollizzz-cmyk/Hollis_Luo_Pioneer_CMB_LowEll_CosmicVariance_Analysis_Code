"""Validate the primordial cutoff-model scan, analysis tables, and figures."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cutoff_models
from analyze_cutoff_models import (
    BEST_LOWELL_SPECTRA_CSV,
    BEST_MODELS_CSV,
    FIGURE_STEMS,
    METHODS_MD,
    PRIMORDIAL_SAMPLES_CSV,
    SCAN_CSV,
    compute_band_variance,
)
from cutoff_models import (
    AS,
    PIVOT_SCALAR,
    CutoffFamily,
    CutoffModel,
    cutoff_function,
    modified_primordial_power,
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
ELL_EXPECTED = np.arange(2, 2501)
LOW_ELL_EXPECTED = np.arange(2, 30)

REQUIRED_SCAN_COLUMNS = [
    "model_family",
    "kc_Mpc_inverse",
    "alpha",
    "pivot_normalized",
    "n_extra_parameters",
    "observed_band_variance_uK2",
    "model_band_variance_uK2",
    "R_low_observed_over_model",
    "percent_band_difference",
    "mean_fractional_residual",
    "mean_absolute_fractional_residual",
    "rms_fractional_residual",
    "count_planck_below_model",
    "count_planck_above_model",
    "chi2_asymmetric_descriptive",
    "Q_ideal_fullsky",
    "Delta_Q_vs_baseline",
    "AIC_ideal",
    "Delta_AIC_vs_baseline",
    "max_abs_fractional_change_ell_30_100",
    "max_abs_fractional_change_ell_30_2500",
    "mean_abs_fractional_change_ell_30_2500",
    "grid_boundary_warning",
    "camb_version",
    "runtime_seconds",
]

REQUIRED_TABLES = [
    SCAN_CSV,
    BEST_MODELS_CSV,
    BEST_LOWELL_SPECTRA_CSV,
    PRIMORDIAL_SAMPLES_CSV,
    BASELINE_REPRODUCTION_CSV,
    SHARP_CONVERGENCE_CSV,
]


@dataclass
class ValidationResult:
    """Single validation check result."""

    name: str
    passed: bool
    detail: str
    critical: bool = True


def print_result(result: ValidationResult) -> None:
    """Print one PASS/FAIL line."""

    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.name} - {result.detail}")


def file_nonempty(path: Path) -> bool:
    """Return True when a path exists and is nonempty."""

    return path.exists() and path.stat().st_size > 0


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if possible, otherwise return an empty dataframe."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric_signature(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load numerical CSV outputs for deterministic rerun comparison."""

    signatures: dict[str, pd.DataFrame] = {}
    for path in paths:
        dataframe = pd.read_csv(path)
        numeric = dataframe.select_dtypes(include=[np.number]).copy()
        signatures[str(path)] = numeric
    return signatures


def signatures_equal(before: dict[str, pd.DataFrame], after: dict[str, pd.DataFrame]) -> bool:
    """Compare numerical CSV signatures exactly within tight floating tolerance."""

    if before.keys() != after.keys():
        return False
    for key in before:
        left = before[key]
        right = after[key]
        if list(left.columns) != list(right.columns) or left.shape != right.shape:
            return False
        if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0.0, atol=1e-12, equal_nan=True):
            return False
    return True


def validate_grid(scan: pd.DataFrame) -> list[ValidationResult]:
    """Validate that the full pre-specified grid exists exactly once."""

    results: list[ValidationResult] = []
    columns_ok = all(column in scan.columns for column in REQUIRED_SCAN_COLUMNS)
    results.append(
        ValidationResult(
            "cutoff scan contains required columns",
            columns_ok,
            f"missing={sorted(set(REQUIRED_SCAN_COLUMNS).difference(scan.columns))}",
        )
    )
    if not columns_ok:
        return results

    expected_count = 1 + 2 * len(KC_VALUES) * len(ALPHA_VALUES) + len(KC_VALUES)
    results.append(
        ValidationResult(
            "full scan has one row per expected model",
            len(scan) == expected_count,
            f"rows={len(scan)}, expected={expected_count}",
        )
    )
    standard_count = int((scan["model_family"] == CutoffFamily.STANDARD.value).sum())
    results.append(
        ValidationResult("standard baseline exists exactly once", standard_count == 1, f"count={standard_count}")
    )

    exact_once = True
    missing: list[str] = []
    for family in [CutoffFamily.EXPONENTIAL.value, CutoffFamily.RATIONAL.value]:
        subset = scan.loc[scan["model_family"] == family]
        for kc in KC_VALUES:
            for alpha in ALPHA_VALUES:
                mask = np.isclose(subset["kc_Mpc_inverse"].to_numpy(dtype=float), float(kc), rtol=0.0, atol=1e-15) & np.isclose(
                    subset["alpha"].to_numpy(dtype=float), float(alpha), rtol=0.0, atol=1e-15
                )
                if int(mask.sum()) != 1:
                    exact_once = False
                    missing.append(f"{family}:kc={kc:.6e},alpha={alpha:g},count={int(mask.sum())}")
    sharp = scan.loc[scan["model_family"] == CutoffFamily.SHARP.value]
    for kc in KC_VALUES:
        mask = np.isclose(sharp["kc_Mpc_inverse"].to_numpy(dtype=float), float(kc), rtol=0.0, atol=1e-15)
        if int(mask.sum()) != 1:
            exact_once = False
            missing.append(f"sharp:kc={kc:.6e},count={int(mask.sum())}")
    results.append(
        ValidationResult(
            "all expected kc and alpha combinations exist exactly once",
            exact_once,
            "all combinations found" if exact_once else "; ".join(missing[:6]),
        )
    )

    cutoff_rows = scan.loc[scan["model_family"] != CutoffFamily.STANDARD.value]
    kc = cutoff_rows["kc_Mpc_inverse"].to_numpy(dtype=float)
    kc_ok = (
        len(kc) > 0
        and np.isfinite(kc).all()
        and bool((kc >= np.min(KC_VALUES) - 1e-18).all())
        and bool((kc <= np.max(KC_VALUES) + 1e-18).all())
    )
    results.append(
        ValidationResult(
            "kc has Mpc^-1 column and remains in the stated range",
            kc_ok,
            f"min={np.nanmin(kc) if len(kc) else 'none'}, max={np.nanmax(kc) if len(kc) else 'none'}",
        )
    )
    return results


def validate_cutoff_formulas(scan: pd.DataFrame) -> list[ValidationResult]:
    """Validate cutoff functions and primordial spectra."""

    results: list[ValidationResult] = []
    x = np.logspace(-12, 12, 2000)
    bounded = True
    small_ok = True
    large_ok = True
    for family in [CutoffFamily.EXPONENTIAL, CutoffFamily.RATIONAL]:
        for alpha in ALPHA_VALUES:
            values = cutoff_function(family, x, float(alpha))
            bounded = bounded and np.isfinite(values).all() and bool((values >= 0.0).all()) and bool((values <= 1.0).all())
            small_ok = small_ok and float(values[0]) < 1e-6
            large_ok = large_ok and float(values[-1]) > 1.0 - 1e-6
    step = cutoff_function(CutoffFamily.SHARP, x)
    bounded = bounded and np.isfinite(step).all() and bool((step >= 0.0).all()) and bool((step <= 1.0).all())
    small_ok = small_ok and float(step[0]) == 0.0
    large_ok = large_ok and float(step[-1]) == 1.0
    results.append(ValidationResult("cutoff functions remain between 0 and 1 before pivot normalization", bounded, "x=1e-12..1e12 checked"))
    results.append(ValidationResult("F approaches 0 at sufficiently small k", small_ok, "x=1e-12 checked"))
    results.append(ValidationResult("F approaches 1 at sufficiently large k", large_ok, "x=1e12 checked"))

    smooth = scan.loc[
        scan["model_family"].isin([CutoffFamily.EXPONENTIAL.value, CutoffFamily.RATIONAL.value])
        & (scan["pivot_normalized"] == True)
    ]
    pivot_ok = True
    for _, row in smooth.iterrows():
        model = CutoffModel(
            CutoffFamily(str(row["model_family"])),
            kc_mpc_inverse=float(row["kc_Mpc_inverse"]),
            alpha=float(row["alpha"]),
            pivot_normalized=True,
            sharp_implementation=str(row.get("sharp_implementation", "not_applicable")),
        )
        value = float(modified_primordial_power(np.array([PIVOT_SCALAR]), model)[0])
        pivot_ok = pivot_ok and np.isclose(value, AS, rtol=1e-12, atol=0.0)
    results.append(
        ValidationResult(
            "pivot-normalized smooth spectra satisfy P(k_pivot) = As",
            pivot_ok and len(smooth) > 0,
            f"checked_rows={len(smooth)}",
        )
    )

    finite_power = True
    k = np.logspace(-6, 1, 700)
    for _, row in scan.iterrows():
        family = CutoffFamily(str(row["model_family"]))
        model = CutoffModel(
            family,
            kc_mpc_inverse=None if pd.isna(row["kc_Mpc_inverse"]) else float(row["kc_Mpc_inverse"]),
            alpha=None if pd.isna(row["alpha"]) else float(row["alpha"]),
            pivot_normalized=bool(row["pivot_normalized"]),
            sharp_implementation=str(row.get("sharp_implementation", "not_applicable")),
        )
        power = modified_primordial_power(k, model)
        finite_power = finite_power and np.isfinite(power).all() and bool((power >= 0.0).all())
    results.append(
        ValidationResult("all primordial spectra are finite and nonnegative", finite_power, f"models_checked={len(scan)}")
    )
    return results


def validate_spectra(index: pd.DataFrame) -> list[ValidationResult]:
    """Validate generated CAMB spectrum files."""

    results: list[ValidationResult] = []
    all_ell_ok = True
    all_values_ok = True
    missing: list[str] = []
    for _, row in index.iterrows():
        path = Path(row["spectrum_csv"])
        if not path.exists():
            all_ell_ok = False
            all_values_ok = False
            missing.append(str(path))
            continue
        spectrum = pd.read_csv(path)
        all_ell_ok = all_ell_ok and np.array_equal(spectrum["ell"].to_numpy(dtype=int), ELL_EXPECTED)
        values = spectrum[["Cl_TT_uK2", "Dl_TT_uK2"]].to_numpy(dtype=float)
        all_values_ok = all_values_ok and np.isfinite(values).all() and bool((spectrum["Dl_TT_uK2"].to_numpy(dtype=float) > 0.0).all())
    results.append(ValidationResult("all spectra include exactly ell = 2 through 2500", all_ell_ok, "all index spectra checked" if not missing else "; ".join(missing[:3])))
    results.append(ValidationResult("all CAMB D_ell values are finite and positive for ell >= 2", all_values_ok, f"spectra_checked={len(index)}"))
    return results


def validate_analysis_math(scan: pd.DataFrame, best: pd.DataFrame) -> list[ValidationResult]:
    """Validate ranking metrics and AIC calculations."""

    results: list[ValidationResult] = []
    baseline = scan.loc[scan["model_family"] == CutoffFamily.STANDARD.value].iloc[0]
    aic_ok = np.allclose(
        scan["AIC_ideal"].to_numpy(dtype=float),
        scan["Q_ideal_fullsky"].to_numpy(dtype=float) + 2.0 * scan["n_extra_parameters"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-10,
    ) and np.allclose(
        scan["Delta_AIC_vs_baseline"].to_numpy(dtype=float),
        scan["AIC_ideal"].to_numpy(dtype=float) - float(baseline["AIC_ideal"]),
        rtol=1e-12,
        atol=1e-10,
    )
    results.append(ValidationResult("AIC parameter penalties are applied correctly", aic_ok, "AIC = Q + 2k checked"))
    delta_q_ok = np.allclose(
        scan["Delta_Q_vs_baseline"].to_numpy(dtype=float),
        scan["Q_ideal_fullsky"].to_numpy(dtype=float) - float(baseline["Q_ideal_fullsky"]),
        rtol=1e-12,
        atol=1e-10,
    )
    results.append(ValidationResult("Delta_Q values are relative to the standard baseline", delta_q_ok, "Q - Q_baseline checked"))

    rankings_ok = True
    for family, label in [
        (CutoffFamily.EXPONENTIAL.value, "best exponential model by minimum Q"),
        (CutoffFamily.RATIONAL.value, "best rational model by minimum Q"),
        (CutoffFamily.SHARP.value, "best sharp model by minimum Q"),
    ]:
        expected_id = scan.loc[scan["model_family"] == family].sort_values("Q_ideal_fullsky").iloc[0]["model_id"]
        reported_id = best.loc[best["selection"] == label].iloc[0]["model_id"]
        rankings_ok = rankings_ok and expected_id == reported_id
    expected_cutoff_id = scan.loc[scan["model_family"] != CutoffFamily.STANDARD.value].sort_values("Q_ideal_fullsky").iloc[0]["model_id"]
    reported_cutoff_id = best.loc[best["selection"] == "overall best cutoff model by minimum Q"].iloc[0]["model_id"]
    expected_aic_id = scan.sort_values("AIC_ideal").iloc[0]["model_id"]
    reported_aic_id = best.loc[best["selection"] == "overall best model by minimum AIC"].iloc[0]["model_id"]
    rankings_ok = rankings_ok and expected_cutoff_id == reported_cutoff_id and expected_aic_id == reported_aic_id
    results.append(ValidationResult("best-model rows really are minima of the stated ranking metric", rankings_ok, "Q and AIC selections checked"))
    return results


def validate_band_variance_equivalence() -> ValidationResult:
    """Validate D_ell and C_ell band-variance formulas."""

    standard = safe_read_csv(STANDARD_CAMB_CSV)
    low = standard.loc[standard["ell"].isin(LOW_ELL_EXPECTED)]
    ell = low["ell"].to_numpy(dtype=int)
    dl = low["Dl_TT_uK2"].to_numpy(dtype=float)
    cl = low["Cl_TT_uK2"].to_numpy(dtype=float)
    v_d = compute_band_variance(ell, dl)
    v_c = float((1.0 / (4.0 * np.pi)) * np.sum((2.0 * ell + 1.0) * cl))
    return ValidationResult(
        "band-variance calculations agree using equivalent C_ell and D_ell formulas",
        np.isclose(v_d, v_c, rtol=1e-12, atol=1e-9),
        f"D_formula={v_d:.12g}, C_formula={v_c:.12g}",
    )


def validate_required_files() -> list[ValidationResult]:
    """Validate that required tables, figures, and text outputs exist."""

    missing_tables = [str(path) for path in REQUIRED_TABLES if not file_nonempty(path)]
    figure_paths = [FIGURE_DIR / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ["png", "pdf"]]
    missing_figures = [str(path) for path in figure_paths if not file_nonempty(path)]
    results = [
        ValidationResult(
            "no required table is empty",
            not missing_tables,
            "all required tables nonempty" if not missing_tables else "; ".join(missing_tables),
        ),
        ValidationResult(
            "no required figure is empty",
            not missing_figures,
            "all required figures nonempty" if not missing_figures else "; ".join(missing_figures[:4]),
        ),
        ValidationResult("methods summary Markdown exists and is nonempty", file_nonempty(METHODS_MD), str(METHODS_MD)),
    ]
    return results


def validate_baseline_and_sharp(scan: pd.DataFrame) -> list[ValidationResult]:
    """Validate baseline reproduction and sharp-step convergence reporting."""

    results: list[ValidationResult] = []
    baseline = safe_read_csv(BASELINE_REPRODUCTION_CSV)
    baseline_ok = (
        len(baseline) == 1
        and float(baseline["max_relative_difference_ell_2_29"].iloc[0]) < 1e-4
        and float(baseline["max_relative_difference_ell_30_2500"].iloc[0]) < 1e-4
    )
    results.append(
        ValidationResult(
            "standard custom-power result reproduces the original standard result",
            baseline_ok,
            baseline.to_dict(orient="records")[0] if len(baseline) else "missing",
        )
    )

    convergence = safe_read_csv(SHARP_CONVERGENCE_CSV)
    selected = ""
    if "selected_sharp_implementation" in convergence.columns and len(convergence):
        selected = str(convergence["selected_sharp_implementation"].iloc[0])
    exact_pass = len(convergence) > 0 and bool(convergence["passed"].all()) if "passed" in convergence.columns else False
    approx_reported = (
        selected == "numerical_sharp_step_approximation"
        and not scan.loc[scan["model_family"] == CutoffFamily.SHARP.value].empty
        and bool((scan.loc[scan["model_family"] == CutoffFamily.SHARP.value, "sharp_implementation"] == selected).all())
    )
    results.append(
        ValidationResult(
            "sharp-cutoff convergence test passes or numerical approximation is explicitly reported",
            exact_pass or approx_reported,
            f"selected_sharp_implementation={selected}",
        )
    )
    return results


def validate_repeated_execution() -> ValidationResult:
    """Rerun analysis and verify that numerical CSV outputs are identical."""

    paths = [SCAN_CSV, BEST_MODELS_CSV, BEST_LOWELL_SPECTRA_CSV, PRIMORDIAL_SAMPLES_CSV]
    before = numeric_signature(paths)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze_cutoff_models.main()
    after = numeric_signature(paths)
    return ValidationResult(
        "repeated execution with the same settings gives identical numerical outputs",
        signatures_equal(before, after),
        "numeric CSV outputs compared before and after rerunning analysis",
    )


def validate_cutoff_outputs() -> list[ValidationResult]:
    """Run all cutoff validation checks."""

    results: list[ValidationResult] = []
    scan = safe_read_csv(SCAN_CSV)
    best = safe_read_csv(BEST_MODELS_CSV)
    index = safe_read_csv(INDEX_CSV)

    results.extend(validate_required_files())
    results.append(ValidationResult("generation metadata JSON exists", file_nonempty(RUN_METADATA_JSON), str(RUN_METADATA_JSON)))
    results.extend(validate_grid(scan))
    if not scan.empty:
        results.extend(validate_cutoff_formulas(scan))
        results.extend(validate_analysis_math(scan, best))
        results.extend(validate_baseline_and_sharp(scan))
    if not index.empty:
        results.extend(validate_spectra(index))
    results.append(validate_band_variance_equivalence())
    if all(file_nonempty(path) for path in [SCAN_CSV, BEST_MODELS_CSV, BEST_LOWELL_SPECTRA_CSV, PRIMORDIAL_SAMPLES_CSV]):
        results.append(validate_repeated_execution())
    return results


def main() -> None:
    """Print validation results and exit nonzero on critical failure."""

    results = validate_cutoff_outputs()
    for result in results:
        print_result(result)

    failed = [result for result in results if result.critical and not result.passed]
    if failed:
        names = ", ".join(result.name for result in failed)
        raise SystemExit(f"Validation failed for critical checks: {names}")
    print("All critical cutoff-model validation checks passed.")


if __name__ == "__main__":
    main()
