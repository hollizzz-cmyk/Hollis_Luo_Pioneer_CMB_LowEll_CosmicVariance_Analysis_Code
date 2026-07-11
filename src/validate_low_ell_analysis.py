"""Validate Stage 2 low-ell Planck-CAMB comparison outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PLANCK_FILE = PROJECT_ROOT / "data" / "raw" / "COM_PowerSpect_CMB-TT-full_R3.01.txt"
PROCESSED_PLANCK_LOWELL = PROJECT_ROOT / "data" / "processed" / "planck2018_tt_lowell.csv"
CAMB_SPECTRUM = PROJECT_ROOT / "outputs" / "tables" / "camb_planck2018_tt_spectrum.csv"
COMPARISON_CSV = PROJECT_ROOT / "outputs" / "tables" / "lowell_planck_camb_comparison.csv"
LOW_ELL_RANGE = np.arange(2, 30)

PROCESSED_PLANCK_COLUMNS = [
    "ell",
    "planck_Dl_uK2",
    "planck_error_lower_uK2",
    "planck_error_upper_uK2",
]
COMPARISON_COLUMNS = [
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
FIGURE_PATHS = [
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_comparison.png",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_comparison.pdf",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_residuals.png",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_residuals.pdf",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_fractional_difference.png",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_fractional_difference.pdf",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_normalized_residuals.png",
    PROJECT_ROOT / "outputs" / "figures" / "planck_camb_lowell_normalized_residuals.pdf",
]


@dataclass
class ValidationResult:
    """Single validation result."""

    name: str
    passed: bool
    detail: str
    critical: bool = True


def print_result(result: ValidationResult) -> None:
    """Print a PASS/FAIL validation result."""

    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.name} - {result.detail}")


def first_nonempty_line(path: Path) -> str:
    """Return the first nonempty line from a text file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def check_file_exists(path: Path, name: str) -> ValidationResult:
    """Validate that a file exists."""

    return ValidationResult(name, path.exists(), str(path))


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty dataframe."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def columns_exist(dataframe: pd.DataFrame, columns: list[str]) -> bool:
    """Return True if all requested columns are present."""

    return all(column in dataframe.columns for column in columns)


def values_are_finite(dataframe: pd.DataFrame, columns: list[str]) -> bool:
    """Return True if all values in selected columns are finite."""

    if not columns_exist(dataframe, columns):
        return False
    return bool(np.isfinite(dataframe[columns].to_numpy(dtype=float)).all())


def sorted_by_ell(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe sorted by ell with a clean integer index."""

    if "ell" not in dataframe.columns:
        return pd.DataFrame()
    return dataframe.sort_values("ell").reset_index(drop=True)


def parse_raw_planck_lowell(raw_path: Path) -> pd.DataFrame:
    """Parse ell = 2 through 29 independently from the official raw Planck file."""

    if not raw_path.exists():
        return pd.DataFrame(columns=PROCESSED_PLANCK_COLUMNS)

    raw = pd.read_csv(
        raw_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=PROCESSED_PLANCK_COLUMNS,
        engine="python",
    )
    if raw.empty:
        return raw

    ell_values = raw["ell"].to_numpy(dtype=float)
    if not np.allclose(ell_values, np.round(ell_values), rtol=0.0, atol=1e-12):
        return pd.DataFrame(columns=PROCESSED_PLANCK_COLUMNS)

    raw["ell"] = np.round(ell_values).astype(int)
    for column in PROCESSED_PLANCK_COLUMNS[1:]:
        raw[column] = raw[column].astype(float)

    low_ell = raw[raw["ell"].isin(LOW_ELL_RANGE)].copy()
    return sorted_by_ell(low_ell)


def validate_stage2_outputs() -> list[ValidationResult]:
    """Run all Stage 2 validation checks."""

    results: list[ValidationResult] = []

    results.append(check_file_exists(RAW_PLANCK_FILE, "official raw Planck file exists"))
    results.append(check_file_exists(PROCESSED_PLANCK_LOWELL, "processed Planck CSV exists"))
    results.append(check_file_exists(CAMB_SPECTRUM, "Stage 1 CAMB CSV exists"))
    results.append(check_file_exists(COMPARISON_CSV, "final comparison CSV exists"))

    processed = safe_read_csv(PROCESSED_PLANCK_LOWELL)
    comparison = safe_read_csv(COMPARISON_CSV)
    camb = safe_read_csv(CAMB_SPECTRUM)
    processed_sorted = sorted_by_ell(processed)
    comparison_sorted = sorted_by_ell(comparison)

    results.append(
        ValidationResult(
            "Planck processed file contains exactly 28 rows",
            len(processed) == 28,
            f"rows={len(processed)}",
        )
    )

    results.append(
        ValidationResult(
            "comparison table contains exactly 28 rows",
            len(comparison) == 28,
            f"rows={len(comparison)}",
        )
    )

    processed_ell_ok = (
        "ell" in processed.columns
        and np.array_equal(processed["ell"].to_numpy(dtype=int), LOW_ELL_RANGE)
    )
    comparison_ell_ok = (
        "ell" in comparison.columns
        and np.array_equal(comparison["ell"].to_numpy(dtype=int), LOW_ELL_RANGE)
    )
    results.append(
        ValidationResult(
            "multipoles are exactly integers 2 through 29",
            processed_ell_ok and comparison_ell_ok,
            "processed and comparison ell columns checked",
        )
    )

    processed_columns_ok = columns_exist(processed, PROCESSED_PLANCK_COLUMNS)
    comparison_columns_ok = columns_exist(comparison, COMPARISON_COLUMNS)
    camb_columns_ok = columns_exist(camb, ["ell", "Dl_TT_uK2"])
    results.append(
        ValidationResult(
            "all required columns exist",
            processed_columns_ok and comparison_columns_ok and camb_columns_ok,
            "processed Planck, CAMB, and comparison columns checked",
        )
    )

    finite_ok = values_are_finite(processed, PROCESSED_PLANCK_COLUMNS) and values_are_finite(
        comparison, COMPARISON_COLUMNS
    )
    results.append(
        ValidationResult(
            "no values are NaN or infinite",
            finite_ok,
            "processed Planck and comparison numeric columns checked",
        )
    )

    errors_ok = (
        processed_columns_ok
        and bool(
            (
                (processed["planck_error_lower_uK2"] >= 0)
                & (processed["planck_error_upper_uK2"] >= 0)
            ).all()
        )
    )
    results.append(
        ValidationResult(
            "Planck uncertainty magnitudes are nonnegative",
            errors_ok,
            "lower and upper asymmetric confidence-interval magnitudes checked",
        )
    )

    camb_low = pd.DataFrame()
    if camb_columns_ok:
        camb_low = camb[camb["ell"].isin(LOW_ELL_RANGE)].copy()
    camb_low_sorted = sorted_by_ell(camb_low)
    camb_positive = (
        camb_columns_ok
        and len(camb_low) == 28
        and bool((camb_low["Dl_TT_uK2"].to_numpy(dtype=float) > 0).all())
    )
    results.append(
        ValidationResult(
            "CAMB theoretical D_ell values are strictly positive over ell 2 through 29",
            camb_positive,
            f"low-ell CAMB rows={len(camb_low)}",
        )
    )

    merge_one_to_one = False
    if processed_columns_ok and camb_columns_ok:
        try:
            pd.merge(
                processed,
                camb_low[["ell", "Dl_TT_uK2"]],
                on="ell",
                how="inner",
                validate="one_to_one",
            )
            merge_one_to_one = True
        except pd.errors.MergeError:
            merge_one_to_one = False
    results.append(
        ValidationResult("merge is one-to-one", merge_one_to_one, "Planck low-ell and CAMB low-ell")
    )

    duplicates_ok = (
        processed_columns_ok
        and comparison_columns_ok
        and not processed["ell"].duplicated().any()
        and not comparison["ell"].duplicated().any()
    )
    results.append(
        ValidationResult(
            "there are no duplicate multipoles",
            duplicates_ok,
            "processed and comparison ell columns checked",
        )
    )

    source_columns = [
        "planck_Dl_uK2",
        "planck_error_lower_uK2",
        "planck_error_upper_uK2",
    ]
    planck_source_matches = False
    if processed_columns_ok and comparison_columns_ok and len(processed_sorted) == len(comparison_sorted):
        ell_match = np.array_equal(
            comparison_sorted["ell"].to_numpy(dtype=int),
            processed_sorted["ell"].to_numpy(dtype=int),
        )
        value_match = np.allclose(
            comparison_sorted[source_columns].to_numpy(dtype=float),
            processed_sorted[source_columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        planck_source_matches = bool(ell_match and value_match)
    results.append(
        ValidationResult(
            "comparison Planck columns match processed Planck source",
            planck_source_matches,
            "ell, Dl, and asymmetric interval columns checked exactly",
        )
    )

    camb_source_matches = False
    if (
        comparison_columns_ok
        and camb_columns_ok
        and len(comparison_sorted) == 28
        and len(camb_low_sorted) == 28
    ):
        ell_match = np.array_equal(
            comparison_sorted["ell"].to_numpy(dtype=int),
            camb_low_sorted["ell"].to_numpy(dtype=int),
        )
        value_match = np.allclose(
            comparison_sorted["camb_Dl_uK2"].to_numpy(dtype=float),
            camb_low_sorted["Dl_TT_uK2"].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-9,
        )
        camb_source_matches = bool(ell_match and value_match)
    results.append(
        ValidationResult(
            "comparison CAMB column matches Stage 1 CAMB source",
            camb_source_matches,
            "ell and D_ell values checked against outputs/tables/camb_planck2018_tt_spectrum.csv",
        )
    )

    raw_low_sorted = parse_raw_planck_lowell(RAW_PLANCK_FILE)
    raw_source_matches = False
    if processed_columns_ok and len(processed_sorted) == 28 and len(raw_low_sorted) == 28:
        ell_match = np.array_equal(
            processed_sorted["ell"].to_numpy(dtype=int),
            raw_low_sorted["ell"].to_numpy(dtype=int),
        )
        value_match = np.allclose(
            processed_sorted[PROCESSED_PLANCK_COLUMNS[1:]].to_numpy(dtype=float),
            raw_low_sorted[PROCESSED_PLANCK_COLUMNS[1:]].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        raw_source_matches = bool(ell_match and value_match)
    results.append(
        ValidationResult(
            "processed Planck table matches official raw Planck source",
            raw_source_matches,
            "ell, Dl, and asymmetric interval columns checked against raw file",
        )
    )

    if comparison_columns_ok:
        residual_expected = comparison["planck_Dl_uK2"] - comparison["camb_Dl_uK2"]
        fractional_expected = residual_expected / comparison["camb_Dl_uK2"]
        percent_expected = 100.0 * fractional_expected
        cv_fraction_expected = np.sqrt(2.0 / (2.0 * comparison["ell"] + 1.0))
        cv_sigma_expected = comparison["camb_Dl_uK2"] * cv_fraction_expected
        normalized_expected = residual_expected / cv_sigma_expected
        cv_lower_expected = comparison["camb_Dl_uK2"] - cv_sigma_expected
        cv_upper_expected = comparison["camb_Dl_uK2"] + cv_sigma_expected
    else:
        residual_expected = fractional_expected = percent_expected = pd.Series(dtype=float)
        cv_fraction_expected = cv_sigma_expected = normalized_expected = pd.Series(dtype=float)
        cv_lower_expected = cv_upper_expected = pd.Series(dtype=float)

    results.append(
        ValidationResult(
            "residual calculation is correct",
            comparison_columns_ok
            and bool(np.allclose(comparison["residual_Dl_uK2"], residual_expected, rtol=1e-12, atol=1e-9)),
            "planck_Dl_uK2 - camb_Dl_uK2",
        )
    )
    results.append(
        ValidationResult(
            "fractional-difference calculation is correct",
            comparison_columns_ok
            and bool(np.allclose(comparison["fractional_difference"], fractional_expected, rtol=1e-12, atol=1e-12))
            and bool(np.allclose(comparison["percent_difference"], percent_expected, rtol=1e-12, atol=1e-10)),
            "residual / CAMB and percent conversion",
        )
    )
    results.append(
        ValidationResult(
            "cosmic-variance calculation is correct",
            comparison_columns_ok
            and bool(
                np.allclose(
                    comparison["cosmic_variance_fraction"],
                    cv_fraction_expected,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            and bool(np.allclose(comparison["cosmic_variance_sigma_uK2"], cv_sigma_expected, rtol=1e-12, atol=1e-9))
            and bool(np.allclose(comparison["camb_cv_lower_uK2"], cv_lower_expected, rtol=1e-12, atol=1e-9))
            and bool(np.allclose(comparison["camb_cv_upper_uK2"], cv_upper_expected, rtol=1e-12, atol=1e-9)),
            "sqrt(2/(2 ell + 1)) and CAMB +/- sigma",
        )
    )
    results.append(
        ValidationResult(
            "normalized-residual calculation is correct",
            comparison_columns_ok
            and bool(np.allclose(comparison["normalized_residual_cv"], normalized_expected, rtol=1e-12, atol=1e-12)),
            "residual divided by ideal cosmic-variance sigma",
        )
    )

    missing_or_empty_figures = [
        str(path) for path in FIGURE_PATHS if (not path.exists()) or path.stat().st_size <= 0
    ]
    results.append(
        ValidationResult(
            "all required PNG and PDF figures exist and are nonempty",
            not missing_or_empty_figures,
            "all figure files present" if not missing_or_empty_figures else "; ".join(missing_or_empty_figures),
        )
    )

    raw_not_processed = False
    if RAW_PLANCK_FILE.exists():
        header = first_nonempty_line(RAW_PLANCK_FILE)
        raw_not_processed = header.startswith("#") and {"l", "Dl", "-dDl", "+dDl"}.issubset(
            set(header.split())
        )
    results.append(
        ValidationResult(
            "original raw Planck file has not been overwritten by processed data",
            raw_not_processed,
            "raw file starts with the expected commented Planck header",
        )
    )

    return results


def main() -> None:
    """Run validation checks and fail if any critical check fails."""

    results = validate_stage2_outputs()
    for result in results:
        print_result(result)

    failed = [result for result in results if result.critical and not result.passed]
    if failed:
        failed_names = ", ".join(result.name for result in failed)
        raise SystemExit(f"Validation failed for critical checks: {failed_names}")

    print("All critical Stage 2 validation checks passed.")


if __name__ == "__main__":
    main()
