"""Validate the generated CAMB theoretical TT spectrum output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECTRUM_CSV = PROJECT_ROOT / "outputs" / "tables" / "camb_planck2018_tt_spectrum.csv"
REQUIRED_COLUMNS = ["ell", "Cl_TT_uK2", "Dl_TT_uK2"]


@dataclass
class ValidationResult:
    """Single validation result with a machine-checkable pass/fail flag."""

    name: str
    passed: bool
    detail: str
    critical: bool = True


def print_result(result: ValidationResult) -> None:
    """Print one validation result in a consistent PASS/FAIL format."""

    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.name} - {result.detail}")


def load_spectrum(csv_path: Path) -> pd.DataFrame:
    """Load the generated spectrum CSV or raise a clear error."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Required output file does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def validate_dataframe(dataframe: pd.DataFrame) -> list[ValidationResult]:
    """Run automated scientific and computational sanity checks."""

    results: list[ValidationResult] = []

    columns_exist = all(column in dataframe.columns for column in REQUIRED_COLUMNS)
    results.append(
        ValidationResult(
            "required columns exist",
            columns_exist,
            f"required={REQUIRED_COLUMNS}; found={list(dataframe.columns)}",
        )
    )
    if not columns_exist:
        return results

    ell = dataframe["ell"].to_numpy()
    cl_tt = dataframe["Cl_TT_uK2"].to_numpy()
    dl_tt = dataframe["Dl_TT_uK2"].to_numpy()

    results.append(
        ValidationResult(
            "first scientific multipole is ell = 2",
            len(ell) > 0 and int(ell[0]) == 2,
            f"first ell={ell[0] if len(ell) else 'none'}",
        )
    )

    consecutive = len(ell) > 1 and np.array_equal(np.diff(ell), np.ones(len(ell) - 1))
    results.append(
        ValidationResult(
            "multipoles increase consecutively",
            consecutive,
            "all ell steps are 1" if consecutive else "one or more ell steps are not 1",
        )
    )

    finite = bool(np.isfinite(dataframe[REQUIRED_COLUMNS].to_numpy()).all())
    results.append(
        ValidationResult(
            "no NaN or infinite values",
            finite,
            "all required numeric values are finite" if finite else "non-finite values found",
        )
    )

    nonnegative = bool(((cl_tt >= 0) & (dl_tt >= 0)).all())
    results.append(
        ValidationResult(
            "TT power values are nonnegative",
            nonnegative,
            "all Cl_TT_uK2 and Dl_TT_uK2 values are >= 0"
            if nonnegative
            else "negative TT power values found",
        )
    )

    expected_dl = ell * (ell + 1) * cl_tt / (2.0 * np.pi)
    conversion_ok = bool(np.allclose(dl_tt, expected_dl, rtol=1e-10, atol=1e-8))
    max_abs_difference = float(np.max(np.abs(dl_tt - expected_dl))) if len(ell) else float("nan")
    results.append(
        ValidationResult(
            "Cl and Dl conversion formula holds",
            conversion_ok,
            f"max absolute difference={max_abs_difference:.3e} microK^2",
        )
    )

    reaches_2500 = len(ell) > 0 and int(np.max(ell)) >= 2500
    results.append(
        ValidationResult(
            "output includes at least ell = 2500",
            reaches_2500,
            f"max ell={int(np.max(ell)) if len(ell) else 'none'}",
        )
    )

    low_ell_expected = np.arange(2, 30)
    low_ell_actual = ell[(ell >= 2) & (ell <= 29)]
    low_ell_ok = np.array_equal(low_ell_actual, low_ell_expected)
    results.append(
        ValidationResult(
            "low-ell range contains exactly ell = 2 through 29",
            low_ell_ok,
            f"count={len(low_ell_actual)}",
        )
    )

    # Theoretical spectra should show broad acoustic peaks: enough variation,
    # a clear first high peak region, and multiple local maxima in D_ell.
    dl_std = float(np.std(dl_tt))
    dl_mean = float(np.mean(dl_tt))
    local_maxima = np.where((dl_tt[1:-1] > dl_tt[:-2]) & (dl_tt[1:-1] > dl_tt[2:]))[0] + 1
    acoustic_behavior = bool(
        dl_mean > 0
        and dl_std > 0.05 * dl_mean
        and len(local_maxima) >= 3
        and 150 <= int(ell[int(np.argmax(dl_tt))]) <= 400
    )
    results.append(
        ValidationResult(
            "spectrum shows recognizable acoustic-peak behavior",
            acoustic_behavior,
            (
                f"std/mean={dl_std / dl_mean:.3f}, local maxima={len(local_maxima)}, "
                f"global-peak ell={int(ell[int(np.argmax(dl_tt))]) if len(ell) else 'none'}"
            ),
        )
    )

    return results


def main() -> None:
    """Run all validation checks and fail if any critical test fails."""

    results: list[ValidationResult] = []
    file_exists = SPECTRUM_CSV.exists()
    results.append(
        ValidationResult(
            "output file exists",
            file_exists,
            str(SPECTRUM_CSV),
        )
    )

    dataframe = load_spectrum(SPECTRUM_CSV) if file_exists else pd.DataFrame()
    if file_exists:
        results.extend(validate_dataframe(dataframe))

    for result in results:
        print_result(result)

    failed_critical = [result for result in results if result.critical and not result.passed]
    if failed_critical:
        failed_names = ", ".join(result.name for result in failed_critical)
        raise SystemExit(f"Validation failed for critical checks: {failed_names}")

    print("All critical validation checks passed.")


if __name__ == "__main__":
    main()
