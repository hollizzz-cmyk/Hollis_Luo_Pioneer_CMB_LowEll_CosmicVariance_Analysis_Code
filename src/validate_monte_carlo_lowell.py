"""Validate Stage 3 low-ell Monte Carlo outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from monte_carlo_lowell import (
    CDF_PDF,
    CDF_PNG,
    COMPARISON_CSV,
    ELL_MAX,
    ELL_MIN,
    HISTOGRAM_PDF,
    HISTOGRAM_PNG,
    METHODS_MD,
    PER_ELL_MOMENTS_CSV,
    QUANTILES_CSV,
    QUANTILE_PROBABILITIES,
    SAMPLES_NPZ,
    SENSITIVITY_CSV,
    SUMMARY_CSV,
    UPPER_CUTOFFS,
    MonteCarloConfig,
    calculate_binomial_interval,
    compute_band_variance,
    compute_variance_weights,
    simulate_lowell_power_ratios,
)


REQUIRED_INPUT_COLUMNS = ["ell", "planck_Dl_uK2", "camb_Dl_uK2"]
EXPECTED_ELL = np.arange(ELL_MIN, ELL_MAX + 1)
OUTPUT_FILES = [
    SUMMARY_CSV,
    SENSITIVITY_CSV,
    QUANTILES_CSV,
    PER_ELL_MOMENTS_CSV,
    SAMPLES_NPZ,
    HISTOGRAM_PNG,
    HISTOGRAM_PDF,
    CDF_PNG,
    CDF_PDF,
    METHODS_MD,
]


@dataclass
class ValidationResult:
    """Single validation result."""

    name: str
    passed: bool
    detail: str
    critical: bool = True


def print_result(result: ValidationResult) -> None:
    """Print one PASS/FAIL result."""

    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.name} - {result.detail}")


def file_exists_and_nonempty(path: Path) -> bool:
    """Return True if a path exists and has nonzero size."""

    return path.exists() and path.stat().st_size > 0


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if possible, otherwise return an empty dataframe."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summary_dict(summary: pd.DataFrame) -> dict[str, float]:
    """Convert the metric/value summary CSV to a dictionary of floats."""

    if not {"metric", "value"}.issubset(summary.columns):
        return {}
    return {str(row["metric"]): float(row["value"]) for _, row in summary.iterrows()}


def load_npz(path: Path) -> np.lib.npyio.NpzFile | None:
    """Load the simulation NPZ if it exists."""

    if not path.exists():
        return None
    return np.load(path)


def compute_observed_r_from_d_ell(comparison: pd.DataFrame) -> float:
    """Independently compute R_low from the direct D_ell weighted formula."""

    ell = comparison["ell"].to_numpy(dtype=int)
    weights = compute_variance_weights(ell)
    observed = comparison["planck_Dl_uK2"].to_numpy(dtype=float)
    theory = comparison["camb_Dl_uK2"].to_numpy(dtype=float)
    return float(np.sum(weights * observed) / np.sum(weights * theory))


def compute_observed_r_from_c_ell(comparison: pd.DataFrame) -> float:
    """Compute the same R_low through the equivalent C_ell variance formula."""

    ell = comparison["ell"].to_numpy(dtype=float)
    observed_dl = comparison["planck_Dl_uK2"].to_numpy(dtype=float)
    theory_dl = comparison["camb_Dl_uK2"].to_numpy(dtype=float)
    observed_cl = 2.0 * np.pi * observed_dl / (ell * (ell + 1.0))
    theory_cl = 2.0 * np.pi * theory_dl / (ell * (ell + 1.0))
    numerator = np.sum((2.0 * ell + 1.0) * observed_cl)
    denominator = np.sum((2.0 * ell + 1.0) * theory_cl)
    return float(numerator / denominator)


def validate_stage3_outputs() -> list[ValidationResult]:
    """Run all Stage 3 validation checks."""

    results: list[ValidationResult] = []

    comparison = safe_read_csv(COMPARISON_CSV)
    summary = safe_read_csv(SUMMARY_CSV)
    sensitivity = safe_read_csv(SENSITIVITY_CSV)
    quantiles = safe_read_csv(QUANTILES_CSV)
    moments = safe_read_csv(PER_ELL_MOMENTS_CSV)
    summary_values = summary_dict(summary)
    npz = load_npz(SAMPLES_NPZ)

    results.append(
        ValidationResult(
            "Stage 2 comparison file exists",
            file_exists_and_nonempty(COMPARISON_CSV),
            str(COMPARISON_CSV),
        )
    )
    results.append(
        ValidationResult(
            "comparison table contains exactly 28 rows",
            len(comparison) == 28,
            f"rows={len(comparison)}",
        )
    )
    ell_ok = "ell" in comparison.columns and np.array_equal(
        comparison["ell"].to_numpy(dtype=int), EXPECTED_ELL
    )
    results.append(
        ValidationResult(
            "multipoles are exactly integers 2 through 29",
            ell_ok,
            "ell column checked",
        )
    )
    required_columns_ok = all(column in comparison.columns for column in REQUIRED_INPUT_COLUMNS)
    results.append(
        ValidationResult(
            "required input columns exist",
            required_columns_ok,
            str(REQUIRED_INPUT_COLUMNS),
        )
    )
    finite_input_ok = required_columns_ok and bool(
        np.isfinite(comparison[["planck_Dl_uK2", "camb_Dl_uK2"]].to_numpy(dtype=float)).all()
    )
    results.append(
        ValidationResult(
            "observed and theoretical D_ell values are finite",
            finite_input_ok,
            "Planck and CAMB D_ell columns checked",
        )
    )
    theory_positive_ok = required_columns_ok and bool(
        (comparison["camb_Dl_uK2"].to_numpy(dtype=float) > 0).all()
    )
    results.append(
        ValidationResult(
            "theoretical D_ell values are strictly positive",
            theory_positive_ok,
            "CAMB D_ell column checked",
        )
    )

    missing_outputs = [str(path) for path in OUTPUT_FILES if not file_exists_and_nonempty(path)]
    results.append(
        ValidationResult(
            "summary, quantile, moment, NPZ, PNG, PDF, and Markdown outputs exist and are nonempty",
            not missing_outputs,
            "all present" if not missing_outputs else "; ".join(missing_outputs),
        )
    )

    r_low_sim = np.array([], dtype=float)
    npz_keys_ok = False
    required_cutoff_keys = {f"R_low_L{cutoff}" for cutoff in UPPER_CUTOFFS}
    if npz is not None:
        allowed_keys = {
            "R_low_sim",
            "n_simulations",
            "seed",
            "ell_min",
            "ell_max",
            "observed_R_low",
            *required_cutoff_keys,
        }
        npz_keys_ok = set(npz.files) == allowed_keys
        if "R_low_sim" in npz.files:
            r_low_sim = npz["R_low_sim"]

    n_summary = int(summary_values.get("n_simulations", -1))
    seed_summary = int(summary_values.get("seed", -1))
    chunk_summary = int(summary_values.get("chunk_size", -1))
    n_npz = int(npz["n_simulations"]) if npz is not None and "n_simulations" in npz.files else -1
    seed_npz = int(npz["seed"]) if npz is not None and "seed" in npz.files else -1

    results.append(
        ValidationResult(
            "saved simulation array contains exactly N values",
            r_low_sim.ndim == 1 and len(r_low_sim) == n_summary and n_summary > 0,
            f"array length={len(r_low_sim)}, summary N={n_summary}",
        )
    )
    cutoff_arrays_ok = False
    if npz is not None and required_cutoff_keys.issubset(npz.files):
        cutoff_arrays_ok = all(
            npz[key].ndim == 1 and len(npz[key]) == n_summary and np.isfinite(npz[key]).all()
            for key in required_cutoff_keys
        )
    results.append(
        ValidationResult(
            "saved upper-cutoff simulation arrays are one-dimensional and finite",
            cutoff_arrays_ok,
            ", ".join(sorted(required_cutoff_keys)),
        )
    )
    results.append(
        ValidationResult(
            "all simulated R_low values are finite and strictly positive",
            len(r_low_sim) > 0 and bool(np.isfinite(r_low_sim).all()) and bool((r_low_sim > 0).all()),
            "R_low_sim checked",
        )
    )
    results.append(
        ValidationResult(
            "saved seed and simulation count match the summary",
            n_npz == n_summary and seed_npz == seed_summary,
            f"NPZ N={n_npz}, summary N={n_summary}; NPZ seed={seed_npz}, summary seed={seed_summary}",
        )
    )

    observed_r_direct = compute_observed_r_from_d_ell(comparison) if required_columns_ok else float("nan")
    observed_r_summary = summary_values.get("observed_R_low", float("nan"))
    results.append(
        ValidationResult(
            "observed R_low is independently recomputed from Stage 2 table",
            np.isclose(observed_r_direct, observed_r_summary, rtol=1e-12, atol=1e-12),
            f"recomputed={observed_r_direct:.12g}, summary={observed_r_summary:.12g}",
        )
    )

    observed_r_c_ell = compute_observed_r_from_c_ell(comparison) if required_columns_ok else float("nan")
    results.append(
        ValidationResult(
            "direct D_ell weighted formula agrees with equivalent C_ell formula",
            np.isclose(observed_r_direct, observed_r_c_ell, rtol=1e-12, atol=1e-12),
            f"D_ell={observed_r_direct:.12g}, C_ell={observed_r_c_ell:.12g}",
        )
    )

    if required_columns_ok:
        ell = comparison["ell"].to_numpy(dtype=int)
        theory = comparison["camb_Dl_uK2"].to_numpy(dtype=float)
        weights = compute_variance_weights(ell)
        theory_r = float(np.sum(weights * theory) / np.sum(weights * theory))
    else:
        theory_r = float("nan")
    results.append(
        ValidationResult(
            "theoretical R_low equals 1 when theory is used in numerator and denominator",
            np.isclose(theory_r, 1.0, rtol=0.0, atol=1e-15),
            f"theoretical_R_low={theory_r:.16g}",
        )
    )

    lower_tail_count = int(np.sum(r_low_sim <= observed_r_summary)) if len(r_low_sim) else -1
    summary_count = int(summary_values.get("lower_tail_count", -2))
    results.append(
        ValidationResult(
            "reported lower-tail count is correct",
            lower_tail_count == summary_count,
            f"recomputed={lower_tail_count}, summary={summary_count}",
        )
    )

    expected_p = (lower_tail_count + 1.0) / (n_summary + 1.0) if n_summary > 0 else float("nan")
    summary_p = summary_values.get("lower_tail_p_value", float("nan"))
    results.append(
        ValidationResult(
            "reported add-one p-value is correct",
            np.isclose(expected_p, summary_p, rtol=0.0, atol=1e-15),
            f"expected={expected_p:.12g}, summary={summary_p:.12g}",
        )
    )

    expected_se = np.sqrt(summary_p * (1.0 - summary_p) / n_summary) if n_summary > 0 else float("nan")
    summary_se = summary_values.get("monte_carlo_standard_error", float("nan"))
    results.append(
        ValidationResult(
            "reported Monte Carlo standard error is correct",
            np.isclose(expected_se, summary_se, rtol=1e-12, atol=1e-15),
            f"expected={expected_se:.12g}, summary={summary_se:.12g}",
        )
    )

    ci_lower, ci_upper = calculate_binomial_interval(lower_tail_count, n_summary) if n_summary > 0 else (float("nan"), float("nan"))
    results.append(
        ValidationResult(
            "reported 95% binomial interval is correct",
            np.isclose(ci_lower, summary_values.get("p_value_ci_95_lower", float("nan")), rtol=1e-12, atol=1e-15)
            and np.isclose(ci_upper, summary_values.get("p_value_ci_95_upper", float("nan")), rtol=1e-12, atol=1e-15),
            f"expected=[{ci_lower:.12g}, {ci_upper:.12g}]",
        )
    )

    sensitivity_required = {
        "ell_min",
        "ell_max",
        "observed_band_variance_uK2",
        "theoretical_band_variance_uK2",
        "observed_R_low",
        "fractional_suppression",
        "percent_suppression",
        "lower_tail_count",
        "lower_tail_p_value",
        "monte_carlo_standard_error",
        "p_value_ci_95_lower",
        "p_value_ci_95_upper",
    }
    sensitivity_columns_ok = sensitivity_required.issubset(sensitivity.columns)
    sensitivity_cutoffs_ok = sensitivity_columns_ok and np.array_equal(
        sensitivity.sort_values("ell_max")["ell_max"].to_numpy(dtype=int),
        np.array(UPPER_CUTOFFS, dtype=int),
    )
    results.append(
        ValidationResult(
            "upper-cutoff sensitivity table contains L=10, L=20, and L=29",
            sensitivity_cutoffs_ok,
            str(SENSITIVITY_CSV),
        )
    )
    sensitivity_values_ok = False
    sensitivity_p_ok = False
    if sensitivity_columns_ok and required_columns_ok and npz is not None:
        value_checks: list[bool] = []
        p_checks: list[bool] = []
        for _, row in sensitivity.iterrows():
            cutoff = int(row["ell_max"])
            subset = comparison.loc[comparison["ell"] <= cutoff]
            ell_subset = subset["ell"].to_numpy(dtype=int)
            observed_band = compute_band_variance(ell_subset, subset["planck_Dl_uK2"].to_numpy(dtype=float))
            theoretical_band = compute_band_variance(ell_subset, subset["camb_Dl_uK2"].to_numpy(dtype=float))
            observed_r = observed_band / theoretical_band
            value_checks.append(
                np.isclose(observed_band, row["observed_band_variance_uK2"], rtol=1e-12, atol=1e-9)
                and np.isclose(theoretical_band, row["theoretical_band_variance_uK2"], rtol=1e-12, atol=1e-9)
                and np.isclose(observed_r, row["observed_R_low"], rtol=1e-12, atol=1e-12)
            )
            key = f"R_low_L{cutoff}"
            if key in npz.files:
                cutoff_count = int(np.sum(npz[key] <= observed_r))
                cutoff_p = (cutoff_count + 1.0) / (n_summary + 1.0)
                cutoff_se = np.sqrt(cutoff_p * (1.0 - cutoff_p) / n_summary) if n_summary > 0 else float("nan")
                ci_lower_cutoff, ci_upper_cutoff = calculate_binomial_interval(cutoff_count, n_summary)
                p_checks.append(
                    cutoff_count == int(row["lower_tail_count"])
                    and np.isclose(cutoff_p, row["lower_tail_p_value"], rtol=0.0, atol=1e-15)
                    and np.isclose(cutoff_se, row["monte_carlo_standard_error"], rtol=1e-12, atol=1e-15)
                    and np.isclose(ci_lower_cutoff, row["p_value_ci_95_lower"], rtol=1e-12, atol=1e-15)
                    and np.isclose(ci_upper_cutoff, row["p_value_ci_95_upper"], rtol=1e-12, atol=1e-15)
                )
        sensitivity_values_ok = bool(value_checks) and all(value_checks)
        sensitivity_p_ok = bool(p_checks) and all(p_checks)
    results.append(
        ValidationResult(
            "upper-cutoff observed statistics are independently recomputed",
            sensitivity_values_ok,
            "L=10,20,29 band variances and R_low checked",
        )
    )
    results.append(
        ValidationResult(
            "upper-cutoff p-values and intervals are independently recomputed",
            sensitivity_p_ok,
            "L=10,20,29 counts, p-values, SE, and intervals checked",
        )
    )

    quantile_columns_ok = {"probability", "R_low_quantile"}.issubset(quantiles.columns)
    quantiles_match = False
    if quantile_columns_ok and len(r_low_sim):
        quantiles_sorted = quantiles.sort_values("probability").reset_index(drop=True)
        quantiles_match = np.array_equal(
            quantiles_sorted["probability"].to_numpy(dtype=float),
            QUANTILE_PROBABILITIES,
        ) and np.allclose(
            quantiles_sorted["R_low_quantile"].to_numpy(dtype=float),
            np.quantile(r_low_sim, QUANTILE_PROBABILITIES),
            rtol=1e-12,
            atol=1e-12,
        )
    results.append(
        ValidationResult(
            "reported quantiles match independently calculated NumPy quantiles",
            quantiles_match,
            "requested probabilities checked",
        )
    )

    quantiles_monotonic = quantile_columns_ok and bool(
        np.all(np.diff(quantiles.sort_values("probability")["R_low_quantile"].to_numpy(dtype=float)) >= 0)
    )
    results.append(
        ValidationResult("quantiles increase monotonically", quantiles_monotonic, "R_low_quantile checked")
    )

    simulated_mean = summary_values.get("simulated_R_mean", float("nan"))
    simulated_std = summary_values.get("simulated_R_standard_deviation", float("nan"))
    sem = simulated_std / np.sqrt(n_summary) if n_summary > 0 else float("nan")
    mean_close = abs(simulated_mean - 1.0) <= max(6.0 * sem, 1e-4)
    results.append(
        ValidationResult(
            "simulated mean R_low is close to 1",
            mean_close,
            f"mean={simulated_mean:.12g}, threshold={max(6.0 * sem, 1e-4):.6g}",
        )
    )

    per_ell_required = [
        "ell",
        "degrees_of_freedom",
        "theoretical_Dl_uK2",
        "empirical_mean_Dl_uK2",
        "expected_standard_deviation_uK2",
        "empirical_standard_deviation_uK2",
    ]
    moments_ok = all(column in moments.columns for column in per_ell_required) and len(moments) == 28
    mean_consistent = False
    std_consistent = False
    if moments_ok and n_summary > 0:
        expected_se_mean = (
            moments["theoretical_Dl_uK2"].to_numpy(dtype=float)
            * np.sqrt(2.0 / moments["degrees_of_freedom"].to_numpy(dtype=float))
            / np.sqrt(n_summary)
        )
        mean_difference = np.abs(
            moments["empirical_mean_Dl_uK2"].to_numpy(dtype=float)
            - moments["theoretical_Dl_uK2"].to_numpy(dtype=float)
        )
        mean_consistent = bool(np.all(mean_difference <= 6.0 * expected_se_mean))

        relative_std_error = np.abs(
            (
                moments["empirical_standard_deviation_uK2"].to_numpy(dtype=float)
                - moments["expected_standard_deviation_uK2"].to_numpy(dtype=float)
            )
            / moments["expected_standard_deviation_uK2"].to_numpy(dtype=float)
        )
        std_threshold = max(0.08, 12.0 / np.sqrt(n_summary))
        std_consistent = bool(np.all(relative_std_error <= std_threshold))
    results.append(
        ValidationResult(
            "per-ell empirical means are consistent with theoretical D_ell",
            mean_consistent,
            "each mean within six expected Monte Carlo standard errors",
        )
    )
    results.append(
        ValidationResult(
            "per-ell empirical standard deviations are reasonably consistent with theory",
            std_consistent,
            "relative standard-deviation errors checked with non-flaky tolerance",
        )
    )

    if required_columns_ok:
        small_ell = comparison["ell"].to_numpy(dtype=int)
        small_theory = comparison["camb_Dl_uK2"].to_numpy(dtype=float)
        sim_a = simulate_lowell_power_ratios(
            small_ell, small_theory, MonteCarloConfig(n_sim=10_000, seed=123456, chunk_size=2_000)
        ).R_low_sim
        sim_b = simulate_lowell_power_ratios(
            small_ell, small_theory, MonteCarloConfig(n_sim=10_000, seed=123456, chunk_size=2_000)
        ).R_low_sim
        sim_c = simulate_lowell_power_ratios(
            small_ell, small_theory, MonteCarloConfig(n_sim=10_000, seed=654321, chunk_size=2_000)
        ).R_low_sim
        sim_d = simulate_lowell_power_ratios(
            small_ell, small_theory, MonteCarloConfig(n_sim=10_000, seed=123456, chunk_size=3_333)
        ).R_low_sim
    else:
        sim_a = sim_b = sim_c = sim_d = np.array([])

    results.append(
        ValidationResult(
            "reproducibility test passes for repeated small simulation",
            np.array_equal(sim_a, sim_b) and len(sim_a) == 10_000,
            "N=10000, seed=123456",
        )
    )
    results.append(
        ValidationResult(
            "different-seed test changes the small simulation",
            len(sim_a) == len(sim_c) and not np.array_equal(sim_a, sim_c),
            "seed=123456 compared with seed=654321",
        )
    )
    results.append(
        ValidationResult(
            "chunking test preserves the generated final R_low array",
            np.array_equal(sim_a, sim_d) and len(sim_a) == 10_000,
            "same seed and N, chunk sizes 2000 and 3333",
        )
    )

    no_cube_saved = npz_keys_ok and r_low_sim.ndim == 1 and r_low_sim.size == n_summary and cutoff_arrays_ok
    results.append(
        ValidationResult(
            "no full simulated-spectrum cube was saved",
            no_cube_saved,
            f"NPZ keys={list(npz.files) if npz is not None else 'none'}",
        )
    )

    figure_paths = [HISTOGRAM_PNG, HISTOGRAM_PDF, CDF_PNG, CDF_PDF]
    missing_figures = [str(path) for path in figure_paths if not file_exists_and_nonempty(path)]
    results.append(
        ValidationResult(
            "all Stage 3 figures exist as PNG and PDF and are nonempty",
            not missing_figures,
            "all figure files present" if not missing_figures else "; ".join(missing_figures),
        )
    )

    return results


def main() -> None:
    """Run validation and exit nonzero if any critical check fails."""

    results = validate_stage3_outputs()
    for result in results:
        print_result(result)

    failed = [result for result in results if result.critical and not result.passed]
    if failed:
        names = ", ".join(result.name for result in failed)
        raise SystemExit(f"Validation failed for critical checks: {names}")

    print("All critical Stage 3 validation checks passed.")


if __name__ == "__main__":
    main()
