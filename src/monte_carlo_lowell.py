"""Stage 3 Monte Carlo test of combined low-multipole TT power.

This script implements a pre-specified ideal full-sky cosmic-variance
simulation. It keeps Planck's reported asymmetric confidence intervals separate
from the simulation model and conditions on the fixed CAMB theoretical spectrum.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_CSV = PROJECT_ROOT / "outputs" / "tables" / "lowell_planck_camb_comparison.csv"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
SIMULATION_DIR = PROJECT_ROOT / "outputs" / "simulations"
TEXT_DIR = PROJECT_ROOT / "outputs" / "text"

SUMMARY_CSV = TABLE_DIR / "monte_carlo_lowell_summary.csv"
SENSITIVITY_CSV = TABLE_DIR / "monte_carlo_upper_cutoff_sensitivity.csv"
QUANTILES_CSV = TABLE_DIR / "monte_carlo_lowell_quantiles.csv"
PER_ELL_MOMENTS_CSV = TABLE_DIR / "monte_carlo_per_ell_moments.csv"
SAMPLES_NPZ = SIMULATION_DIR / "monte_carlo_R_low_samples.npz"
METHODS_MD = TEXT_DIR / "stage3_methods_and_results.md"
HISTOGRAM_PNG = FIGURE_DIR / "monte_carlo_lowell_R_histogram.png"
HISTOGRAM_PDF = FIGURE_DIR / "monte_carlo_lowell_R_histogram.pdf"
CDF_PNG = FIGURE_DIR / "monte_carlo_lowell_R_cdf.png"
CDF_PDF = FIGURE_DIR / "monte_carlo_lowell_R_cdf.pdf"

ELL_MIN = 2
ELL_MAX = 29
UPPER_CUTOFFS = (10, 20, 29)
DEFAULT_N_SIM = 1_000_000
DEFAULT_SEED = 20260701
DEFAULT_CHUNK_SIZE = 50_000
QUANTILE_PROBABILITIES = np.array(
    [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for the Stage 3 simulation."""

    n_sim: int
    seed: int
    chunk_size: int


@dataclass(frozen=True)
class ObservedStatistic:
    """Observed low-ell band-variance statistic."""

    observed_band_variance_uK2: float
    theoretical_band_variance_uK2: float
    observed_R_low: float
    fractional_suppression: float
    percent_suppression: float


@dataclass(frozen=True)
class SimulationResult:
    """Monte Carlo samples and streaming per-ell moments."""

    R_low_sim: np.ndarray
    R_low_by_cutoff: dict[int, np.ndarray]
    empirical_mean_Dl_uK2: np.ndarray
    empirical_standard_deviation_uK2: np.ndarray


@dataclass(frozen=True)
class PValueResult:
    """Lower-tail p-value and finite-simulation uncertainty."""

    lower_tail_count: int
    lower_tail_p_value: float
    monte_carlo_standard_error: float
    p_value_ci_95_lower: float
    p_value_ci_95_upper: float


def parse_arguments() -> MonteCarloConfig:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Run the Stage 3 low-ell ideal full-sky Monte Carlo test."
    )
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    if args.n_sim <= 0:
        raise ValueError("--n-sim must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    return MonteCarloConfig(n_sim=args.n_sim, seed=args.seed, chunk_size=args.chunk_size)


def load_comparison_table(path: Path = COMPARISON_CSV) -> pd.DataFrame:
    """Load the finalized Stage 2 comparison table."""

    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 comparison table: {path}")
    return pd.read_csv(path)


def validate_input_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and return ell = 2 through 29 comparison data."""

    required_columns = ["ell", "planck_Dl_uK2", "camb_Dl_uK2"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Stage 2 comparison table is missing columns: {missing}")

    low_ell = dataframe[required_columns].copy()
    low_ell["ell"] = low_ell["ell"].astype(int)
    low_ell = low_ell.sort_values("ell").reset_index(drop=True)
    expected_ell = np.arange(ELL_MIN, ELL_MAX + 1)
    if len(low_ell) != len(expected_ell) or not np.array_equal(low_ell["ell"].to_numpy(), expected_ell):
        raise ValueError("Stage 2 comparison table must contain exactly ell = 2 through 29.")

    values = low_ell[["planck_Dl_uK2", "camb_Dl_uK2"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Observed and theoretical D_ell values must be finite.")
    if not (low_ell["camb_Dl_uK2"].to_numpy(dtype=float) > 0).all():
        raise ValueError("Theoretical CAMB D_ell values must be strictly positive.")

    return low_ell


def compute_variance_weights(ell: np.ndarray) -> np.ndarray:
    """Compute D_ell weights for band-limited temperature variance."""

    ell_float = ell.astype(float)
    return (2.0 * ell_float + 1.0) / (ell_float * (ell_float + 1.0))


def compute_band_variance(ell: np.ndarray, dl_values: np.ndarray) -> float:
    """Compute V = 1/2 sum[(2ell+1) D_ell / (ell(ell+1))] in microkelvin^2."""

    weights = compute_variance_weights(ell)
    return float(0.5 * np.sum(weights * dl_values))


def compute_observed_statistic(dataframe: pd.DataFrame) -> ObservedStatistic:
    """Calculate the observed primary statistic and variance quantities."""

    ell = dataframe["ell"].to_numpy(dtype=int)
    planck_dl = dataframe["planck_Dl_uK2"].to_numpy(dtype=float)
    camb_dl = dataframe["camb_Dl_uK2"].to_numpy(dtype=float)
    observed_band_variance = compute_band_variance(ell, planck_dl)
    theoretical_band_variance = compute_band_variance(ell, camb_dl)
    observed_R = observed_band_variance / theoretical_band_variance
    fractional_suppression = 1.0 - observed_R
    return ObservedStatistic(
        observed_band_variance_uK2=observed_band_variance,
        theoretical_band_variance_uK2=theoretical_band_variance,
        observed_R_low=float(observed_R),
        fractional_suppression=float(fractional_suppression),
        percent_suppression=float(100.0 * fractional_suppression),
    )


def compute_observed_statistic_for_cutoff(dataframe: pd.DataFrame, upper_cutoff: int) -> ObservedStatistic:
    """Calculate the observed band-power statistic for ell = 2 through L."""

    subset = dataframe.loc[dataframe["ell"] <= upper_cutoff]
    if subset.empty:
        raise ValueError(f"No multipoles are available for upper cutoff L={upper_cutoff}.")
    ell = subset["ell"].to_numpy(dtype=int)
    planck_dl = subset["planck_Dl_uK2"].to_numpy(dtype=float)
    camb_dl = subset["camb_Dl_uK2"].to_numpy(dtype=float)
    observed_band_variance = compute_band_variance(ell, planck_dl)
    theoretical_band_variance = compute_band_variance(ell, camb_dl)
    observed_R = observed_band_variance / theoretical_band_variance
    fractional_suppression = 1.0 - observed_R
    return ObservedStatistic(
        observed_band_variance_uK2=observed_band_variance,
        theoretical_band_variance_uK2=theoretical_band_variance,
        observed_R_low=float(observed_R),
        fractional_suppression=float(fractional_suppression),
        percent_suppression=float(100.0 * fractional_suppression),
    )


def simulate_lowell_power_ratios(
    ell: np.ndarray,
    theory_dl: np.ndarray,
    config: MonteCarloConfig,
    upper_cutoffs: Iterable[int] = UPPER_CUTOFFS,
) -> SimulationResult:
    """Simulate R_low values with chunk-independent random-number ordering."""

    cutoffs = tuple(sorted(set(int(cutoff) for cutoff in upper_cutoffs)))
    invalid = [cutoff for cutoff in cutoffs if cutoff < int(np.min(ell)) or cutoff > int(np.max(ell))]
    if invalid:
        raise ValueError(f"Upper cutoffs must lie within the simulated ell range: {invalid}")

    degrees_of_freedom = 2 * ell + 1
    weights = compute_variance_weights(ell)
    denominators = {
        cutoff: float(np.sum(weights[ell <= cutoff] * theory_dl[ell <= cutoff]))
        for cutoff in cutoffs
    }
    # Use one deterministic RNG stream per multipole. Each stream is consumed in
    # row order, so changing the chunk size changes memory batching but not the
    # final simulated R_low array for the same seed and N.
    seed_sequence = np.random.SeedSequence(config.seed)
    ell_rngs = [np.random.default_rng(child) for child in seed_sequence.spawn(len(ell))]
    n_ell = len(ell)
    r_low_by_cutoff = {
        cutoff: np.zeros(config.n_sim, dtype=np.float64)
        for cutoff in cutoffs
    }
    sum_dl = np.zeros(n_ell, dtype=np.float64)
    sumsq_dl = np.zeros(n_ell, dtype=np.float64)

    for ell_index, rng in enumerate(ell_rngs):
        # Generate one one-dimensional sequence per multipole. The sequence is
        # independent of chunk size, while chunked accumulation keeps memory far
        # below a full (n_simulations x n_multipoles) simulated spectrum cube.
        chi_square = rng.chisquare(df=degrees_of_freedom[ell_index], size=config.n_sim)
        simulated_dl_for_ell = (
            theory_dl[ell_index] * chi_square / degrees_of_freedom[ell_index]
        )
        contributing_cutoffs = [cutoff for cutoff in cutoffs if ell[ell_index] <= cutoff]
        start = 0
        while start < config.n_sim:
            stop = min(start + config.chunk_size, config.n_sim)
            contribution = weights[ell_index] * simulated_dl_for_ell[start:stop]
            for cutoff in contributing_cutoffs:
                r_low_by_cutoff[cutoff][start:stop] += contribution
            start = stop
        sum_dl[ell_index] = simulated_dl_for_ell.sum()
        sumsq_dl[ell_index] = np.square(simulated_dl_for_ell).sum()

    for cutoff in cutoffs:
        r_low_by_cutoff[cutoff] /= denominators[cutoff]

    mean_dl = sum_dl / config.n_sim
    variance_dl = np.maximum((sumsq_dl / config.n_sim) - np.square(mean_dl), 0.0)
    std_dl = np.sqrt(variance_dl)
    primary_cutoff = ELL_MAX if ELL_MAX in r_low_by_cutoff else max(cutoffs)
    return SimulationResult(
        R_low_sim=r_low_by_cutoff[primary_cutoff],
        R_low_by_cutoff=r_low_by_cutoff,
        empirical_mean_Dl_uK2=mean_dl,
        empirical_standard_deviation_uK2=std_dl,
    )


def calculate_p_value(r_low_sim: np.ndarray, observed_R_low: float) -> PValueResult:
    """Calculate lower-tail count, add-one p-value, and Monte Carlo uncertainty."""

    n_sim = len(r_low_sim)
    count = int(np.sum(r_low_sim <= observed_R_low))
    p_lower = (count + 1.0) / (n_sim + 1.0)
    se = float(np.sqrt(p_lower * (1.0 - p_lower) / n_sim))
    ci_lower, ci_upper = calculate_binomial_interval(count=count, n_sim=n_sim)
    return PValueResult(
        lower_tail_count=count,
        lower_tail_p_value=float(p_lower),
        monte_carlo_standard_error=se,
        p_value_ci_95_lower=ci_lower,
        p_value_ci_95_upper=ci_upper,
    )


def calculate_binomial_interval(count: int, n_sim: int) -> tuple[float, float]:
    """Calculate a Wilson 95% interval for finite Monte Carlo p uncertainty."""

    z = 1.959963984540054
    phat = count / n_sim
    denominator = 1.0 + (z**2) / n_sim
    center = (phat + (z**2) / (2.0 * n_sim)) / denominator
    half_width = (
        z
        * np.sqrt((phat * (1.0 - phat) / n_sim) + (z**2) / (4.0 * n_sim**2))
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def calculate_quantiles(r_low_sim: np.ndarray) -> pd.DataFrame:
    """Calculate requested quantiles of the simulated R_low distribution."""

    return pd.DataFrame(
        {
            "probability": QUANTILE_PROBABILITIES,
            "R_low_quantile": np.quantile(r_low_sim, QUANTILE_PROBABILITIES),
        }
    )


def build_per_ell_moments(
    ell: np.ndarray,
    theory_dl: np.ndarray,
    simulation: SimulationResult,
) -> pd.DataFrame:
    """Build per-multipole simulation moment checks."""

    degrees_of_freedom = 2 * ell + 1
    expected_std = theory_dl * np.sqrt(2.0 / degrees_of_freedom)
    empirical_mean = simulation.empirical_mean_Dl_uK2
    empirical_std = simulation.empirical_standard_deviation_uK2
    return pd.DataFrame(
        {
            "ell": ell,
            "degrees_of_freedom": degrees_of_freedom,
            "theoretical_Dl_uK2": theory_dl,
            "empirical_mean_Dl_uK2": empirical_mean,
            "expected_standard_deviation_uK2": expected_std,
            "empirical_standard_deviation_uK2": empirical_std,
            "relative_mean_error": (empirical_mean - theory_dl) / theory_dl,
            "relative_standard_deviation_error": (empirical_std - expected_std) / expected_std,
        }
    )


def build_summary(
    config: MonteCarloConfig,
    observed: ObservedStatistic,
    p_value: PValueResult,
    r_low_sim: np.ndarray,
) -> pd.DataFrame:
    """Create the main summary table as metric/value rows."""

    values: list[tuple[str, float | int]] = [
        ("ell_min", ELL_MIN),
        ("ell_max", ELL_MAX),
        ("n_multipoles", ELL_MAX - ELL_MIN + 1),
        ("n_simulations", config.n_sim),
        ("seed", config.seed),
        ("chunk_size", config.chunk_size),
        ("observed_band_variance_uK2", observed.observed_band_variance_uK2),
        ("theoretical_band_variance_uK2", observed.theoretical_band_variance_uK2),
        ("observed_R_low", observed.observed_R_low),
        ("fractional_suppression", observed.fractional_suppression),
        ("percent_suppression", observed.percent_suppression),
        ("lower_tail_count", p_value.lower_tail_count),
        ("lower_tail_p_value", p_value.lower_tail_p_value),
        ("monte_carlo_standard_error", p_value.monte_carlo_standard_error),
        ("p_value_ci_95_lower", p_value.p_value_ci_95_lower),
        ("p_value_ci_95_upper", p_value.p_value_ci_95_upper),
        ("simulated_R_mean", float(np.mean(r_low_sim))),
        ("simulated_R_standard_deviation", float(np.std(r_low_sim))),
        ("simulated_R_median", float(np.median(r_low_sim))),
        ("simulated_R_minimum", float(np.min(r_low_sim))),
        ("simulated_R_maximum", float(np.max(r_low_sim))),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def save_summary(dataframe: pd.DataFrame, path: Path = SUMMARY_CSV) -> None:
    """Save the main summary table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def build_cutoff_sensitivity_table(
    dataframe: pd.DataFrame,
    simulation: SimulationResult,
) -> pd.DataFrame:
    """Build the predetermined L=10, L=20, and L=29 robustness table."""

    rows: list[dict[str, float | int]] = []
    for cutoff in UPPER_CUTOFFS:
        observed = compute_observed_statistic_for_cutoff(dataframe, cutoff)
        p_value = calculate_p_value(simulation.R_low_by_cutoff[cutoff], observed.observed_R_low)
        rows.append(
            {
                "ell_min": ELL_MIN,
                "ell_max": cutoff,
                "observed_band_variance_uK2": observed.observed_band_variance_uK2,
                "theoretical_band_variance_uK2": observed.theoretical_band_variance_uK2,
                "observed_R_low": observed.observed_R_low,
                "fractional_suppression": observed.fractional_suppression,
                "percent_suppression": observed.percent_suppression,
                "lower_tail_count": p_value.lower_tail_count,
                "lower_tail_p_value": p_value.lower_tail_p_value,
                "monte_carlo_standard_error": p_value.monte_carlo_standard_error,
                "p_value_ci_95_lower": p_value.p_value_ci_95_lower,
                "p_value_ci_95_upper": p_value.p_value_ci_95_upper,
            }
        )
    return pd.DataFrame(rows)


def save_cutoff_sensitivity(dataframe: pd.DataFrame, path: Path = SENSITIVITY_CSV) -> None:
    """Save the upper-multipole cutoff robustness table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_quantiles(dataframe: pd.DataFrame, path: Path = QUANTILES_CSV) -> None:
    """Save simulation quantiles."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_per_ell_moments(dataframe: pd.DataFrame, path: Path = PER_ELL_MOMENTS_CSV) -> None:
    """Save per-multipole moment checks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_simulation_samples(
    r_low_sim: np.ndarray,
    r_low_by_cutoff: dict[int, np.ndarray],
    config: MonteCarloConfig,
    observed: ObservedStatistic,
    path: Path = SAMPLES_NPZ,
) -> None:
    """Save one-dimensional simulated R_low arrays and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "R_low_sim": r_low_sim,
        "n_simulations": np.array(config.n_sim, dtype=np.int64),
        "seed": np.array(config.seed, dtype=np.int64),
        "ell_min": np.array(ELL_MIN, dtype=np.int64),
        "ell_max": np.array(ELL_MAX, dtype=np.int64),
        "observed_R_low": np.array(observed.observed_R_low, dtype=np.float64),
    }
    for cutoff, values in sorted(r_low_by_cutoff.items()):
        arrays[f"R_low_L{cutoff}"] = values
    np.savez_compressed(path, **arrays)


def save_figure(fig: plt.Figure, paths: Iterable[Path]) -> None:
    """Save a figure as PNG/PDF."""

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
    plt.close(fig)


def create_histogram(
    r_low_sim: np.ndarray,
    observed: ObservedStatistic,
    p_value: PValueResult,
    quantiles: pd.DataFrame,
) -> None:
    """Create the simulated R_low histogram."""

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bins = min(120, max(40, int(np.sqrt(len(r_low_sim)))))
    ax.hist(r_low_sim, bins=bins, density=True, color="#4c78a8", alpha=0.72)
    ax.axvline(observed.observed_R_low, color="#d62728", linewidth=2.0, label="Observed R_low")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="R_low = 1")
    fifth = float(quantiles.loc[np.isclose(quantiles["probability"], 0.05), "R_low_quantile"].iloc[0])
    ax.axvline(fifth, color="#2ca02c", linestyle=":", linewidth=1.6, label="5th percentile")
    ax.text(
        0.03,
        0.95,
        f"lower-tail p = {p_value.lower_tail_p_value:.5g}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    ax.set_title("Ideal Full-Sky Lambda-CDM Monte Carlo Distribution of Low-ell R_low")
    ax.set_xlabel("Low-ell band-power ratio R_low")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, [HISTOGRAM_PNG, HISTOGRAM_PDF])


def create_empirical_cdf_plot(
    r_low_sim: np.ndarray,
    observed: ObservedStatistic,
    p_value: PValueResult,
) -> None:
    """Create the empirical CDF plot for R_low."""

    sorted_values = np.sort(r_low_sim)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(sorted_values, cumulative, color="#1f77b4", linewidth=1.7)
    ax.axvline(observed.observed_R_low, color="#d62728", linewidth=2.0, label="Observed R_low")
    ax.axhline(p_value.lower_tail_p_value, color="#d62728", linestyle=":", linewidth=1.5)
    ax.scatter([observed.observed_R_low], [p_value.lower_tail_p_value], color="#d62728", zorder=3)
    ax.text(
        observed.observed_R_low,
        min(p_value.lower_tail_p_value + 0.06, 0.95),
        f"p = {p_value.lower_tail_p_value:.5g}",
        color="#d62728",
        ha="center",
    )
    ax.set_title("Empirical CDF of R_low Under Ideal Full-Sky Lambda-CDM")
    ax.set_xlabel("Low-ell band-power ratio R_low")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, [CDF_PNG, CDF_PDF])


def write_methods_and_results(
    config: MonteCarloConfig,
    observed: ObservedStatistic,
    p_value: PValueResult,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    quantiles: pd.DataFrame,
    path: Path = METHODS_MD,
) -> None:
    """Write a human-readable Stage 3 methods and results summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary_values = dict(zip(summary["metric"], summary["value"]))
    conclusion = make_primary_conclusion(observed, p_value)
    text = f"""# Stage 3 Methods and Results

## Primary Statistic

The pre-specified statistic is:

```text
weight_ell = (2ell + 1) / (ell(ell + 1))

R_low =
    sum(weight_ell * D_ell_observed)
    /
    sum(weight_ell * D_ell_theory)
```

The factor `1/[ell(ell+1)]` is required because the public spectrum is reported
as `D_ell`, while the band-limited temperature variance is defined using
`C_ell`.

## Simulation Equation

For each multipole, the ideal full-sky Gaussian CMB approximation uses:

```text
X_ell ~ chi-square(df = 2ell + 1)
D_ell_sim = D_ell_theory * X_ell / (2ell + 1)
```

The simulation includes ideal cosmic variance only. Planck's asymmetric
reported confidence intervals are not sampled here.

## Configuration

- Number of simulations: {config.n_sim}
- Random seed: {config.seed}
- Chunk size: {config.chunk_size}
- Multipole range: {ELL_MIN} through {ELL_MAX}

## Results

- Observed band-limited variance: {observed.observed_band_variance_uK2:.12g} microK^2
- Theoretical band-limited variance: {observed.theoretical_band_variance_uK2:.12g} microK^2
- Observed R_low: {observed.observed_R_low:.12g}
- Fractional suppression: {observed.fractional_suppression:.12g}
- Percent suppression: {observed.percent_suppression:.6g}%
- Lower-tail count: {p_value.lower_tail_count}
- Add-one-corrected lower-tail p-value: {p_value.lower_tail_p_value:.12g}
- Monte Carlo standard error: {p_value.monte_carlo_standard_error:.12g}
- 95% binomial interval for finite Monte Carlo p-value uncertainty:
  [{p_value.p_value_ci_95_lower:.12g}, {p_value.p_value_ci_95_upper:.12g}]
- Simulated R_low mean: {summary_values["simulated_R_mean"]}
- Simulated R_low standard deviation: {summary_values["simulated_R_standard_deviation"]}
- Simulated R_low median: {summary_values["simulated_R_median"]}

## Selected Quantiles

```text
{quantiles.to_string(index=False)}
```

## Upper-Multipole Cutoff Sensitivity

```text
{sensitivity.to_string(index=False)}
```

## Primary Conclusion

{conclusion}

## Assumptions and Limitations

1. The simulation includes ideal cosmic variance only.
2. Planck's asymmetric reported confidence intervals are not sampled here.
3. The simulation assumes full-sky coverage and independent multipoles.
4. Real Planck analysis includes masking, foreground treatment, anisotropic
   noise, likelihood construction, and mode coupling.
5. The CAMB parameters are held fixed rather than refit for each simulation.
6. The parameters were inferred using Planck-related observations, so this is a
   conditional descriptive consistency test rather than a fully independent
   hypothesis test.
7. The public Planck spectrum points are used as observed band-power estimates,
   not as a replacement for the official low-ell likelihood.
8. A small p-value would indicate that the observed combined power is unusual
   under this simplified model, but would not by itself prove new physics.
9. A moderate p-value would indicate that cosmic variance can reasonably account
   for the observed suppression under this simplified model.

This result is not the official Planck low-ell likelihood and is not an official
Planck anomaly significance.
"""
    path.write_text(text, encoding="utf-8")


def make_primary_conclusion(observed: ObservedStatistic, p_value: PValueResult) -> str:
    """Generate the primary conclusion after the result has been calculated."""

    return (
        f"The observed low-ell band-power ratio is R_low = {observed.observed_R_low:.6g}, "
        f"corresponding to a {observed.percent_suppression:.3g}% suppression relative to "
        "the fixed CAMB theory over ell = 2 through 29. In the ideal full-sky "
        f"cosmic-variance simulation, {p_value.lower_tail_count} simulated skies had "
        "R_low less than or equal to the observed value, giving an add-one-corrected "
        f"lower-tail p-value of {p_value.lower_tail_p_value:.6g}. This is a "
        "conditional descriptive consistency test under a simplified model, not proof "
        "or disproof of Lambda-CDM and not an official Planck anomaly significance."
    )


def main() -> None:
    """Run the full Stage 3 Monte Carlo workflow."""

    config = parse_arguments()
    comparison = validate_input_table(load_comparison_table())
    ell = comparison["ell"].to_numpy(dtype=int)
    theory_dl = comparison["camb_Dl_uK2"].to_numpy(dtype=float)

    observed = compute_observed_statistic(comparison)
    simulation = simulate_lowell_power_ratios(ell, theory_dl, config)
    p_value = calculate_p_value(simulation.R_low_sim, observed.observed_R_low)
    quantiles = calculate_quantiles(simulation.R_low_sim)
    moments = build_per_ell_moments(ell, theory_dl, simulation)
    summary = build_summary(config, observed, p_value, simulation.R_low_sim)
    sensitivity = build_cutoff_sensitivity_table(comparison, simulation)

    save_summary(summary)
    save_cutoff_sensitivity(sensitivity)
    save_quantiles(quantiles)
    save_per_ell_moments(moments)
    save_simulation_samples(simulation.R_low_sim, simulation.R_low_by_cutoff, config, observed)
    create_histogram(simulation.R_low_sim, observed, p_value, quantiles)
    create_empirical_cdf_plot(simulation.R_low_sim, observed, p_value)
    write_methods_and_results(config, observed, p_value, summary, sensitivity, quantiles)

    print("\nStage 3 Monte Carlo low-ell analysis complete")
    print(f"n_simulations: {config.n_sim}")
    print(f"seed: {config.seed}")
    print(f"chunk_size: {config.chunk_size}")
    print(f"observed_band_variance_uK2: {observed.observed_band_variance_uK2:.12g}")
    print(f"theoretical_band_variance_uK2: {observed.theoretical_band_variance_uK2:.12g}")
    print(f"observed_R_low: {observed.observed_R_low:.12g}")
    print(f"fractional_suppression: {observed.fractional_suppression:.12g}")
    print(f"percent_suppression: {observed.percent_suppression:.6g}")
    print(f"lower_tail_count: {p_value.lower_tail_count}")
    print(f"lower_tail_p_value: {p_value.lower_tail_p_value:.12g}")
    print(f"monte_carlo_standard_error: {p_value.monte_carlo_standard_error:.12g}")
    print(
        "p_value_ci_95: "
        f"[{p_value.p_value_ci_95_lower:.12g}, {p_value.p_value_ci_95_upper:.12g}]"
    )
    print("Output files:")
    for path in [
        SUMMARY_CSV,
        SENSITIVITY_CSV,
        QUANTILES_CSV,
        PER_ELL_MOMENTS_CSV,
        SAMPLES_NPZ,
        METHODS_MD,
        HISTOGRAM_PNG,
        HISTOGRAM_PDF,
        CDF_PNG,
        CDF_PDF,
    ]:
        print(f"  - {path}")
    print(make_primary_conclusion(observed, p_value))


if __name__ == "__main__":
    main()
