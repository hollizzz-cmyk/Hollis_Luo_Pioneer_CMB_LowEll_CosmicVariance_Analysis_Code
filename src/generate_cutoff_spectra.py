"""Generate CAMB TT spectra for pre-specified primordial cutoff models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import camb
import numpy as np
import pandas as pd

from cutoff_models import (
    AS,
    NS,
    PIVOT_SCALAR,
    SHARP_APPROXIMATION_ALPHA,
    CutoffFamily,
    CutoffModel,
    modified_primordial_power,
    standard_primordial_power,
)
from generate_camb_spectrum import CosmologyConfig, calculate_tt_spectrum, create_camb_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
SPECTRA_DIR = PROJECT_ROOT / "outputs" / "cutoff_spectra"
STANDARD_CAMB_CSV = TABLE_DIR / "camb_planck2018_tt_spectrum.csv"
INDEX_CSV = SPECTRA_DIR / "cutoff_spectrum_index.csv"
RUN_METADATA_JSON = SPECTRA_DIR / "cutoff_generation_metadata.json"
BASELINE_REPRODUCTION_CSV = TABLE_DIR / "cutoff_baseline_reproduction.csv"

ELL_MAX = 2500
KMIN = 1e-6
KMAX = 10.0
N_MIN = 1000
RTOL = 1e-6
KC_VALUES = np.logspace(np.log10(5e-5), np.log10(2e-3), 31)
ALPHA_VALUES = np.array([1.0, 2.0, 4.0, 8.0])
QUICK_KC_VALUES = np.array([KC_VALUES[0], KC_VALUES[len(KC_VALUES) // 2], KC_VALUES[-1]])
QUICK_ALPHA_VALUES = np.array([1.0, 4.0])
SHARP_CONVERGENCE_CSV = TABLE_DIR / "cutoff_sharp_convergence.csv"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Generate primordial-cutoff CAMB spectra.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="Run a small representative subset.")
    mode.add_argument("--full", action="store_true", help="Run the full pre-specified grid.")
    parser.add_argument(
        "--no-pivot-normalization",
        action="store_true",
        help="Use P0(k) * F(k/kc) without smooth-model pivot normalization.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reuse valid cached spectra when present (default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate spectra even if valid cache files exist.",
    )
    return parser.parse_args()


def model_to_metadata(model: CutoffModel, runtime_seconds: float) -> dict[str, object]:
    """Return serializable metadata for one generated spectrum."""

    return {
        "model_id": model.model_id,
        "model_family": model.family.value,
        "kc_Mpc_inverse": model.kc_mpc_inverse,
        "alpha": model.alpha,
        "pivot_normalized": model.pivot_normalized,
        "sharp_implementation": model.sharp_implementation,
        "sharp_approximation_alpha": (
            SHARP_APPROXIMATION_ALPHA
            if model.sharp_implementation == "numerical_sharp_step_approximation"
            else None
        ),
        "n_extra_parameters": model.n_extra_parameters,
        "ell_max": ELL_MAX,
        "kmin_Mpc_inverse": KMIN,
        "kmax_Mpc_inverse": KMAX,
        "N_min": N_MIN,
        "rtol": RTOL,
        "As": AS,
        "ns": NS,
        "pivot_scalar_Mpc_inverse": PIVOT_SCALAR,
        "camb_version": camb.__version__,
        "runtime_seconds": runtime_seconds,
    }


def spectrum_paths(model: CutoffModel) -> tuple[Path, Path]:
    """Return CSV and metadata paths for a model."""

    safe_id = model.model_id.replace("+", "p").replace("-", "m")
    return SPECTRA_DIR / f"{safe_id}.csv", SPECTRA_DIR / f"{safe_id}.json"


def is_valid_cached_spectrum(model: CutoffModel, csv_path: Path, metadata_path: Path) -> bool:
    """Check whether a cached spectrum matches this run's settings."""

    if not csv_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataframe = pd.read_csv(csv_path, usecols=["ell", "Cl_TT_uK2", "Dl_TT_uK2"])
    except Exception:
        return False
    expected = model_to_metadata(model, runtime_seconds=float(metadata.get("runtime_seconds", 0.0)))
    for key in [
        "model_id",
        "model_family",
        "kc_Mpc_inverse",
        "alpha",
        "pivot_normalized",
        "sharp_implementation",
        "sharp_approximation_alpha",
        "ell_max",
        "As",
        "ns",
        "pivot_scalar_Mpc_inverse",
        "camb_version",
    ]:
        if metadata.get(key) != expected.get(key):
            return False
    ell = dataframe["ell"].to_numpy(dtype=int)
    values = dataframe[["Cl_TT_uK2", "Dl_TT_uK2"]].to_numpy(dtype=float)
    return (
        len(dataframe) == ELL_MAX - 1
        and np.array_equal(ell, np.arange(2, ELL_MAX + 1))
        and np.isfinite(values).all()
        and bool((dataframe["Dl_TT_uK2"] > 0.0).all())
    )


def create_custom_power_parameters(model: CutoffModel, config: CosmologyConfig) -> camb.CAMBparams:
    """Create CAMB parameters using a custom primordial scalar spectrum."""

    pars = create_camb_parameters(config)
    if model.family in {CutoffFamily.STANDARD, CutoffFamily.EXPONENTIAL, CutoffFamily.RATIONAL}:
        pars.set_initial_power_function(
            lambda k: modified_primordial_power(k, model),
            kmin=KMIN,
            kmax=KMAX,
            N_min=N_MIN,
            rtol=RTOL,
            effective_ns_for_nonlinear=NS,
        )
    elif model.family == CutoffFamily.SHARP:
        if model.sharp_implementation == "numerical_sharp_step_approximation":
            pars.set_initial_power_function(
                lambda k: modified_primordial_power(k, model),
                kmin=KMIN,
                kmax=KMAX,
                N_min=max(N_MIN, 2000),
                rtol=RTOL,
                effective_ns_for_nonlinear=NS,
            )
        else:
            k_values, p_values = build_sharp_power_table(model, density=9000)
            pars.set_initial_power_table(k_values, p_values, effective_ns_for_nonlinear=NS)
    else:
        raise ValueError(f"Unsupported model family: {model.family}")
    pars.WantTensors = False
    return pars


def build_sharp_power_table(model: CutoffModel, density: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a dense tabulated sharp-step primordial spectrum."""

    if model.kc_mpc_inverse is None:
        raise ValueError("sharp cutoff requires kc_mpc_inverse.")
    base = np.logspace(np.log10(KMIN), np.log10(KMAX), density)
    eps = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
    around = np.concatenate(
        [
            model.kc_mpc_inverse * (1.0 - eps),
            np.array([model.kc_mpc_inverse]),
            model.kc_mpc_inverse * (1.0 + eps),
        ]
    )
    k_values = np.unique(np.concatenate([base, around]))
    k_values = k_values[(k_values >= KMIN) & (k_values <= KMAX)]
    k_values.sort()
    p_values = modified_primordial_power(k_values, model)
    if not np.all(np.isfinite(p_values)) or np.any(p_values < 0.0):
        raise ValueError("sharp power table contains non-finite or negative values.")
    return k_values, p_values


def calculate_model_spectrum(model: CutoffModel) -> pd.DataFrame:
    """Calculate a model's CAMB TT spectrum."""

    config = CosmologyConfig()
    pars = create_custom_power_parameters(model, config)
    ell, cl_tt, dl_tt = calculate_tt_spectrum(pars, config.ell_max)
    dataframe = pd.DataFrame(
        {
            "ell": ell[ell >= 2],
            "Cl_TT_uK2": cl_tt[ell >= 2],
            "Dl_TT_uK2": dl_tt[ell >= 2],
        }
    )
    if not np.isfinite(dataframe[["Cl_TT_uK2", "Dl_TT_uK2"]].to_numpy(dtype=float)).all():
        raise ValueError(f"CAMB produced non-finite values for {model.model_id}.")
    if not bool((dataframe["Dl_TT_uK2"] > 0.0).all()):
        raise ValueError(f"CAMB produced non-positive D_ell values for {model.model_id}.")
    return dataframe


def load_or_generate_spectrum(model: CutoffModel, force: bool) -> dict[str, object]:
    """Load a valid cached spectrum or generate it with CAMB."""

    csv_path, metadata_path = spectrum_paths(model)
    if not force and is_valid_cached_spectrum(model, csv_path, metadata_path):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["spectrum_csv"] = str(csv_path)
        metadata["from_cache"] = True
        return metadata

    start = time.perf_counter()
    dataframe = calculate_model_spectrum(model)
    runtime = time.perf_counter() - start
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    metadata = model_to_metadata(model, runtime)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["spectrum_csv"] = str(csv_path)
    metadata["from_cache"] = False
    return metadata


def build_model_grid(full: bool, pivot_normalized: bool, sharp_implementation: str) -> list[CutoffModel]:
    """Return the requested pre-specified model grid."""

    kc_values = KC_VALUES if full else QUICK_KC_VALUES
    alpha_values = ALPHA_VALUES if full else QUICK_ALPHA_VALUES
    models: list[CutoffModel] = [
        CutoffModel(CutoffFamily.STANDARD, pivot_normalized=pivot_normalized)
    ]
    for family in [CutoffFamily.EXPONENTIAL, CutoffFamily.RATIONAL]:
        for kc in kc_values:
            for alpha in alpha_values:
                models.append(
                    CutoffModel(
                        family=family,
                        kc_mpc_inverse=float(kc),
                        alpha=float(alpha),
                        pivot_normalized=pivot_normalized,
                    )
                )
    for kc in kc_values:
        models.append(
            CutoffModel(
                family=CutoffFamily.SHARP,
                kc_mpc_inverse=float(kc),
                pivot_normalized=pivot_normalized,
                sharp_implementation=sharp_implementation,
            )
        )
    return models


def run_baseline_reproduction_test(force: bool) -> dict[str, float]:
    """Require custom P(k)=P0(k) to reproduce the existing standard CAMB table."""

    if not STANDARD_CAMB_CSV.exists():
        raise FileNotFoundError(f"Missing existing standard CAMB spectrum: {STANDARD_CAMB_CSV}")
    model = CutoffModel(CutoffFamily.STANDARD)
    metadata = load_or_generate_spectrum(model, force=force)
    custom = pd.read_csv(metadata["spectrum_csv"])
    standard = pd.read_csv(STANDARD_CAMB_CSV)
    merged = pd.merge(
        standard[["ell", "Dl_TT_uK2"]],
        custom[["ell", "Dl_TT_uK2"]],
        on="ell",
        suffixes=("_standard", "_custom"),
        validate="one_to_one",
    )
    low = merged[(merged["ell"] >= 2) & (merged["ell"] <= 29)]
    high = merged[(merged["ell"] >= 30) & (merged["ell"] <= ELL_MAX)]
    low_rel = np.abs(low["Dl_TT_uK2_custom"] - low["Dl_TT_uK2_standard"]) / low["Dl_TT_uK2_standard"]
    high_rel = (
        np.abs(high["Dl_TT_uK2_custom"] - high["Dl_TT_uK2_standard"])
        / high["Dl_TT_uK2_standard"]
    )
    result = {
        "max_relative_difference_ell_2_29": float(low_rel.max()),
        "max_relative_difference_ell_30_2500": float(high_rel.max()),
        "tolerance": 1e-4,
    }
    pd.DataFrame([result]).to_csv(BASELINE_REPRODUCTION_CSV, index=False)
    if result["max_relative_difference_ell_2_29"] >= 1e-4 or result["max_relative_difference_ell_30_2500"] >= 1e-4:
        raise RuntimeError(
            "Custom primordial-power baseline failed to reproduce existing standard CAMB output: "
            f"{result}"
        )
    return result


def evaluate_sharp_convergence(full: bool, pivot_normalized: bool) -> tuple[str, pd.DataFrame]:
    """Test exact sharp-step convergence and decide whether approximation is needed."""

    selected = [float(KC_VALUES[0]), float(KC_VALUES[len(KC_VALUES) // 2]), float(KC_VALUES[-1])] if full else [float(QUICK_KC_VALUES[1])]
    rows: list[dict[str, object]] = []
    exact_passes = True
    for kc in selected:
        model = CutoffModel(
            CutoffFamily.SHARP,
            kc_mpc_inverse=kc,
            pivot_normalized=pivot_normalized,
            sharp_implementation="exact_table",
        )
        try:
            low_k, low_p = build_sharp_power_table(model, density=5000)
            high_k, high_p = build_sharp_power_table(model, density=12000)
            config = CosmologyConfig()
            pars_low = create_camb_parameters(config)
            pars_low.set_initial_power_table(low_k, low_p, effective_ns_for_nonlinear=NS)
            ell_low, _, dl_low = calculate_tt_spectrum(pars_low, 100)
            pars_high = create_camb_parameters(config)
            pars_high.set_initial_power_table(high_k, high_p, effective_ns_for_nonlinear=NS)
            ell_high, _, dl_high = calculate_tt_spectrum(pars_high, 100)
            mask = (ell_low >= 2) & (ell_low <= 100)
            rel = np.abs(dl_low[mask] - dl_high[mask]) / dl_high[mask]
            max_rel = float(np.max(rel))
            passed = bool(max_rel < 1e-3 and np.all(np.isfinite(dl_low[mask])) and np.all(dl_low[mask] > 0.0))
            detail = "exact tabulated step"
        except Exception as exc:
            max_rel = float("nan")
            passed = False
            detail = f"exact tabulated step failed: {exc}"
        rows.append(
            {
                "kc_Mpc_inverse": kc,
                "implementation_tested": "exact_table",
                "max_relative_Dl_difference_ell_2_100": max_rel,
                "passed": passed,
                "detail": detail,
            }
        )
        exact_passes = exact_passes and passed

    implementation = "exact_table" if exact_passes else "numerical_sharp_step_approximation"
    convergence = pd.DataFrame(rows)
    convergence["selected_sharp_implementation"] = implementation
    SHARP_CONVERGENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(SHARP_CONVERGENCE_CSV, index=False)
    return implementation, convergence


def write_index(rows: list[dict[str, object]]) -> None:
    """Write the generated spectrum index."""

    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(INDEX_CSV, index=False)


def main() -> None:
    """Run cutoff spectrum generation."""

    args = parse_arguments()
    total_start = time.perf_counter()
    mode = "full" if args.full else "quick"
    pivot_normalized = not args.no_pivot_normalization
    force = bool(args.force)

    baseline = run_baseline_reproduction_test(force=force)
    sharp_implementation, convergence = evaluate_sharp_convergence(
        full=args.full,
        pivot_normalized=pivot_normalized,
    )
    models = build_model_grid(args.full, pivot_normalized, sharp_implementation)

    rows: list[dict[str, object]] = []
    for index, model in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] {model.model_id}")
        metadata = load_or_generate_spectrum(model, force=force)
        metadata["scan_mode"] = mode
        rows.append(metadata)
        write_index(rows)

    total_runtime = time.perf_counter() - total_start
    run_metadata = {
        "scan_mode": mode,
        "pivot_normalized": pivot_normalized,
        "no_pivot_normalization": args.no_pivot_normalization,
        "model_count": len(models),
        "smooth_alpha_values": (ALPHA_VALUES if args.full else QUICK_ALPHA_VALUES).tolist(),
        "kc_values_Mpc_inverse": (KC_VALUES if args.full else QUICK_KC_VALUES).tolist(),
        "sharp_implementation": sharp_implementation,
        "sharp_approximation_alpha": (
            SHARP_APPROXIMATION_ALPHA
            if sharp_implementation == "numerical_sharp_step_approximation"
            else None
        ),
        "baseline_reproduction": baseline,
        "sharp_convergence_rows": convergence.to_dict(orient="records"),
        "camb_version": camb.__version__,
        "total_runtime_seconds": total_runtime,
    }
    RUN_METADATA_JSON.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("\nCutoff CAMB spectrum generation complete")
    print(f"scan_mode: {mode}")
    print(f"pivot_normalized: {pivot_normalized}")
    print(f"models generated or cached: {len(models)}")
    print(f"sharp_implementation: {sharp_implementation}")
    print(f"baseline max rel diff ell 2-29: {baseline['max_relative_difference_ell_2_29']:.3e}")
    print(f"baseline max rel diff ell 30-2500: {baseline['max_relative_difference_ell_30_2500']:.3e}")
    print(f"CAMB version: {camb.__version__}")
    print(f"total_runtime_seconds: {total_runtime:.3f}")
    print("Output files:")
    for path in [INDEX_CSV, RUN_METADATA_JSON, BASELINE_REPRODUCTION_CSV, SHARP_CONVERGENCE_CSV]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
