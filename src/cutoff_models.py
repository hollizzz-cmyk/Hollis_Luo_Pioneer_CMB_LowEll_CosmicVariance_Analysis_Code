"""Primordial cutoff models for low-k CMB spectrum tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


AS = 2.1005e-9
NS = 0.96605
PIVOT_SCALAR = 0.05
SHARP_APPROXIMATION_ALPHA = 50.0


class CutoffFamily(str, Enum):
    """Supported primordial cutoff model families."""

    STANDARD = "standard"
    EXPONENTIAL = "exponential"
    RATIONAL = "rational"
    SHARP = "sharp"


@dataclass(frozen=True)
class CutoffModel:
    """Identifier for one tested primordial spectrum."""

    family: CutoffFamily
    kc_mpc_inverse: float | None = None
    alpha: float | None = None
    pivot_normalized: bool = True
    sharp_implementation: str = "not_applicable"

    @property
    def n_extra_parameters(self) -> int:
        if self.family == CutoffFamily.STANDARD:
            return 0
        if self.family == CutoffFamily.SHARP:
            return 1
        return 2

    @property
    def model_id(self) -> str:
        if self.family == CutoffFamily.STANDARD:
            return "standard_custom_power"
        if self.family == CutoffFamily.SHARP:
            return f"sharp_kc_{self.kc_mpc_inverse:.10e}_{self._norm_tag()}"
        return (
            f"{self.family.value}_kc_{self.kc_mpc_inverse:.10e}_"
            f"alpha_{self.alpha:.6g}_{self._norm_tag()}"
        )

    def _norm_tag(self) -> str:
        return "pivotnorm" if self.pivot_normalized else "unnormalized"


def as_array(k: np.ndarray | float) -> np.ndarray:
    """Return k as a floating NumPy array."""

    values = np.asarray(k, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("k values must be finite.")
    if np.any(values <= 0.0):
        raise ValueError("k values must be positive.")
    return values


def validate_kc_alpha(kc_mpc_inverse: float, alpha: float | None = None) -> None:
    """Validate cutoff parameters."""

    if not np.isfinite(kc_mpc_inverse) or kc_mpc_inverse <= 0.0:
        raise ValueError("kc_mpc_inverse must be finite and positive.")
    if alpha is not None and (not np.isfinite(alpha) or alpha <= 0.0):
        raise ValueError("alpha must be finite and positive.")


def standard_primordial_power(
    k: np.ndarray | float,
    scalar_amplitude: float = AS,
    scalar_spectral_index: float = NS,
    pivot_scalar: float = PIVOT_SCALAR,
) -> np.ndarray:
    """Return P0(k) = As * (k / k_pivot) ** (ns - 1)."""

    kvals = as_array(k)
    power = scalar_amplitude * np.power(kvals / pivot_scalar, scalar_spectral_index - 1.0)
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("standard primordial spectrum is not finite and nonnegative.")
    return power


def _safe_x_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute x**alpha without overflowing double precision."""

    log_value = alpha * np.log(x)
    return np.exp(np.clip(log_value, -745.0, 709.0))


def exponential_cutoff(x: np.ndarray | float, alpha: float) -> np.ndarray:
    """Return F_exp(x, alpha) = 1 - exp(-x**alpha) stably."""

    xvals = as_array(x)
    validate_kc_alpha(1.0, alpha)
    xa = _safe_x_alpha(xvals, alpha)
    values = -np.expm1(-xa)
    return np.clip(values, 0.0, 1.0)


def rational_cutoff(x: np.ndarray | float, alpha: float) -> np.ndarray:
    """Return F_rat(x, alpha) = x**alpha / (1 + x**alpha) stably."""

    xvals = as_array(x)
    validate_kc_alpha(1.0, alpha)
    log_xa = alpha * np.log(xvals)
    values = np.empty_like(log_xa, dtype=float)
    positive = log_xa >= 0.0
    values[positive] = 1.0 / (1.0 + np.exp(-log_xa[positive]))
    exp_log = np.exp(log_xa[~positive])
    values[~positive] = exp_log / (1.0 + exp_log)
    return np.clip(values, 0.0, 1.0)


def sharp_cutoff(x: np.ndarray | float) -> np.ndarray:
    """Return an exact Heaviside-style cutoff, zero below x=1 and one above."""

    xvals = as_array(x)
    return (xvals >= 1.0).astype(float)


def numerical_sharp_step_approximation(
    x: np.ndarray | float, alpha: float = SHARP_APPROXIMATION_ALPHA
) -> np.ndarray:
    """Return a very sharp smooth approximation to the step cutoff."""

    return rational_cutoff(x, alpha)


def cutoff_function(
    family: CutoffFamily,
    x: np.ndarray | float,
    alpha: float | None = None,
    sharp_implementation: str = "exact_table",
) -> np.ndarray:
    """Evaluate an unnormalized cutoff function."""

    if family == CutoffFamily.EXPONENTIAL:
        if alpha is None:
            raise ValueError("exponential cutoff requires alpha.")
        return exponential_cutoff(x, alpha)
    if family == CutoffFamily.RATIONAL:
        if alpha is None:
            raise ValueError("rational cutoff requires alpha.")
        return rational_cutoff(x, alpha)
    if family == CutoffFamily.SHARP:
        if sharp_implementation == "numerical_sharp_step_approximation":
            return numerical_sharp_step_approximation(x, alpha=SHARP_APPROXIMATION_ALPHA)
        return sharp_cutoff(x)
    if family == CutoffFamily.STANDARD:
        return np.ones_like(as_array(x), dtype=float)
    raise ValueError(f"Unknown cutoff family: {family}")


def modified_primordial_power(
    k: np.ndarray | float,
    model: CutoffModel,
    scalar_amplitude: float = AS,
    scalar_spectral_index: float = NS,
    pivot_scalar: float = PIVOT_SCALAR,
) -> np.ndarray:
    """Return the cutoff-modified primordial curvature spectrum."""

    kvals = as_array(k)
    p0 = standard_primordial_power(
        kvals,
        scalar_amplitude=scalar_amplitude,
        scalar_spectral_index=scalar_spectral_index,
        pivot_scalar=pivot_scalar,
    )
    if model.family == CutoffFamily.STANDARD:
        return p0
    if model.kc_mpc_inverse is None:
        raise ValueError("cutoff models require kc_mpc_inverse.")
    validate_kc_alpha(model.kc_mpc_inverse, model.alpha)
    x = kvals / model.kc_mpc_inverse
    f_values = cutoff_function(
        model.family,
        x,
        alpha=model.alpha,
        sharp_implementation=model.sharp_implementation,
    )
    if model.pivot_normalized and model.family in {CutoffFamily.EXPONENTIAL, CutoffFamily.RATIONAL}:
        f_pivot = float(cutoff_function(model.family, pivot_scalar / model.kc_mpc_inverse, model.alpha))
        if not np.isfinite(f_pivot) or f_pivot <= 0.0:
            raise ValueError("pivot normalization is not finite and positive.")
        f_values = f_values / f_pivot
    power = p0 * f_values
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("modified primordial spectrum is not finite and nonnegative.")
    return power


def model_cutoff_values(k: np.ndarray | float, model: CutoffModel) -> np.ndarray:
    """Return the unnormalized F(k/kc) values for a model."""

    kvals = as_array(k)
    if model.family == CutoffFamily.STANDARD:
        return np.ones_like(kvals, dtype=float)
    if model.kc_mpc_inverse is None:
        raise ValueError("cutoff models require kc_mpc_inverse.")
    return cutoff_function(
        model.family,
        kvals / model.kc_mpc_inverse,
        alpha=model.alpha,
        sharp_implementation=model.sharp_implementation,
    )
