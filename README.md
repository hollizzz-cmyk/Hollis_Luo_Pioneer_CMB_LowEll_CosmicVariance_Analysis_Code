# Pioneer CMB Project

This project supports a cosmology research paper titled
"Investigating Low-ell Power Suppression in the Planck 2018 CMB Temperature
Spectrum Using Cosmic-Variance Analysis."

This first stage contains theoretical predictions only. It does not include
real Planck observational data, residuals, fractional differences, z-scores,
p-values, chi-square tests, significance analysis, or Monte Carlo simulation.
Those steps belong to a later stage.

## What This Stage Does

The project uses CAMB as the Boltzmann solver. Python does not recreate the
Einstein-Boltzmann equations directly; it configures CAMB, runs the solver, and
processes the resulting CMB temperature spectrum.

The current calculation uses the lensed scalar TT spectrum for an approximate
Planck 2018 best-fit spatially flat Lambda-CDM cosmology.

## Project Structure

```text
Pioneer_CMB_Project/
|-- data/
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- src/
|   |-- generate_camb_spectrum.py
|   `-- validate_camb_output.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

The `data` directory is intentionally empty in this stage. It will be used later
for real Planck observational data.

## Windows Setup

This computer uses `python`, not the `py` launcher.

```powershell
cd Pioneer_CMB_Project
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To verify CAMB manually:

```powershell
python -c "import camb; print(camb.__version__)"
```

## Run the Spectrum Generation

```powershell
python src\generate_camb_spectrum.py
```

This creates:

```text
outputs/tables/camb_planck2018_tt_spectrum.csv
outputs/figures/camb_planck2018_tt_full.png
outputs/figures/camb_planck2018_tt_full.pdf
outputs/figures/camb_planck2018_tt_lowell.png
outputs/figures/camb_planck2018_tt_lowell.pdf
```

## Run Validation

```powershell
python src\validate_camb_output.py
```

The validation script checks the table structure, ell range, finite values,
nonnegative TT power, the Cl-to-Dl conversion, low-ell coverage, and simple
acoustic-peak behavior.

## Cl and Dl

`C_ell` is the angular power spectrum coefficient. For temperature, this project
stores it in microkelvin squared.

`D_ell` is the commonly plotted rescaled spectrum:

```text
D_ell = ell * (ell + 1) * C_ell / (2 * pi)
```

The generation script calculates the CAMB cosmology once, then retrieves the
lensed scalar spectrum in both conventions: `raw_cl=True` for `C_ell` and
`raw_cl=False` for `D_ell`. This avoids accidentally converting `D_ell` twice
and avoids an unnecessary second CAMB spectrum calculation.

## Why ell = 0 and ell = 1 Are Excluded

The monopole, `ell = 0`, is the average CMB temperature. The dipole, `ell = 1`,
is dominated by our motion relative to the CMB rest frame. They are not part of
the standard cosmological anisotropy comparison, so the main scientific table
starts at `ell = 2`.

## Cosmological Parameters

Approximate Planck 2018 best-fit flat Lambda-CDM parameters:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `H0` | 67.32 km/s/Mpc | Hubble constant today |
| `ombh2` | 0.02238 | Physical baryon density, Omega_b h^2 |
| `omch2` | 0.12011 | Physical cold dark matter density, Omega_c h^2 |
| `tau` | 0.0543 | Optical depth to reionization |
| `As` | 2.1005e-9 | Scalar primordial curvature amplitude |
| `ns` | 0.96605 | Scalar spectral index |
| `mnu` | 0.06 eV | Sum of neutrino masses |
| `omk` | 0 | Spatial curvature density parameter |
| `pivot_scalar` | 0.05 Mpc^-1 | CAMB scalar primordial-power pivot scale |

`As` is passed directly to CAMB as `As`; it is not
`ln(10^10 As)`.

## Stage 2: Planck 2018 Low-ell Comparison

Stage 2 imports the official Planck Public Data Release 3 unbinned TT spectrum
and compares its low-multipole values with the Stage 1 CAMB theoretical lensed
scalar TT output.

The official file used is:

```text
COM_PowerSpect_CMB-TT-full_R3.01.txt
```

Source:

```text
https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt
```

The original columns are:

```text
l    Dl    -dDl    +dDl
```

These mean:

| Column | Meaning |
| --- | --- |
| `l` | Multipole, ell |
| `Dl` | Observed TT `D_ell` in microkelvin squared |
| `-dDl` | Positive magnitude of the lower asymmetric 68% observational uncertainty |
| `+dDl` | Positive magnitude of the upper asymmetric 68% observational uncertainty |

The public Planck spectrum reports `D_ell` directly in microkelvin squared, so
the Stage 2 comparison uses the CAMB `Dl_TT_uK2` column. It does not convert
Planck values to `C_ell` for the main comparison.

Only the low-multipole rows `ell = 2` through `ell = 29` are used in Stage 2.
The full raw file is saved unchanged in `data/raw/`, and the cleaned low-ell
subset is saved in `data/processed/`.

### Stage 2 Commands

```powershell
.venv\Scripts\activate
python src\download_planck_data.py
python src\analyze_low_ell.py
python src\validate_low_ell_analysis.py
```

Use this command only when the official file should be downloaded again:

```powershell
python src\download_planck_data.py --force
```

### Stage 2 Output Files

```text
data/raw/COM_PowerSpect_CMB-TT-full_R3.01.txt
data/processed/planck2018_tt_lowell.csv
outputs/tables/lowell_planck_camb_comparison.csv
outputs/figures/planck_camb_lowell_comparison.png
outputs/figures/planck_camb_lowell_comparison.pdf
outputs/figures/planck_camb_lowell_residuals.png
outputs/figures/planck_camb_lowell_residuals.pdf
outputs/figures/planck_camb_lowell_fractional_difference.png
outputs/figures/planck_camb_lowell_fractional_difference.pdf
outputs/figures/planck_camb_lowell_normalized_residuals.png
outputs/figures/planck_camb_lowell_normalized_residuals.pdf
```

### Stage 2 Equations

The merged comparison table uses:

```text
residual_Dl_uK2 = planck_Dl_uK2 - camb_Dl_uK2
```

```text
fractional_difference = residual_Dl_uK2 / camb_Dl_uK2
```

```text
percent_difference = 100 * fractional_difference
```

```text
cosmic_variance_fraction = sqrt(2 / (2 * ell + 1))
```

```text
cosmic_variance_sigma_uK2 = camb_Dl_uK2 * cosmic_variance_fraction
```

```text
normalized_residual_cv = residual_Dl_uK2 / cosmic_variance_sigma_uK2
```

```text
camb_cv_lower_uK2 = camb_Dl_uK2 - cosmic_variance_sigma_uK2
```

```text
camb_cv_upper_uK2 = camb_Dl_uK2 + cosmic_variance_sigma_uK2
```

### Scientific Interpretation and Limitations

Planck observational confidence intervals and theoretical cosmic variance are
different quantities. Stage 2 keeps them separate and does not add them in
quadrature. The Planck bars are reported asymmetric 68% confidence intervals
from the public spectrum. The shaded CAMB band is ideal full-sky theoretical
cosmic variance based on the model spectrum.

The Stage 2 validator checks that the final comparison table still matches both
source tables: the processed Planck low-ell CSV and the original Stage 1 CAMB
CSV. It also re-parses the official raw Planck text file to confirm that the
processed low-ell Planck table still matches the official source rows.

`normalized_residual_cv` is a convenient residual measured relative to the
ideal full-sky cosmic-variance standard deviation. It is not an exact Gaussian
z-score, not an official Planck significance, and not the official Planck
low-ell likelihood. At very low ell, the sampling distribution is not well
approximated as Gaussian.

This direct comparison is descriptive. It does not calculate an overall
p-value, chi-square statistic, look-elsewhere correction, or Monte Carlo
simulation. Stage 3 has not yet been implemented.

## Stage 3: Pre-Specified Low-ell Monte Carlo Test

Stage 3 asks whether the combined observed Planck TT power over the fixed range
`ell = 2` through `ell = 29` is unusually low compared with the fixed CAMB
theoretical spectrum, under an ideal full-sky Gaussian CMB approximation.

The analysis choices are recorded before simulation in:

```text
STAGE3_ANALYSIS_PLAN.md
```

### Primary Statistic

The statistic is the ratio of observed to theoretical band-limited temperature
variance:

```text
R_low =
    sum(weight_ell * D_ell_observed)
    /
    sum(weight_ell * D_ell_theory)
```

with:

```text
weight_ell = (2ell + 1) / (ell(ell + 1))
```

The factor `1/[ell(ell+1)]` is necessary because the public spectrum is in
`D_ell`, while the temperature variance is defined using `C_ell`:

```text
V = (1 / 4pi) * sum[(2ell + 1) C_ell]
```

Since:

```text
D_ell = ell(ell + 1) C_ell / (2pi)
```

the common constants cancel in the ratio, but the `1/[ell(ell+1)]` dependence
does not.

The primary hypothesis is low-power suppression, so the p-value is one-sided
and lower-tail.

### Monte Carlo Model

For an ideal full-sky statistically isotropic Gaussian CMB:

```text
(2ell + 1) * C_hat_ell / C_ell_theory
    follows chi-square(df = 2ell + 1)
```

The script simulates directly in `D_ell`:

```text
X_ell ~ chi-square(df = 2ell + 1)
D_ell_sim = D_ell_theory * X_ell / (2ell + 1)
```

Each simulated sky gives one `R_low_sim` value. The reported lower-tail p-value
uses the add-one correction:

```text
p_lower = (k + 1) / (N + 1)
```

where `k` is the number of simulated `R_low` values less than or equal to the
observed `R_low`. The Monte Carlo standard error is:

```text
SE_p = sqrt(p_lower * (1 - p_lower) / N)
```

The reported 95% interval is the finite-simulation uncertainty of the Monte
Carlo p-value, not cosmological uncertainty.

### Stage 3 Commands

```powershell
.venv\Scripts\activate
python src\monte_carlo_lowell.py --n-sim 1000000 --seed 20260701 --chunk-size 50000
python src\validate_monte_carlo_lowell.py
```

For a quick test run:

```powershell
python src\monte_carlo_lowell.py --n-sim 10000 --seed 20260701 --chunk-size 2000
python src\validate_monte_carlo_lowell.py
```

### Stage 3 Output Files

```text
outputs/tables/monte_carlo_lowell_summary.csv
outputs/tables/monte_carlo_lowell_quantiles.csv
outputs/tables/monte_carlo_per_ell_moments.csv
outputs/simulations/monte_carlo_R_low_samples.npz
outputs/text/stage3_methods_and_results.md
outputs/figures/monte_carlo_lowell_R_histogram.png
outputs/figures/monte_carlo_lowell_R_histogram.pdf
outputs/figures/monte_carlo_lowell_R_cdf.png
outputs/figures/monte_carlo_lowell_R_cdf.pdf
```

### Stage 3 Assumptions and Limitations

The simulation includes ideal cosmic variance only. Planck's asymmetric
reported confidence intervals are not sampled. The model assumes full-sky
coverage, independent multipoles, and fixed CAMB parameters. Real Planck
analysis includes masking, foreground treatment, anisotropic noise, likelihood
construction, and mode coupling.

The CAMB parameters were inferred using Planck-related observations, so this is
a conditional descriptive consistency test rather than a fully independent
hypothesis test. The public Planck spectrum points are used as observed
band-power estimates, not as a replacement for the official low-ell likelihood.
A small p-value would indicate that the observed combined power is unusual under
this simplified model, but would not by itself prove new physics. A moderate
p-value would indicate that ideal cosmic variance can reasonably account for the
observed suppression under this simplified model.
