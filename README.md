# Investigating Low-ℓ Power Suppression in the Planck 2018 CMB Temperature Spectrum Using Cosmic-Variance Analysis

This repository contains the computational pipeline developed for the Pioneer Academics research project:

**“Investigating Low-ℓ Power Suppression in the Planck 2018 CMB Temperature Spectrum Using Cosmic-Variance Analysis.”**

The project investigates the observed suppression of large-scale temperature fluctuations in the Cosmic Microwave Background (CMB) and evaluates whether this deviation from the standard ΛCDM prediction represents a statistically significant anomaly or a fluctuation consistent with cosmic variance.

---

## Research Overview

The low-multipole (low-ℓ) region of the CMB temperature power spectrum describes fluctuations on the largest observable angular scales of the Universe. Previous observations have reported lower-than-expected power in this region compared with standard ΛCDM predictions.

This project studies this feature by:

- Generating theoretical ΛCDM CMB temperature spectra using **CAMB**
- Comparing predictions with **Planck 2018 TT power spectrum data**
- Calculating residuals and fractional differences
- Evaluating deviations using cosmic-variance uncertainties
- Performing statistical significance analysis
- Testing phenomenological primordial power-spectrum cutoff models
- Running Monte Carlo simulations to evaluate the likelihood of observing similar low-ℓ deviations

---

## Computational Pipeline

The analysis follows this workflow:

```text
Planck 2018 TT Data
        |
        v
Data Processing
        |
        v
ΛCDM Spectrum Generation (CAMB)
        |
        v
Residual and Cosmic-Variance Analysis
        |
        v
Statistical Evaluation
        |
        v
Primordial Cutoff-Model Testing
        |
        v
Validation
```

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
└── src/
    ├── analyze_cutoff_models.py
    ├── analyze_low_ell.py
    ├── cutoff_models.py
    ├── download_planck_data.py
    ├── generate_camb_spectrum.py
    ├── generate_cutoff_spectra.py
    ├── monte_carlo_lowell.py
    ├── validate_camb_output.py
    ├── validate_cutoff_models.py
    ├── validate_low_ell_analysis.py
    └── validate_monte_carlo_lowell.py
```

---

## Main Components

### ΛCDM Spectrum Generation

`generate_camb_spectrum.py`

Generates theoretical CMB temperature power spectra based on the standard ΛCDM cosmological model using CAMB.

### Planck Data Processing and Low-ℓ Analysis

`download_planck_data.py` and `analyze_low_ell.py`

These scripts process Planck 2018 TT observations and compare them with theoretical predictions using:

- Absolute residuals
- Fractional differences
- Cosmic-variance uncertainty
- Normalized deviations

### Cosmic-Variance Monte Carlo Analysis

`monte_carlo_lowell.py`

Generates simulated CMB realizations under ΛCDM assumptions to estimate how frequently similar low-ℓ power suppression can occur because of statistical fluctuations.

### Primordial Cutoff Models

`cutoff_models.py`, `generate_cutoff_spectra.py`, and `analyze_cutoff_models.py`

These scripts implement and evaluate phenomenological modifications to the primordial power spectrum, including cutoff models designed to reduce large-scale power.

---

## Validation

The pipeline includes validation scripts that check:

- CAMB spectrum generation
- Planck-data processing
- Low-ℓ statistical calculations
- Cutoff-model implementation
- Monte Carlo analysis

All critical validation checks passed during the final analysis pipeline.

---

## Requirements

The main Python packages used in this project are:

```text
numpy
scipy
matplotlib
pandas
astropy
camb
```

These can be installed manually or listed in a future `requirements.txt` file.

---

## Reproducibility

To reproduce the analysis:

1. Install the required Python packages.
2. Download or provide the Planck 2018 TT observational data.
3. Run the scripts in the `src/` directory according to the computational workflow.
4. Compare the generated spectra, residuals, statistical results, and cutoff-model outputs.

---

## Research Conclusion

The analysis finds that the observed low-ℓ temperature power is lower than the standard ΛCDM expectation, but the deviation remains consistent with statistical fluctuations associated with cosmic variance.

The results do not provide statistically significant evidence requiring a modification of the standard ΛCDM cosmological model.

---

## Author

**Hollis Luo**

Pioneer Academics Research Project

---

## License

This project is released under the MIT License.
