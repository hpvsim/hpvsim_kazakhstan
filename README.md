# hpvsim_kazakhstan

An [HPVsim](https://hpvsim.org) model of cervical cancer for Kazakhstan, calibrated to
Globocan/IARC incidence data. Built on **hpvsim v3.0**, which is rebuilt on
[Starsim](https://docs.starsim.org) — see the
[v2->v3 migration guide](https://github.com/starsimhub/hpvsim/blob/v3.0-dev/docs/migration.qmd)
for what's changed.

## Install

hpvsim v3.0 has not yet been released to PyPI; install from the `v3.0-dev` branch:

```bash
pip install "git+https://github.com/starsimhub/hpvsim.git@v3.0-dev"
```

## What's here

| File | Purpose |
|------|---------|
| `model.py` | Defines the Kazakhstan simulation (`make_sim`, `run_sim`) and sexual network (`make_network`). |
| `run_calibration.py` | Calibrates the model to Kazakhstan cancer-cases-by-age data (`hpv.Calibration`). |
| `data/` | Calibration targets (cancer cases, age-standardized incidence). |
| `temp/` | Source materials from an earlier hpvsim v2.2.6 port (not part of the v3 model). |

Demographics (age pyramid, births, deaths) are pulled automatically from UN WPP data
for `location='kazakhstan'` — no local demographic data needed. Sexual network
behaviour (debut, marital/casual partnership probabilities, partner counts) is ported
from a DHS-fitted v2.2.6 script — see the module docstring in `model.py` for the
per-timestep-to-annual conversion this required.

Note: female casual-partnership participation in the DHS data is very low relative to
male, and partnership formation is female-driven, so the casual layer is structurally
thin pre-calibration — see `model.py`'s `_KAZAKHSTAN_LAYER_PROBS_PT` comment. This is a
real feature of the data, and calibration (`m/f_cross_layer`, `m/f_partners.c` in
`run_calibration.py`) is the intended lever, not `beta`.

## Data provenance

- `kazakhstan_cancer_cases.csv` — Globocan/IARC cervical cancer cases by age, 2020
  (the calibration target).
- `kazakhstan_asr_cancer_incidence.csv` — Globocan/IARC age-standardized incidence rate,
  2020 (15.7 per 100k) — not a calibration target (age-standardization isn't one of
  v3's `AgeResults` keys); use as an external sanity check on the calibrated fit.

## How to run

```bash
python model.py            # single baseline (uncalibrated) run + plot (local)

# Calibration — RUN only on a multi-core VM (edit `to_run` in the file):
python run_calibration.py  # 'plot_calibration' extracts/plots locally;
                            # 'run_calibration' fits (VM only)
```

## Status

Model ported from an uncalibrated hpvsim v2.2.6 script; not yet calibrated.
