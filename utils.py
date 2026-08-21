"""Shared utilities for Kazakhstan figS1-S4 plot scripts.

- ``set_font``: consistent matplotlib font size across figures.
- ``AGE_EDGES``: standard age binning used by the calibration + figures
  (matches ``kazakhstan_cancer_cases.csv``).
- ``WHO2000_WEIGHTS``: WHO 2000 world standard population weights per
  ``AGE_EDGES`` bin (sum = 100000), used for age-standardized rate (ASR).
- ``compute_asr``: age-standardized cancer incidence per 100k, given per-bin
  annual cancer counts and per-bin female-population counts.
- ``build_best_fit_sim``: rebuild the calibrated Kazakhstan sim from a saved
  ``best_pars`` dict (typically ``results/kazakhstan_pars.obj``), applying
  the calibrated pars via ``hpv.route_pars``.
"""
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc

import hpvsim as hpv
import model as md


# Age bins used by the calibration target (kazakhstan_cancer_cases.csv) and
# reused across the figS series so the WHO2000 aggregation lines up.
AGE_EDGES = np.array([0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 150], dtype=float)

# WHO 2000 world standard population weights, aggregated to AGE_EDGES bins
# (sum = 100000). Used for age-standardized incidence: ASR = Σ (rate_i × w_i)
# / Σ w_i, where rate_i is age-specific incidence per 100k in bin i.
# 0-15 combines the 0-4, 5-9, 10-14 five-year bands; 85+ combines 85-89, 90+.
WHO2000_WEIGHTS = np.array([
    26150,  # 0-15
    8470,   # 15-20
    8220,   # 20-25
    7930,   # 25-30
    7610,   # 30-35
    7150,   # 35-40
    6590,   # 40-45
    6040,   # 45-50
    5370,   # 50-55
    4550,   # 55-60
    3720,   # 60-65
    2960,   # 65-70
    2210,   # 70-75
    1520,   # 75-80
    910,    # 80-85
    635,    # 85+ (85-89 + 90-94 + 95-99 + 100+ = 440+150+40+5)
], dtype=float)
# WHO 5-year weights sum to ~100035 as commonly published (a rounding
# artifact); ASR normalizes by weights.sum() so this is harmless.
assert len(WHO2000_WEIGHTS) == len(AGE_EDGES) - 1


def set_font(size=14):
    """Consistent font size across figS plots."""
    plt.rcParams.update({'font.size': size})


def age_labels(edges=None):
    """Match ``hpv.by_age``'s labeling convention: 'lo-hi' bins with an
    open-ended '85+' final bin."""
    edges = AGE_EDGES if edges is None else edges
    labels = [f'{int(edges[i])}-{int(edges[i + 1])}' for i in range(len(edges) - 2)]
    labels.append(f'{int(edges[-2])}+')
    return labels


def compute_asr(cancers_by_age, n_female_by_age, weights=None):
    """Age-standardized cancer incidence per 100k person-years.

    Args:
        cancers_by_age: array of new cancer counts per age bin (annual events).
        n_female_by_age: array of female population at risk per age bin.
        weights: WHO standard weights per bin (default: WHO 2000 aggregated
            to AGE_EDGES; must sum to 100000).

    Returns: ASR as a float (per 100k).
    """
    if weights is None:
        weights = WHO2000_WEIGHTS
    cancers_by_age = np.asarray(cancers_by_age, dtype=float)
    n_female_by_age = np.asarray(n_female_by_age, dtype=float)
    # Age-specific rate per 100k, guarding against empty bins.
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(n_female_by_age > 0,
                         cancers_by_age / n_female_by_age * 1e5,
                         0.0)
    return float(np.sum(rates * weights) / weights.sum())


def _build_sim_with_pars(pars, sim_kwargs, analyzers=None):
    """Build a Kazakhstan sim from an arbitrary calibration par set.
    Used by both ``build_best_fit_sim`` and the top-N ribbon helper.
    """
    p = dict(pars)
    p.pop('rand_seed', None)  # Optuna leaks this; not a calibratable model par
    sim_kw = dict(sim_kwargs)
    if analyzers is not None:
        sim_kw['analyzers'] = analyzers
    sim = md.make_sim(**sim_kw)
    hpv.route_pars(sim, p)
    return sim


def _run_and_extract(pars, sim_kwargs, analyzers_factory, extract_fn):
    """sc.parallelize worker: build sim with ``pars``, run, apply
    ``extract_fn(sim)``, return its result. Uses ``analyzers_factory()`` so
    each subprocess gets its own analyzer instances (pickling shared instances
    corrupts their state)."""
    sim = _build_sim_with_pars(pars, sim_kwargs, analyzers=analyzers_factory())
    sim.run()
    return extract_fn(sim)


def run_top_n(n, sim_kwargs, analyzers_factory, extract_fn,
              calib_path='results/kazakhstan_calib.obj', n_workers=None):
    """Rebuild + run the top-``n`` calibration trials (sorted by mismatch)
    in parallel and return a list of ``extract_fn(sim)`` results.

    ``analyzers_factory``: zero-arg callable returning a fresh list of
        analyzers (fresh instances per subprocess).
    ``extract_fn(sim) -> anything picklable``: runs post-sim.run().
    """
    calib = sc.load(calib_path)
    top = calib.df.nsmallest(n, 'mismatch')
    # rand_seed is Optuna leakage when reseed=True (not calibratable).
    par_cols = [c for c in top.columns if c not in ('index', 'mismatch', 'rand_seed')]
    par_sets = [{c: row[c] for c in par_cols} for _, row in top.iterrows()]
    if n_workers is None:
        n_workers = min(len(par_sets), sc.cpu_count())
    return sc.parallelize(
        _run_and_extract, iterkwargs=[{'pars': p} for p in par_sets],
        kwargs=dict(sim_kwargs=sim_kwargs, analyzers_factory=analyzers_factory,
                    extract_fn=extract_fn),
        ncpus=n_workers, serial=False,
    )


def build_best_fit_sim(pars_path='results/kazakhstan_pars.obj', **sim_kwargs):
    """Build a Kazakhstan sim with calibrated pars applied via ``hpv.route_pars``.

    ``pars_path`` points to the ``best_pars`` dict saved by
    ``run_calibration.load_calib``. ``sim_kwargs`` overrides ``md.make_sim`` defaults.
    """
    best_pars = dict(sc.load(pars_path))
    sim = md.make_sim(**sim_kwargs)
    hpv.route_pars(sim, best_pars)
    return sim
