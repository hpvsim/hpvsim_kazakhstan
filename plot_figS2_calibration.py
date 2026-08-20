"""
Plot Kazakhstan calibration fit with top-N trial uncertainty.

Two panels (top-N=50 trials by mismatch → median + 95% PI ribbon):
  (A) Cervical cancers by age, 2020: model band vs. Globocan target
      (from data/kazakhstan_cancer_cases.csv).
  (B) HPV prevalence by age, 2020: model band. No external target overlaid
      (add a scatter here if a prevalence survey lands).

Usage:
  python plot_figS2_calibration.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hpvsim as hpv
import utils as ut


CANCER_YEAR = 2020
TOP_N = 50


def _load_globocan_cancers():
    df = pd.read_csv('data/kazakhstan_cancer_cases.csv')
    row = df[df['year'] == CANCER_YEAR].sort_values('age')
    return row['value'].to_numpy()


def _analyzers_factory():
    """One fresh by_age(cancers + hpv_prev) per subprocess run."""
    return [hpv.by_age(['cancers', 'hpv_prevalence'],
                       years=[CANCER_YEAR], edges=ut.AGE_EDGES)]


def _extract(sim):
    """Return (cancers_by_age, hpv_prev_by_age) as 1D arrays for CANCER_YEAR."""
    ar = sim.analyzers.by_age
    cancers = ar.to_dataframe('cancers').loc[float(CANCER_YEAR)].to_numpy()
    prev = ar.to_dataframe('hpv_prevalence').loc[float(CANCER_YEAR)].to_numpy()
    return dict(cancers=cancers, prev=prev)


def _ribbon(values):
    """values: list of 1D arrays (n_trials, n_bins). Return (median, lo, hi)."""
    arr = np.array(values)
    return (np.median(arr, axis=0),
            np.percentile(arr, 2.5, axis=0),
            np.percentile(arr, 97.5, axis=0))


def _plot_cancers(ax, med, lo, hi, cancers_data, labels):
    x = np.arange(len(labels))
    ax.fill_between(x, lo, hi, color='#c1981d', alpha=0.25, label=f'Top {TOP_N} 95% PI')
    ax.plot(x, med, marker='o', color='#c1981d', lw=2, label=f'Top {TOP_N} median')
    ax.scatter(x, cancers_data, marker='d', s=60, color='k', zorder=3, label='Globocan 2020')
    ax.set_title(f'Cervical cancers by age, {CANCER_YEAR}')
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual cases')
    ax.legend()


def _plot_hpv_prevalence(ax, med, lo, hi, labels):
    x = np.arange(len(labels))
    ax.fill_between(x, lo * 100, hi * 100, color='#3a6b8e', alpha=0.25,
                    label=f'Top {TOP_N} 95% PI')
    ax.plot(x, med * 100, marker='o', color='#3a6b8e', lw=2, label=f'Top {TOP_N} median')
    ax.set_title(f'HPV prevalence by age, {CANCER_YEAR}')
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel('Age')
    ax.set_ylabel('HPV prevalence (%)')
    ax.set_ylim(bottom=0)
    ax.legend()


def main(outpath='figures/figS2_calibration.png'):
    ut.set_font(12)
    results = ut.run_top_n(
        n=TOP_N,
        sim_kwargs=dict(stop=CANCER_YEAR + 1),
        analyzers_factory=_analyzers_factory,
        extract_fn=_extract,
    )
    cancers_med, cancers_lo, cancers_hi = _ribbon([r['cancers'] for r in results])
    prev_med, prev_lo, prev_hi = _ribbon([r['prev'] for r in results])

    cancers_data = _load_globocan_cancers()
    labels = ut.age_labels()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='tight')
    _plot_cancers(axes[0], cancers_med, cancers_lo, cancers_hi, cancers_data, labels)
    _plot_hpv_prevalence(axes[1], prev_med, prev_lo, prev_hi, labels)
    fig.savefig(outpath, dpi=200)
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
