"""
Plot Kazakhstan time series 2005-2045 with top-N trial uncertainty.

Three panels (top-N=50 trials by mismatch → median + 95% PI ribbon):
  (A) HPV prevalence in women PREV_MIN_AGE+ with normal cervical cytology
      (precin_prevalence), aggregated across bins by pop-weighted mean.
  (B) CIN prevalence in women PREV_MIN_AGE+, pop-weighted mean.
  (C) Age-standardized cancer incidence per 100k (WHO 2000 world standard).

Usage:
  python plot_figS4_timeseries.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

import hpvsim as hpv
import utils as ut


START = 2005
STOP = 2045
YEARS = list(range(START, STOP + 1))
TOP_N = 50

# Restrict the prevalence panels to adult women. Standard epidemiological
# reporting for HPV-in-normal-cytology and CIN prevalence excludes 0-14
# (~0 prevalence, dilutes the mean); ASR keeps the full age range because
# WHO2000 standardization is defined over 0+.
PREV_MIN_AGE = 15


def _analyzers_factory():
    """One fresh by_age + age_pyramid pair per subprocess run."""
    return [
        hpv.by_age(['cancers', 'precin_prevalence', 'cin_prevalence'],
                   years=YEARS, edges=ut.AGE_EDGES),
        hpv.age_pyramid(timepoints=[f'{y}-01-01' for y in YEARS],
                        edges=ut.AGE_EDGES),
    ]


def _pop_weighted_mean(prev_by_age_year, n_female_by_year):
    weighted = (prev_by_age_year.values * n_female_by_year.values).sum(axis=1)
    denom = n_female_by_year.values.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(denom > 0, weighted / denom, np.nan)
    return pd.Series(out, index=prev_by_age_year.index)


def _extract(sim):
    """Return dict of per-year 1D arrays: precin, cin, asr."""
    ar = sim.analyzers.by_age
    ap = sim.analyzers.age_pyramid

    cancers = ar.to_dataframe('cancers')
    precin = ar.to_dataframe('precin_prevalence')
    cin = ar.to_dataframe('cin_prevalence')

    female_rows = []
    for date, arr in sc.odict(ap.age_pyramids).items():
        year = float(int(date.years))
        female_rows.append([year, *arr[:, 1]])
    female_df = (pd.DataFrame(female_rows, columns=['t', *cancers.columns])
                 .set_index('t').reindex(cancers.index))

    adult_bins = [c for c, lo in zip(cancers.columns, ut.AGE_EDGES[:-1])
                  if lo >= PREV_MIN_AGE]
    precin_mean = _pop_weighted_mean(precin[adult_bins], female_df[adult_bins])
    cin_mean = _pop_weighted_mean(cin[adult_bins], female_df[adult_bins])
    asr = pd.Series(
        [ut.compute_asr(cancers.iloc[i].values, female_df.iloc[i].values)
         for i in range(len(cancers))],
        index=cancers.index,
    )
    return dict(precin=precin_mean.values, cin=cin_mean.values,
                asr=asr.values, years=cancers.index.values)


def _ribbon(values):
    """values: list of 1D arrays over years. Return (median, lo, hi)."""
    arr = np.array(values)
    return (np.median(arr, axis=0),
            np.percentile(arr, 2.5, axis=0),
            np.percentile(arr, 97.5, axis=0))


def _plot(ax, years, med, lo, hi, title, ylabel, color):
    ax.fill_between(years, lo, hi, color=color, alpha=0.25, label=f'Top {TOP_N} 95% PI')
    ax.plot(years, med, color=color, lw=2, label=f'Top {TOP_N} median')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_xlim(START, STOP)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='best')


def main(outpath='figures/figS4_timeseries.png'):
    ut.set_font(12)
    results = ut.run_top_n(
        n=TOP_N,
        sim_kwargs=dict(stop=STOP + 1),
        analyzers_factory=_analyzers_factory,
        extract_fn=_extract,
    )
    years = results[0]['years']
    precin_med, precin_lo, precin_hi = _ribbon([r['precin'] for r in results])
    cin_med, cin_lo, cin_hi = _ribbon([r['cin'] for r in results])
    asr_med, asr_lo, asr_hi = _ribbon([r['asr'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='tight')
    _plot(axes[0], years, precin_med * 100, precin_lo * 100, precin_hi * 100,
          f'HPV in women {PREV_MIN_AGE}+ with normal cytology',
          'Prevalence (%)', '#3a6b8e')
    _plot(axes[1], years, cin_med * 100, cin_lo * 100, cin_hi * 100,
          f'CIN prevalence, women {PREV_MIN_AGE}+',
          'Prevalence (%)', '#c1981d')
    _plot(axes[2], years, asr_med, asr_lo, asr_hi,
          'Age-standardized cancer incidence',
          'ASR per 100,000 (WHO 2000)', '#a63636')
    fig.savefig(outpath, dpi=200)
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
