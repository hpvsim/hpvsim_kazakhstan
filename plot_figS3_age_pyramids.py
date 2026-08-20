"""
Plot Kazakhstan age pyramids at four snapshot years.

Uses the ``hpv.age_pyramid`` analyzer with 5-year bins to 85+. Male on the
left, female on the right (WHO convention).

Usage:
  python plot_figS3_age_pyramids.py
"""
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import hpvsim as hpv
import utils as ut


YEARS = (2025, 2050, 2075, 2100)
# 5-year bins to 85+, matching UN WPP demographic tables.
EDGES = np.array(list(range(0, 90, 5)) + [150], dtype=float)


def _labels():
    return [f'{int(EDGES[i])}-{int(EDGES[i+1])-1}' for i in range(len(EDGES) - 2)] + ['85+']


def _run_sim():
    """Best-fit sim with age_pyramid at requested years."""
    ap = hpv.age_pyramid(timepoints=[f'{y}-01-01' for y in YEARS], edges=EDGES)
    sim = ut.build_best_fit_sim(stop=max(YEARS) + 1, analyzers=[ap])
    sim.run()
    return sim.analyzers.age_pyramid


def _plot_pyramid(ax, pyr_arr, labels, year):
    """pyr_arr: (nbins, 2) [male, female]."""
    male, female = pyr_arr[:, 0], pyr_arr[:, 1]
    y = np.arange(len(labels))
    ax.barh(y, -male, color='C0', label='Male')
    ax.barh(y, female, color='C3', label='Female')
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title(str(year))
    xmax = max(male.max(), female.max())
    ax.set_xlim(-1.1 * xmax, 1.1 * xmax)
    # Symmetric abs-value x labels
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [f'{abs(int(t)):,}' for t in ticks], fontsize=8)
    ax.set_xlabel('Population')


def main(outpath='figures/figS3_age_pyramids.png'):
    ut.set_font(11)
    ap = _run_sim()
    pyrs = sc.odict(ap.age_pyramids)  # keyed by ss.date, insertion order
    labels = _labels()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='tight')
    axes = axes.flatten()
    for i, (year, ax) in enumerate(zip(YEARS, axes)):
        _plot_pyramid(ax, pyrs[i], labels, year)
        if i == 0:
            ax.legend(loc='upper right')
    fig.suptitle('Kazakhstan age pyramids', y=1.02)
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
