"""
Plot sexual behavior in the Kazakhstan model (no sim required).

Three panels:
  (A) Age at first sex CDF: lognormal debut (per sex).
  (B) Annual participation probability by age band (marital + casual, per sex).
  (C) Casual-layer partner-count PMF: Poisson mean (per sex).

All values are read from ``model.network_pars()`` at run time.

Usage:
  python plot_figS1_behavior.py
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

import model as md
import utils as ut


NETWORK_PARS = md.network_pars()


def _lognorm_from_mean_std(mean, std):
    """scipy.stats.lognorm parametrized to match ``ss.lognorm_ex(mean, std)``."""
    var = std ** 2
    sigma = np.sqrt(np.log(1 + var / mean ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return sps.lognorm(s=sigma, scale=np.exp(mu))


def _plot_debut(ax):
    ax.set_title('Age at first sex (lognormal CDF)')
    x = np.linspace(10, 40, 300)
    for sex, color, label in (('f', 'C0', 'female'), ('m', 'C1', 'male')):
        dist = NETWORK_PARS[f'debut_{sex}']
        mean, std = dist.pars['mean'], dist.pars['std']
        rv = _lognorm_from_mean_std(mean, std)
        ax.plot(x, rv.cdf(x) * 100, color=color, lw=2,
                label=f'{label} (μ={mean}, σ={std})')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('% ever sexually active')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')


def _plot_layer_probs(ax):
    ax.set_title('Annual participation probability by age')
    for layer, sex, color, ls, label in (
        ('marital', 'f', 'C0', '-',  'marital, female'),
        ('marital', 'm', 'C0', '--', 'marital, male'),
        ('casual',  'f', 'C3', '-',  'casual, female'),
        ('casual',  'm', 'C3', '--', 'casual, male'),
    ):
        arr = NETWORK_PARS[f'layer_probs_{layer}']
        bins, probs = arr[0, :], arr[1 if sex == 'f' else 2, :]
        ax.step(bins, probs, where='post', color=color, ls=ls, lw=2, label=label)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Annual probability of participation')
    ax.set_xlim(10, 80)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9)


def _plot_partners(ax):
    ax.set_title('Concurrent casual partners (Poisson PMF)')
    max_k = 4
    ks = np.arange(0, max_k + 1)
    width = 0.35
    for i, (sex, color, label) in enumerate((('f', 'C0', 'female'), ('m', 'C1', 'male'))):
        lam = NETWORK_PARS[f'{sex}_partners_casual']
        pmf = sps.poisson.pmf(ks, mu=lam)
        offset = (i - 0.5) * width
        ax.bar(ks + offset, pmf * 100, width=width, color=color,
               label=f'{label} (μ={lam})')
    ax.set_xticks(ks)
    ax.set_xlabel('Casual partners (count)')
    ax.set_ylabel('% of agents')
    ax.legend(loc='upper right')


def main(outpath='figures/figS1_behavior.png'):
    ut.set_font(12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='tight')
    _plot_debut(axes[0])
    _plot_layer_probs(axes[1])
    _plot_partners(axes[2])
    fig.savefig(outpath, dpi=200)
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
