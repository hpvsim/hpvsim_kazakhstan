"""
Plot Kazakhstan sexual-network structure.

Three panels:
  (A) Age at first sex CDF: lognormal debut (per sex), from network_pars().
  (B) Age-mixing heatmap of currently-active pairs at the snapshot year,
      binned into 5-year age bands. Rows = female age, cols = male age.
  (C) Partnership status by age at the snapshot year: % of women / men
      with >=1 marital partner and % with >=1 casual partner.

(B) and (C) are computed from a sim run to ``SNAPSHOT_YEAR``. (A) is
read directly from ``model.network_pars()``.

Usage:
  python plot_figS1_network.py
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import starsim as ss

import model as md
import utils as ut


SNAPSHOT_YEAR = 2020
AGE_BINS = np.array([0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
                    dtype=float)
_AGE_LABELS = ([f'{int(AGE_BINS[i])}-{int(AGE_BINS[i+1])-1}'
                for i in range(len(AGE_BINS) - 2)] + ['65+'])
F_COLOR = '#d46e9c'
M_COLOR = '#4a90d9'

NETWORK_PARS = md.network_pars()


class NetworkTracker(ss.Analyzer):
    """Snapshot of the sexual-network state at the requested year:
    age-mixing pairs + per-agent marital/casual presence flags."""

    def __init__(self, snapshot_year):
        super().__init__()
        self.snapshot_year = snapshot_year
        self.snapshot = None
        self._snapshot_taken = False

    def step(self):
        if self._snapshot_taken:
            return
        if float(self.sim.now.years) < self.snapshot_year:
            return
        self._take_snapshot()
        self._snapshot_taken = True

    def _take_snapshot(self):
        net = self.sim.networks.sexualnetwork
        people = self.sim.people
        edges = net.edges
        p1 = np.asarray(edges.p1)
        p2 = np.asarray(edges.p2)
        lid = np.asarray(edges.layer_id)
        m_idx = net._layer_idx['m']
        c_idx = net._layer_idx['c']

        ages = np.asarray(people.age.raw)
        alive = np.asarray(people.alive.raw)
        female = np.asarray(people.female.raw)
        fine = (np.asarray(people.fine.raw) if 'fine' in people.states
                else np.zeros_like(alive))
        alive = alive & ~fine

        f_age = ages[p1]
        m_age = ages[p2]

        n = len(people.age.raw)
        has_m = np.zeros(n, dtype=bool)
        has_c = np.zeros(n, dtype=bool)
        m_mask = lid == m_idx
        c_mask = lid == c_idx
        for endpts in (p1[m_mask], p2[m_mask]):
            has_m[endpts] = True
        for endpts in (p1[c_mask], p2[c_mask]):
            has_c[endpts] = True

        self.snapshot = dict(
            f_age=f_age, m_age=m_age,
            has_m=has_m, has_c=has_c,
            age=ages, alive=alive, female=female,
        )


def _lognorm_from_mean_std(mean, std):
    """scipy.stats.lognorm parametrized to match ``ss.lognorm_ex(mean, std)``."""
    var = std ** 2
    sigma = np.sqrt(np.log(1 + var / mean ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return sps.lognorm(s=sigma, scale=np.exp(mu))


def _plot_debut(ax):
    ax.set_title('Age at first sex (lognormal CDF)')
    x = np.linspace(10, 40, 300)
    for sex, color, label in (('f', F_COLOR, 'female'), ('m', M_COLOR, 'male')):
        dist = NETWORK_PARS[f'debut_{sex}']
        mean, std = dist.pars['mean'], dist.pars['std']
        rv = _lognorm_from_mean_std(mean, std)
        ax.plot(x, rv.cdf(x) * 100, color=color, lw=2,
                label=f'{label} (μ={mean}, σ={std})')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('% ever sexually active')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')


def _plot_mixing(ax, tracker):
    snap = tracker.snapshot
    if snap is None or len(snap['f_age']) == 0:
        ax.set_title(f'Age mixing at {SNAPSHOT_YEAR} (no pairs)')
        return
    edges = np.arange(15, 71, 5)
    h, _, _ = np.histogram2d(snap['f_age'], snap['m_age'], bins=[edges, edges])
    row_sums = h.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_norm = np.where(row_sums > 0, h / row_sums, 0.0)
    im = ax.imshow(h_norm, origin='lower', cmap='magma', aspect='auto',
                   extent=[edges[0], edges[-1], edges[0], edges[-1]])
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color='w', ls='--', lw=1)
    ax.set_xlabel('Male partner age (years)')
    ax.set_ylabel('Female age (years)')
    ax.set_title(f'Age mixing among active pairs, {SNAPSHOT_YEAR}\n'
                 '(row-normalized; dashed = equal ages)')
    plt.colorbar(im, ax=ax, label='P(male age | female age)')


def _plot_status(ax, tracker):
    snap = tracker.snapshot
    if snap is None:
        ax.set_title(f'Partnership status at {SNAPSHOT_YEAR} (no snapshot)')
        return
    age = snap['age']
    alive = snap['alive']
    female = snap['female']
    for sex_mask, sex_label, ls in ((female & alive, 'Female', '-'),
                                     ((~female) & alive, 'Male', '--')):
        n_by_age, _ = np.histogram(age[sex_mask], bins=AGE_BINS)
        m_by_age, _ = np.histogram(age[sex_mask & snap['has_m']], bins=AGE_BINS)
        c_by_age, _ = np.histogram(age[sex_mask & snap['has_c']], bins=AGE_BINS)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_m = np.where(n_by_age > 0, m_by_age / n_by_age, 0.0) * 100
            p_c = np.where(n_by_age > 0, c_by_age / n_by_age, 0.0) * 100
        x = np.arange(len(_AGE_LABELS))
        ax.plot(x, p_m, color='#2171b5', ls=ls, lw=2, label=f'{sex_label}, marital')
        ax.plot(x, p_c, color='#ff7f00', ls=ls, lw=2, label=f'{sex_label}, casual')
    ax.set_xticks(np.arange(len(_AGE_LABELS)), _AGE_LABELS, rotation=45)
    ax.set_xlabel('Age')
    ax.set_ylabel('% with active partner in layer')
    ax.set_ylim(0, 100)
    ax.set_title(f'Partnership status by age, {SNAPSHOT_YEAR}')
    ax.legend(fontsize=9, loc='upper right')


def _run_sim():
    tracker = NetworkTracker(snapshot_year=SNAPSHOT_YEAR)
    sim = ut.build_best_fit_sim(stop=SNAPSHOT_YEAR + 1, analyzers=[tracker])
    sim.run()
    return sim.analyzers.networktracker


def main(outpath='figures/figS1_network.png'):
    ut.set_font(11)
    tracker = _run_sim()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='tight')
    _plot_debut(axes[0])
    _plot_mixing(axes[1], tracker)
    _plot_status(axes[2], tracker)
    fig.savefig(outpath, dpi=200)
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
