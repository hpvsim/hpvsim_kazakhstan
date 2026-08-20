"""
Plot Kazakhstan sexual-network model outputs (as realized by the sim).

Three panels:
  (A) Lifetime partner distribution per layer (marital, casual), per sex.
      A running counter increments each time an agent appears in a
      newly-formed edge (identified via ``edges.start_ti == sim.ti``).
  (B) Age-mixing heatmap of currently-active pairs at the snapshot year,
      binned into 5-year age bands. Rows = female age, cols = male age.
  (C) Partnership status by age at the snapshot year: % of women / men
      with >=1 marital partner and % with >=1 casual partner.

Usage:
  python plot_figS5_network.py
"""
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss

import utils as ut


SNAPSHOT_YEAR = 2020
AGE_BINS = np.array([0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
                    dtype=float)
_AGE_LABELS = ([f'{int(AGE_BINS[i])}-{int(AGE_BINS[i+1])-1}'
                for i in range(len(AGE_BINS) - 2)] + ['65+'])
F_COLOR = '#d46e9c'
M_COLOR = '#4a90d9'


class NetworkTracker(ss.Analyzer):
    """Per-agent lifetime marital / casual partner counts + snapshot of the
    network state at the requested year (age-mixing pairs, marital / casual
    presence flags per agent)."""

    def __init__(self, snapshot_year):
        super().__init__()
        self.snapshot_year = snapshot_year
        # Populated in init_pre once population size is known.
        self.lifetime_m = None
        self.lifetime_c = None
        # Populated on the last matching timestep (see step).
        self.snapshot = None  # dict of arrays
        self._snapshot_taken = False

    def init_pre(self, sim):
        super().init_pre(sim)
        # Per-agent lifetime edge counts, keyed by uid. sim.people can grow
        # (births); index into .raw and let it auto-resize via full().
        n = len(sim.people)
        self.lifetime_m = np.zeros(n, dtype=int)
        self.lifetime_c = np.zeros(n, dtype=int)

    def _ensure_capacity(self, needed):
        if needed <= len(self.lifetime_m):
            return
        grow = needed - len(self.lifetime_m)
        self.lifetime_m = np.concatenate([self.lifetime_m, np.zeros(grow, dtype=int)])
        self.lifetime_c = np.concatenate([self.lifetime_c, np.zeros(grow, dtype=int)])

    def step(self):
        net = self.sim.networks.sexualnetwork
        edges = net.edges
        n_edges = len(edges)
        if n_edges:
            start_ti = np.asarray(edges.start_ti)
            mask = start_ti == self.sim.ti  # new this step
            if mask.any():
                p1 = np.asarray(edges.p1)[mask]
                p2 = np.asarray(edges.p2)[mask]
                lid = np.asarray(edges.layer_id)[mask]
                self._ensure_capacity(int(max(p1.max(), p2.max())) + 1)
                # Marital layer id = 0, casual = 1 (order in net.layers).
                m_mask = lid == net._layer_idx['m']
                c_mask = lid == net._layer_idx['c']
                for arr, sub in ((self.lifetime_m, m_mask), (self.lifetime_c, c_mask)):
                    if sub.any():
                        np.add.at(arr, p1[sub], 1)
                        np.add.at(arr, p2[sub], 1)

        # Take snapshot on the last step in the snapshot year.
        if not self._snapshot_taken:
            cur_year = float(self.sim.now.years)
            if cur_year >= self.snapshot_year:
                self._take_snapshot(net)
                self._snapshot_taken = True

    def _take_snapshot(self, net):
        people = self.sim.people
        edges = net.edges
        p1 = np.asarray(edges.p1)
        p2 = np.asarray(edges.p2)
        lid = np.asarray(edges.layer_id)
        m_idx = net._layer_idx['m']
        c_idx = net._layer_idx['c']

        # Age-mixing pairs: p1 is female by construction (see net docs).
        # Exclude multiscale "fine" agents — they're cancer-resolution stand-ins
        # and don't participate in the sexual network.
        ages = np.asarray(people.age.raw)
        alive = np.asarray(people.alive.raw)
        female = np.asarray(people.female.raw)
        fine = (np.asarray(people.fine.raw) if 'fine' in people.states
                else np.zeros_like(alive))
        alive = alive & ~fine

        f_age = ages[p1]
        m_age = ages[p2]

        # Per-agent has-marital / has-casual flags at snapshot. Size against
        # the raw backing array (edges can reference uids past ``len(people)``
        # if births have grown the pool beyond the initial allocation).
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


def _plot_lifetime(ax, tracker, sim):
    people = sim.people
    # Tracker only grew as fast as it saw edges; slice sim.people arrays to
    # the tracker's size (later-born agents with no edges = not counted).
    n = len(tracker.lifetime_m)
    alive = np.asarray(people.alive.raw)[:n]
    female = np.asarray(people.female.raw)[:n]
    fine = (np.asarray(people.fine.raw)[:n] if 'fine' in people.states
            else np.zeros(n, dtype=bool))
    alive = alive & ~fine
    ever = tracker.lifetime_m + tracker.lifetime_c > 0
    keep = alive & ever
    for layer, arr, label in (('Marital', tracker.lifetime_m, 'marital'),
                              ('Casual', tracker.lifetime_c, 'casual')):
        pass  # not used; use per-sex bars below

    max_partners = 10
    bins = np.arange(max_partners + 2) - 0.5
    centers = np.arange(max_partners + 1)
    width = 0.35

    # We show casual-layer only (marital is ~always 0 or 1 given poisson1(0.01)).
    casual = tracker.lifetime_c
    f_vals = np.clip(casual[keep & female], 0, max_partners)
    m_vals = np.clip(casual[keep & ~female], 0, max_partners)
    f_h, _ = np.histogram(f_vals, bins=bins)
    m_h, _ = np.histogram(m_vals, bins=bins)
    f_h = f_h / max(f_h.sum(), 1)
    m_h = m_h / max(m_h.sum(), 1)
    ax.bar(centers - width / 2, f_h, width=width, color=F_COLOR, label='Female')
    ax.bar(centers + width / 2, m_h, width=width, color=M_COLOR, label='Male')
    ax.set_xticks([0, 2, 4, 6, 8, 10], ['0', '2', '4', '6', '8', '10+'])
    ax.set_xlabel('Lifetime casual partners')
    ax.set_ylabel('Proportion of debuted agents')
    ax.set_title('Lifetime casual-partner distribution')
    ax.legend()


def _plot_mixing(ax, tracker):
    snap = tracker.snapshot
    if snap is None or len(snap['f_age']) == 0:
        ax.set_title(f'Age mixing at {SNAPSHOT_YEAR} (no pairs)')
        return
    # 5-year bins to 65+; drop the tail if it's empty.
    edges = np.arange(15, 71, 5)
    h, _, _ = np.histogram2d(snap['f_age'], snap['m_age'], bins=[edges, edges])
    # Row-normalize so each female age band's row sums to 1 (conditional
    # distribution of male age | female age).
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
    # Fetch the actual attached instance (ss.Sim's copy_inputs deep-copies).
    return sim, sim.analyzers.networktracker


def main(outpath='figures/figS5_network.png'):
    ut.set_font(11)
    sim, tracker = _run_sim()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='tight')
    _plot_lifetime(axes[0], tracker, sim)
    _plot_mixing(axes[1], tracker)
    _plot_status(axes[2], tracker)
    fig.savefig(outpath, dpi=200)
    print(f'saved {outpath}')
    return fig


if __name__ == '__main__':
    main()
