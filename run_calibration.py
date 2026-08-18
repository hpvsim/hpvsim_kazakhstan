"""
Calibrate HPVsim Kazakhstan.

Heavy calibration (`run_calibration`) is fast ONLY on multi-core VMs -- never
local. Plotting/extraction (`plot_calibration`) runs locally.

Calibration targets the 2020 Globocan/IARC cancer-cases-by-age data
(data/kazakhstan_cancer_cases.csv) via an `hpv.AgeResults` analyzer, matching
the only target used in the source v2.2.6 script. The ASR file
(data/kazakhstan_asr_cancer_incidence.csv) isn't a v3 AgeResults key
(age-standardization isn't computed by AgeResults) -- use it as an external
sanity check on the calibrated fit, not a calibration target.

Custom build_fn. hpv.calibration.build_sim's generic dotted-key router can't
reach three things the source script calibrates:
  - Network pars (m/f_cross_layer, m/f_partners.c) live on the SexualNetwork
    module in v3, not on the sim -- rebuilt per trial via model.make_network.
  - dur_cin is an ss.lognorm_ex Dist (mean/std carry ss.years units), not a
    plain dict -- rebuilt directly rather than mutated in place.
  - sev_dist (v2's global severity-scaling knob) maps to the CrossImmunity
    connector's rel_sev_loc, set via its _rel_sev_dist attribute.
The remaining pars (a shared beta broadcast to every genotype, and hi5/ohr
cancer_fn.transform_prob / cin_fn.k) delegate to hpv.calibration.build_sim.
"""
import matplotlib.pyplot as plt
import numpy as np
import optuna as op
import pandas as pd
import sciris as sc
import starsim as ss
import hpvsim as hpv
from hpvsim.hpv import HPV
from hpvsim.cross_genotype import CrossImmunity
from hpvsim.calibration import build_sim as _default_build_sim

import model as md


# Backport stisim's crash-tolerant worker (stisim/calibration.py:585-590):
# upstream ss.Calibration.worker calls study.optimize bare, so any worker
# SQLite-lock error kills the whole run. TODO: PR into hpvsim.Calibration.
def _safe_worker(self):
    op.logging.set_verbosity(op.logging.DEBUG if self.verbose else op.logging.ERROR)
    study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name,
                          sampler=self.run_args.sampler)
    try:
        return study.optimize(self.run_trial, n_trials=self.run_args.n_trials, callbacks=None)
    except Exception as e:
        print(f'Worker failed: {e}')
        return None
ss.Calibration.worker = _safe_worker

# Set by user before running
to_run = [
    # 'run_calibration',   # uncomment to RUN (VM only)
    'plot_calibration',     # uncomment to PLOT/extract (local)
]
debug = False
do_save = True
n_trials = [1000, 2][debug]
n_workers = [32, 2][debug]  # 160 deadlocked; 32 is safer on zebra

GENOTYPES = ['hpv16', 'hpv18', 'hi5', 'ohr']
CALIB_GENOTYPES = ['hi5', 'ohr']  # per-genotype progression pars calibrated (source script)

CANCER_YEAR = 2020
AGE_EDGES = np.array([0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 150], dtype=float)


def _age_labels(edges):
    """Match hpv.AgeResults' own labeling exactly, so columns line up in eval_fn."""
    labels = [f'{int(edges[i])}-{int(edges[i + 1])}' for i in range(len(edges) - 2)]
    labels.append(f'{int(edges[-2])}+')
    return labels


def load_cancer_data():
    """2020 cancer cases by age, reshaped to the t-indexed frame hpv.Calibration expects."""
    df = pd.read_csv('data/kazakhstan_cancer_cases.csv')
    row = df[df['year'] == CANCER_YEAR].sort_values('age')
    assert len(row) == len(AGE_EDGES) - 1, 'data/kazakhstan_cancer_cases.csv age bins != AGE_EDGES'
    return pd.DataFrame([row['value'].to_numpy()], columns=_age_labels(AGE_EDGES),
                        index=pd.Index([float(CANCER_YEAR)], name='t'))


def make_calib_sim(debug=0):
    ar = hpv.AgeResults(result_args=sc.objdict(
        cancers=sc.objdict(years=[CANCER_YEAR], edges=AGE_EDGES),
    ))
    # +1 year of margin past the target year (see hpvsim v3 migration notes on
    # under-counting a sim's final partial calendar year).
    return md.make_sim(debug=debug, stop=CANCER_YEAR + 1, analyzers=[ar])


def make_calib_pars():
    """Source (v2.2.6) calib_pars, converted to v3's {low, high, guess} spec.

    m/f_cross_layer bounds are annualized (md._to_annual) for the same reason
    the baseline network pars are in model.py: the source values are
    per-timestep (dt=0.25). m/f_partners.c (partner counts) and the genotype/
    beta/sev_dist pars are not rate-like and carry over unconverted.
    """
    def annual(low, high, guess):
        return dict(low=md._to_annual(low), high=md._to_annual(high), guess=md._to_annual(guess))

    pars = dict(
        beta=dict(low=0.1, high=0.34, guess=0.2),
        m_cross_layer=annual(0.1, 0.7, 0.3),
        f_cross_layer=annual(0.05, 0.5, 0.1),
        m_partners_c=dict(low=0.1, high=0.6, guess=0.2),
        f_partners_c=dict(low=0.1, high=0.6, guess=0.2),
        sev_dist=dict(low=0.5, high=1.5, guess=1.0),
    )
    for g in CALIB_GENOTYPES:
        pars[f'{g}.cancer_fn.transform_prob'] = dict(low=0.5e-3, high=2.5e-3, guess=1.5e-3)
        pars[f'{g}.cin_fn.k'] = dict(low=0.1, high=0.25, guess=0.15)
        pars[f'{g}.dur_cin_mean'] = dict(low=3.5, high=5.5, guess=4.5)
        pars[f'{g}.dur_cin_std'] = dict(low=16, high=24, guess=20)
    return pars


def build_kazakhstan_sim(sim, calib_pars, **kwargs):
    p = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v)
         for k, v in calib_pars.items()}

    # Pars that fit hpv.calibration.build_sim's default dotted-key routing:
    # a shared beta broadcast to every genotype, plus hi5/ohr cancer_fn/cin_fn.
    generic = {}
    if 'beta' in p:
        for g in GENOTYPES:
            generic[f'{g}.beta'] = p['beta']
    for g in CALIB_GENOTYPES:
        for key in ('cancer_fn.transform_prob', 'cin_fn.k'):
            pkey = f'{g}.{key}'
            if pkey in p:
                generic[pkey] = p[pkey]
    if generic:
        _default_build_sim(sim, generic)

    # dur_cin: rebuild the Dist directly (mean/std carry ss.years units).
    disease_lookup = {d.name: d for d in sim.pars.get('diseases', []) if isinstance(d, HPV)}
    for g in CALIB_GENOTYPES:
        mean_key, std_key = f'{g}.dur_cin_mean', f'{g}.dur_cin_std'
        if mean_key in p and std_key in p:
            disease_lookup[g].pars.dur_cin = ss.lognorm_ex(
                mean=ss.years(p[mean_key]), std=ss.years(p[std_key]))

    # sev_dist -> CrossImmunity connector's severity-location parameter.
    if 'sev_dist' in p:
        conn = next(c for c in sim.pars.get('connectors', []) if isinstance(c, CrossImmunity))
        conn._rel_sev_dist = ss.normal(loc=p['sev_dist'], scale=0.2)

    # Network: rebuild with the calibrated concurrency/partner-count knobs.
    net_keys = ('m_cross_layer', 'f_cross_layer', 'm_partners_c', 'f_partners_c')
    if any(k in p for k in net_keys):
        overrides = {}
        if 'm_cross_layer' in p:
            overrides['m_cross_layer'] = p['m_cross_layer']
        if 'f_cross_layer' in p:
            overrides['f_cross_layer'] = p['f_cross_layer']
        # make_network's merge is shallow (sc.mergedicts, top-level keys only),
        # so overriding m_partners/f_partners must supply the whole {m, c}
        # dict -- a partial {'c': ...} would silently drop the 'm' sub-dict.
        if 'm_partners_c' in p:
            mp = sc.dcp(md._KAZAKHSTAN_M_PARTNERS)
            mp['c']['par1'] = p['m_partners_c']
            overrides['m_partners'] = mp
        if 'f_partners_c' in p:
            fp = sc.dcp(md._KAZAKHSTAN_F_PARTNERS)
            fp['c']['par1'] = p['f_partners_c']
            overrides['f_partners'] = fp
        sim.pars['networks'] = [md.make_network(pars=overrides)]

    return sim


def run_calib(n_trials=None, n_workers=None, do_plot=False, do_save=True, filestem=''):
    sim = make_calib_sim(debug=debug)
    data = {'cancers': load_cancer_data()}
    calib = hpv.Calibration(
        sim, calib_pars=make_calib_pars(), build_fn=build_kazakhstan_sim,
        data=data, total_trials=n_trials, n_workers=n_workers,
    )
    try:
        calib.calibrate()
    except Exception as e:
        print(f'calibrate() raised: {e}; saving partial results anyway')
    if do_save:
        sc.saveobj(f'raw_results/kazakhstan_calib{filestem}.obj', calib)
    if do_plot:
        fig = hpv.plot_calibration(calib)
        fig.savefig('figures/kazakhstan_calib.png')
    if getattr(calib, 'best_pars', None) is not None:
        print(f'Best pars: {calib.best_pars}')
    return sim, calib


def plot_calib_fit(calib, n=100, outpath='figures/kazakhstan_calib_fit.png'):
    """Top-N trials -> cancers-by-age envelope vs Globocan target."""
    n = min(n, len(calib.analyzer_results))
    per_trial = np.array([calib.analyzer_results[pos]['cancers'][CANCER_YEAR] for pos in range(n)])
    med = np.median(per_trial, axis=0)
    lo = np.percentile(per_trial, 2.5, axis=0)
    hi = np.percentile(per_trial, 97.5, axis=0)
    target = load_cancer_data().iloc[0].to_numpy()
    labels = _age_labels(AGE_EDGES)
    x = np.arange(len(med))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, med, color='#c1981d', label=f'Model median (top {n})')
    ax.fill_between(x, lo, hi, alpha=0.3, color='#c1981d', label='95% PI')
    ax.scatter(x, target, marker='d', s=60, color='k', label='Globocan 2020')
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel('Age')
    ax.set_ylabel('Cancers')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    return fig


def load_calib(do_plot=True, filestem=''):
    calib = sc.load(f'raw_results/kazakhstan_calib{filestem}.obj')
    if do_plot:
        fig = hpv.plot_calibration(calib)
        fig.savefig(f'figures/kazakhstan_calib{filestem}.png')
        # plot_calib_fit needs calib.analyzer_results (populated by top-N re-runs),
        # not available on the saved calib obj -- rework pending.
        # plot_calib_fit(calib, outpath=f'figures/kazakhstan_calib_fit{filestem}.png')
    sc.save(f'results/kazakhstan_pars{filestem}.obj', calib.best_pars)
    return calib


if __name__ == '__main__':
    T = sc.timer()
    if 'run_calibration' in to_run:
        run_calib(n_trials=n_trials, n_workers=n_workers, do_save=do_save)
    if 'plot_calibration' in to_run:
        load_calib(do_plot=True)
    T.toc('Done')
