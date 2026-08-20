"""
Calibrate HPVsim Kazakhstan.

Heavy calibration is fast ONLY on multi-core VMs -- never local. Plotting /
extraction (`plot_calibration`) runs locally.

Calibration targets are provided as long-format CSVs via ``datafiles=``;
``hpv.Calibration`` derives age bins + years from the data, attaches a
``by_age`` analyzer named ``'calib_by_age'``, and extends ``sim.stop`` past
the latest data year as needed.

``calib_pars`` uses the nested v3 form (see hpv.Calibration docstring):
scopes nest by module, leaves are ``[best, low, high, step]`` lists collapsed
by ``sc.flattendict`` before Optuna sees them.
"""
import sciris as sc
import hpvsim as hpv

import model as md


# Set by user before running
to_run = [
    'run_calibration',   # uncomment to RUN (VM only)
    # 'plot_calibration',     # uncomment to PLOT/extract (local)
]
debug = False
do_save = True
n_trials = [1000, 2][debug]
n_workers = [64, 2][debug]

DATA = [
    'data/kazakhstan_cancer_cases.csv',
    'data/kazakhstan_asr_cancer_incidence.csv',
]


def make_calib_pars():
    """Nested [best, low, high, step] specs for each calibration parameter."""
    pars = dict(
        beta=[0.2, 0.1, 0.34],
        m_cross_layer=[0.76, 0.34, 0.99],
        f_cross_layer=[0.34, 0.19, 0.94],
        network=dict(
            m_partners_casual=[0.2, 0.1, 0.6],
            f_partners_casual=[0.2, 0.1, 0.6],
        ),
        cross_immunity=dict(rel_sev=dict(loc=[1.0, 0.5, 1.5])),
    )
    for g in ['hi5', 'ohr']:
        pars[g] = dict(
            cancer_fn=dict(transform_prob=[1.5e-3, 0.5e-3, 2.5e-3]),
            cin_fn=dict(k=[0.15, 0.1, 0.25]),
            dur_cin=dict(mean=[4.5, 3.5, 5.5], std=[20, 16, 24]),
        )
    return pars


def run_calib(n_trials=None, n_workers=None, do_plot=False, do_save=True, filestem=''):
    calib = hpv.Calibration(
        md.make_sim(debug=debug), calib_pars=make_calib_pars(),
        data=DATA, total_trials=n_trials, n_workers=n_workers,
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
    return calib


def load_calib(do_plot=True, filestem=''):
    calib = sc.load(f'raw_results/kazakhstan_calib{filestem}.obj')
    if do_plot:
        fig = hpv.plot_calibration(calib)
        fig.savefig(f'figures/kazakhstan_calib{filestem}.png')
    sc.save(f'results/kazakhstan_pars{filestem}.obj', calib.best_pars)
    return calib


if __name__ == '__main__':
    T = sc.timer()
    if 'run_calibration' in to_run:
        run_calib(n_trials=n_trials, n_workers=n_workers, do_save=do_save)
    if 'plot_calibration' in to_run:
        load_calib(do_plot=True)
    T.toc('Done')
