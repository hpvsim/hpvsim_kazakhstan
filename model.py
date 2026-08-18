"""
Define the HPVsim simulation for Kazakhstan (hpvsim v3.0).

v3 is a thin wrapper around Starsim: `hpv.Sim` takes keyword arguments (not a
pars dict positionally), `stop` replaces `end`, and demographics are pulled
automatically from UN WPP data for `location='kazakhstan'`. Sexual network
behaviour is ported from an uncalibrated hpvsim v2.2.6 script (debut, layer
probs, partner counts fitted to the Kazakhstan DHS).

PER-TIMESTEP vs ANNUAL. The v2.2.6 script (like all hpvsim versions before the
v2.3 dt-fix) calibrates layer_probs/cross_layer at dt=0.25, per timestep, not
per year. v3's SexualNetwork always treats these as ANNUAL probabilities
(``ss.prob(p, annual)``, converted internally via ``.to_prob(dt)``). Porting
the raw DHS-fitted numbers unconverted would make the network ~4x too sparse
at dt=0.25 -- see `_to_annual`. `debut` (an age, not a rate) and the partner
COUNT distributions (`m_partners`/`f_partners`, Poisson) are unaffected and
carry over as-is.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv
from hpvsim.data.country import _network_pars

LOCATION = 'kazakhstan'
_DT = 0.25  # timestep the v2.2.6 network pars were fitted/used at

# Sexual debut, DHS-fitted lognormal (mean, std in years) -- ported as-is;
# debut is an age, not a rate, so no dt conversion applies.
_KAZAKHSTAN_DEBUT = dict(
    f=dict(dist='lognormal', par1=21.20, par2=2.68),
    m=dict(dist='lognormal', par1=19.72, par2=3.13),
)

# Marital ('m') and casual ('c') partnership probabilities by age, fitted to
# Kazakhstan DHS data; PER-TIMESTEP (dt=0.25) as ported from the v2.2.6
# script -- annualized below before use. Rows: [age bins], [female], [male].
# Note the marked asymmetry: female casual participation is very low
# (<=0.044) vs. male (<=0.524) -- partnership formation is female-driven
# (matched to the smaller of male/female demand), so this makes the casual
# layer collapse to near-zero regardless of male demand. Marital participation
# is high for both sexes. This is a real feature of the DHS data, not a bug;
# calibration (m/f_cross_layer, m/f_partners.c) is the intended lever -- see
# run_calibration.py.
_KAZAKHSTAN_LAYER_PROBS_PT = dict(
    m=np.array([
        [0, 5, 10,   15,    20,    25,    30,    35,    40,    45,    50,    55,   60,   65,   70,   75],
        [0, 0,  0, 0.075, 0.512, 0.718, 0.765, 0.805, 0.788, 0.727, 0.65,  0.55,  0.4,  0.25, 0.12, 0.05],   # female
        [0, 0,  0, 0.005, 0.299, 0.67,  0.764, 0.922, 0.87,  0.899, 0.942, 0.939, 0.88, 0.75, 0.55, 0.35],   # male
    ]),
    c=np.array([
        [0, 5, 10,   15,    20,    25,    30,    35,    40,    45,   50,    55,    60,    65,    70,    75],
        [0, 0,  0, 0.002, 0.026, 0.030, 0.044, 0.022, 0.036, 0.030, 0.020, 0.010, 0.005, 0.002, 0.001, 0.001],  # female
        [0, 0,  0, 0.209, 0.524, 0.264, 0.119, 0.113, 0.057, 0.048, 0.040, 0.030, 0.020, 0.010, 0.005, 0.005],  # male
    ]),
)

_KAZAKHSTAN_M_PARTNERS = dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2))
_KAZAKHSTAN_F_PARTNERS = dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.2))


def _to_annual(p, dt=_DT):
    """Per-timestep probability -> annual: ``1 - (1 - p)**(1/dt)`` (v2 form)."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0 - 1e-10)
    return 1.0 - (1.0 - p) ** (1.0 / dt)


def _layer_probs_annual():
    out = {}
    for lkey, lp in _KAZAKHSTAN_LAYER_PROBS_PT.items():
        a = lp.copy().astype(float)
        a[1, :] = _to_annual(a[1, :])
        a[2, :] = _to_annual(a[2, :])
        out[lkey] = a
    return out


def kazakhstan_network_overrides():
    """Raw network overrides in ``_default_network_pars`` form (annual)."""
    return dict(
        debut=_KAZAKHSTAN_DEBUT,
        layer_probs=_layer_probs_annual(),
        m_partners=sc.dcp(_KAZAKHSTAN_M_PARTNERS),
        f_partners=sc.dcp(_KAZAKHSTAN_F_PARTNERS),
    )


def make_network(pars=None):
    """Build the Kazakhstan SexualNetwork, with optional overrides merged over
    the baseline (used by calibration to vary cross_layer/partners.c)."""
    overrides = sc.mergedicts(kazakhstan_network_overrides(), pars)
    return hpv.SexualNetwork(**_network_pars(LOCATION, pars=overrides))


def make_sim(debug=0, n_agents=None, dt=None, start=1960, stop=2100,
             genotypes=None, networks=None, ms_agent_ratio=100,
             interventions=None, analyzers=None, seed=1):
    """Build the baseline Kazakhstan sim."""
    if n_agents is None:
        n_agents = [10_000, 1_000][debug]
    if dt is None:
        dt = [0.25, 1.0][debug]
    if genotypes is None:
        genotypes = [16, 18, 'hi5', 'ohr']
    if networks is None:
        networks = [make_network()]

    sim = hpv.Sim(
        location=LOCATION,
        n_agents=n_agents,
        dt=dt,
        start=start,
        stop=stop,
        genotypes=genotypes,
        networks=networks,
        ms_agent_ratio=ms_agent_ratio,
        interventions=interventions,
        analyzers=analyzers,
        rand_seed=seed,
    )
    return sim


def run_sim(seed=1, do_save=False, do_shrink=True, **kwargs):
    sim = make_sim(seed=seed, **kwargs)
    sim.label = f'kazakhstan--{seed}'
    sim.run()
    if do_shrink:
        sim.shrink()
    if do_save:
        sim.save('results/kazakhstan.sim')
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim = run_sim(stop=2020)
    sim.plot()
    T.toc('Done')
