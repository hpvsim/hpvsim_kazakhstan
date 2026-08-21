"""
Define the HPVsim simulation for Kazakhstan (hpvsim v3.1).
"""
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv


# Settings
LOCATION = 'kazakhstan'

def network_pars():
    """
    Kazakhstan-specific network pars to layer over ``hpv.NetworkPars`` defaults.
    
    Sexual debut age (lognormal). Medians pinned to 18 (female) / 19 (male),
    matching the 1999 Kazakhstan DHS (last DHS available; median AFS was 18.4
    for women per collaborator).
    """
    pars = dict(
        debut_f=ss.lognorm_ex(mean=18.0, std=2.68),
        debut_m=ss.lognorm_ex(mean=19.0, std=3.13),
        m_partners_marital=0.01,
        m_partners_casual=0.2,
        f_partners_marital=0.01,
        f_partners_casual=0.2,
    )

    # Age-band participation probs. Rows: [age bins], [female], [male].
    # Bins are lower edges. 
    _HI = 1.0 - 1e-10  # Upper bound slightly less than 1; avoids log(0) errors

    pars['layer_probs_marital'] = np.array([
        [ 0,   5,  10,       15,       20,       25,       30,       35,       40,       45,       50,       55,       60,       65,       70,       75],
        [ 0,   0,   0, 0.267906, 0.943287, 0.993676, 0.996950, 0.998554, 0.997980, 0.994445, 0.984994, 0.958994, 0.870400, 0.683594, 0.400305, 0.185494],  # female
        [ 0,   0,   0, 0.019850, 0.758525, 0.988141, 0.996898, 0.999963, 0.999714, 0.999896, 0.999989, 0.999986, 0.999793, 0.996094, 0.958994, 0.821494],  # male
    ])
    pars['layer_probs_casual'] = np.array([
        [ 0, 5, 10,  15,   20,   25,  30,   35,   40,   45,    50,  55,   60,    65,     70,   75],
        [ 0, 0,  0, _HI, 0.75, 0.75, 0.5, 0.25, 0.25, 0.25, 0.125, 0.1, 0.05, 0.025, 0.0125, 0.01],  # female
        [ 0, 0,  0, _HI,  _HI,  _HI, _HI,  _HI,  _HI, 0.75,  0.25, 0.25, 0.2,   0.1,   0.05, 0.05],  # male
    ])

    return pars


def make_sim(pars=None, debug=0, n_agents=None, dt=None, start=1960, stop=2100,
             genotypes=None, ms_agent_ratio=100,
             interventions=None, analyzers=None, seed=1):
    """Build the baseline Kazakhstan sim."""
    if n_agents is None:
        n_agents = [10_000, 1_000][debug]
    if dt is None:
        dt = [0.25, 1.0][debug]
    if genotypes is None:
        genotypes = [16, 18, 'hi5', 'ohr']
    pars = sc.mergedicts(network_pars(), pars)

    sim = hpv.Sim(
        location=LOCATION,
        n_agents=n_agents,
        dt=dt,
        start=start,
        stop=stop,
        genotypes=genotypes,
        ms_agent_ratio=ms_agent_ratio,
        interventions=interventions,
        analyzers=analyzers,
        rand_seed=seed,
        pars=pars,
    )
    return sim


def run_sim(pars=None, seed=1, do_save=False, do_shrink=True, **kwargs):
    sim = make_sim(pars=pars, seed=seed, **kwargs)
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