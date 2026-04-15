import itertools

import numpy as np
from brian2 import nS, SpikeMonitor, PopulationRateMonitor, StateMonitor, ms, prefs, defaultclock, nF, mV, start_scope, \
    NeuronGroup, Synapses, PoissonInput, second, devices, run
from joblib import delayed, Parallel

from iteration_13_compte.compte_utils_deserialized import plot_compte_results, run_cartezian_product
from iteration_13_compte.configs import AnExampleExperiment, CompteResults
from iteration_13_compte.compte_utils import plot_compte

sim_time = 10 * second


def execute_compte_experiment(example: AnExampleExperiment):
    in_testing = True

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    # NE = 8000
    # NI = 2000
    NE = example.NE
    NI = example.NI
    N = NE + NI

    # Neuron parameters
    # Pyramidal cells
    Cm_E = 0.5 * nF
    gL_E = 25 * nS
    EL_E = -70 * mV
    Vth_E = -50 * mV
    Vres_E = -60 * mV
    tau_ref_E = 2 * ms

    # Interneurons
    Cm_I = 0.2 * nF
    gL_I = 20 * nS
    EL_I = -70 * mV
    Vth_I = -50 * mV
    Vres_I = -60 * mV
    tau_ref_I = 1 * ms

    # Synaptic reversal potentials
    E_AMPA = 0 * mV
    E_NMDA = 0 * mV
    E_GABA = -70 * mV

    # Synaptic time constants
    tau_AMPA = 2 * ms
    tau_NMDA = 100 * ms
    tau_GABA = 10 * ms

    # Recurrent conductances (control set)
    GEE_AMPA = example.G_EE_AMPA
    GEE_NMDA = example.G_EE_NMDA
    GEI = example.GEI
    GIE = example.GIE
    GII = example.GII

    # Connectivity footprint parameters
    Jp_EE = 1.62
    sigma_EE = 18.0  # degrees

    # Neuron equations
    eqs = '''
    dv/dt = (-gL*(v-EL) - gAMPA*(v-E_AMPA) - gNMDA*(v-E_NMDA) - gGABA*(v-E_GABA)) / Cm : volt (unless refractory)

    dgAMPA/dt = -gAMPA/tau_AMPA : siemens
    dgAMPA_cue/dt = -gAMPA_cue/tau_AMPA : siemens
    dgNMDA/dt = -gNMDA/tau_NMDA : siemens
    dgGABA/dt = -gGABA/tau_GABA : siemens

    I_AMPA = gAMPA*(v-E_AMPA): ampere
    I_NMDA = gNMDA*(v-E_NMDA): ampere
    I_GABA = gGABA*(v-E_GABA): ampere

    gL : siemens
    Cm : farad
    EL : volt
    theta: 1
    '''

    start_scope()

    E = NeuronGroup(
        NE, eqs,
        threshold='v > Vth_E',
        reset='v = Vres_E',
        refractory=tau_ref_E,
        method='euler'
    )

    I = NeuronGroup(
        NI, eqs,
        threshold='v > Vth_I',
        reset='v = Vres_I',
        refractory=tau_ref_I,
        method='euler'
    )

    E.v = EL_E
    E.gL = gL_E
    E.Cm = Cm_E
    E.EL = EL_E

    I.v = EL_I
    I.gL = gL_I
    I.Cm = Cm_I
    I.EL = EL_I

    # Preferred cue angles
    theta_E = np.linspace(0, 360, NE, endpoint=False)
    theta_I = np.linspace(0, 360, NI, endpoint=False)
    dtheta = np.linspace(-180, 180, 360)

    E.theta = theta_E
    I.theta = theta_I

    G = np.exp(-dtheta ** 2 / (2 * sigma_EE ** 2))
    alpha = np.mean(G)  # ≈ sqrt(2π)σ / 360

    Jm = (1 - alpha * Jp_EE) / (1 - alpha)

    # E → E (NMDA only, structured)
    SEE = Synapses(E, E,
                   model="w: 1",
                   on_pre='''
       gNMDA += w*GEE_NMDA 
       gAMPA += w*GEE_AMPA
    ''')

    SEE.connect(condition='i!=j')

    delta = np.abs(theta_E[:, None] - theta_E[None, :])
    delta = np.minimum(delta, 360 - delta)

    W_EE = Jm + (Jp_EE - Jm) * np.exp(-delta ** 2 / (2 * sigma_EE ** 2))
    W_EE = W_EE / np.sum(W_EE, axis=1) * 360

    synaptic_weights = W_EE.flatten()
    excluded_diagonal_elements = (NE + 1) * np.arange(0, NE)
    mask = np.ones_like(synaptic_weights, dtype=np.bool)
    mask[excluded_diagonal_elements] = False

    SEE.w = synaptic_weights[mask]

    # E → I
    SEI = Synapses(E, I, on_pre='gNMDA += GEI')
    # I → E
    SIE = Synapses(I, E, on_pre='gGABA += GIE')
    SEI.connect(p=1.0)
    SIE.connect(p=1.0)

    # I → I
    SII = Synapses(I, I, on_pre='gGABA += GII')
    SII.connect(condition='i!=j')

    Pext_E = PoissonInput(E, 'gAMPA', N=example.N_ext, rate=example.nu_ext, weight=example.gext_E)
    Pext_I = PoissonInput(I, 'gAMPA', N=example.N_ext, rate=example.nu_ext, weight=example.gext_I)

    spike_monitor = SpikeMonitor(E)
    population_rate_monitor = PopulationRateMonitor(E)

    currents_monitor = StateMonitor(source=E, variables=["I_AMPA", "I_NMDA", "I_GABA"],
                                    record=True)
    if in_testing:
        np.random.seed(example.seed)
        devices.device.seed(example.seed)

    runtime = sim_time
    run(runtime, report="text", profile=True)

    compte_result = CompteResults(population_rate_monitor=population_rate_monitor, spikes_monitor=spike_monitor,
                                  currents_monitor=currents_monitor, example=example, sim_time=runtime / ms,
                                  dt=defaultclock.dt / ms)

    if interesting(compte_result):
        plot_compte_results(results=compte_result, theta_array_assignment=theta_E)

    return compte_result.stats()


default_experiment = AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0.7, G_EI=0.292, G_IE=1.64, G_II=1, NE=800, NI=200,
                                         label="Experiment showing unstable high activity")



current_prod = product([0.84], [1.625], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

unstable_exps = [AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0.84, G_EI=0.292, G_IE=1.823, G_II=1, NE=800, NI=200,
                                     label=f"No Cue. Look for stable dynamics. G_E_NMDA=0.84, G_IE = 1.823", seed=0)]

unstable_runs = [
    product([0.84], [1.608], [2, 4, 5, 6]),
    product([0.84], [1.61], [4, 5, 7, 8, 9]),
    product([0.84], [1.62], [2, 5]),
    product([0.84], [1.65], [5]),
]
runs_with_higher_background_rate = [
    product([0.84], [1.608], [0, 1, 3]),
    product([0.84], [1.61], [0, 2, 4]),
]

'''
scans: product([0.84], np.linspace(1.584, 1.608, num=6), [0, 1, 2]) i.e. scan g_IE i.e. increase inhibition for async irregular
1.608 still produces unstable dynamics. Seeds [2, 4, 5, 9]
other interesting 


G IE  1.6750 XXXX all stable?  False [3] for time 10*s
G IE  1.6833 XXXX all stable?  False [34] for time 10*s
G IE  1.6917 XXXX all stable?  False [7] for time 10*s
G IE  1.7000 XXXX all stable?  True!!
G IE  1.7100 XXXX all stable?  True
G IE  1.7200 XXXX all stable?  True
G IE  1.7300 XXXX all stable?  True
G IE  1.7400 XXXX all stable?  True
G IE  1.7500 XXXX all stable?  True
G IE  1.7600 XXXX all stable?  True
G IE  1.7700 XXXX all stable?  True
G IE  1.7800 XXXX all stable?  True
G IE  1.7900 XXXX all stable?  True
G IE  1.8000 
'''

def interesting(compte_result: CompteResults):
    return compte_result.extract_end_rates() > 100


if __name__ == "__main__":
    # run_prod(product([0.84], [1.625], [0]))
    for g_IE in np.linspace(1.7, 1.8, 11):
        print(f"G IE {g_IE: .4f}")
        run_cartezian_product(product([0.84], [g_IE], np.arange(0, 100)))