import numpy as np
from brian2 import nS, SpikeMonitor, PopulationRateMonitor, StateMonitor, ms, prefs, defaultclock, nF, mV, start_scope, \
    NeuronGroup, Synapses, PoissonInput, second, devices, run, network_operation, Hz, Network

from iteration_13_compte.compte_utils_deserialized import plot_compte_results
from iteration_13_compte.configs import AnExampleExperiment, CompteResults, CueInfo, default_cue_info

sim_time = 3 * second

def add_cue_input(example: AnExampleExperiment, group: NeuronGroup, cue: CueInfo = default_cue_info):

    dtheta = np.abs((example.theta_E - cue.degree + 180) % 360 - 180)
    mask = dtheta <= cue.spread

    mask_start = np.argwhere(mask)[0][0]
    mask_end = np.argwhere(mask)[-1][0] + 1
    print(f"Mask: {mask_start}: {mask_end}")
    group_getting_cue = group[mask_start:mask_end]

    cue_w = example.Jm + (example.Jp - example.Jm) * np.exp(-(dtheta[mask] ** 2) / (2 * cue.sigma ** 2))
    cue_w = cue_w / np.sum(cue_w)

    group_getting_cue.cue_gain = cue_w
    print(f"Cue {cue.label}: {np.sum(cue_w)}")
    print(f"Whole neurons: {np.sum(group.cue_gain)}")

    Pext_E = PoissonInput(
        group_getting_cue,
        target_var='gAMPA_cue',
        N=1000,
        weight=example.gext_E,
        rate=cue.rate,
    )

    @network_operation(dt=200 * ms)
    def cue_controller(t):
        #print("====================================")
        if cue.delay <= t < cue.delay + cue.duration:
            Pext_E.active = True
            group_getting_cue.cue_on = 1
            #print(cue.label)
            #print(f"{Pext_E} at {t}: {list(Pext_E.group.indices)}: cue on {group[list(Pext_E.group.indices)].cue_on}, gain: {group[list(Pext_E.group.indices)].cue_gain}")
        else:
            Pext_E.active = False
            group_getting_cue.cue_on = 0
            #print(cue.label)
            #print(f"{Pext_E} at {t}: {list(Pext_E.group.indices)}: cue on {group[list(Pext_E.group.indices)].cue_on}, gain: {group[list(Pext_E.group.indices)].cue_gain}")

        #print("====================================")

    return Pext_E, cue_controller


def execute_compte_experiment(example: AnExampleExperiment):
    cue_1 = CueInfo(degree=90, delay=200 * ms, duration=200 * ms, sigma=2, spread=10, cue_rate=50 * Hz)
    cue_2 = CueInfo(degree=270, delay=600 * ms, duration=400 * ms, sigma=4, spread=20, cue_rate=1000 * Hz)

    compte_result = run_compte_experiment(example)

    #plot_compte_results(results=compte_result, theta_array_assignment=theta_E, cues = [cue_1])
    #plot_compte_results(results=compte_result, theta_array_assignment=theta_E, cues = [cue_2])
    plot_compte_results(results=compte_result)

    return compte_result


def run_compte_experiment(example: AnExampleExperiment):
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

    # Neuron equations
    eqs = '''
    dv/dt = ( -gL*(v-EL) - gAMPA*(v-E_AMPA) - I_AMPA_cue
              - gNMDA*(v-E_NMDA) - gGABA*(v-E_GABA) ) / Cm : volt (unless refractory)

    dgAMPA/dt = -gAMPA/tau_AMPA : siemens
    dgAMPA_cue/dt = -gAMPA_cue/tau_AMPA : siemens
    dgNMDA/dt = -gNMDA/tau_NMDA : siemens
    dgGABA/dt = -gGABA/tau_GABA : siemens

    I_AMPA = gAMPA*(v-E_AMPA): ampere
    I_NMDA = gNMDA*(v-E_NMDA): ampere
    I_GABA = gGABA*(v-E_GABA): ampere
    I_AMPA_cue = cue_on*cue_gain*gAMPA_cue*(v-E_AMPA): ampere

    gL : siemens
    Cm : farad
    EL : volt
    theta: 1
    cue_on: 1
    cue_gain : 1
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

    E.theta = example.theta_E
    I.theta = example.theta_I

    # E → E (NMDA only, structured)
    SEE = Synapses(E, E,
                   model="w: 1",
                   on_pre='''
       gNMDA += w*GEE_NMDA 
       gAMPA += w*GEE_AMPA
    ''')
    SEE.connect(condition='i!=j')
    delta = np.abs(example.theta_E[:, None] - example.theta_E[None, :])
    delta = np.minimum(delta, 360 - delta)

    W_EE = example.Jm + (example.Jp - example.Jm) * np.exp(-delta ** 2 / (2 * example.sigma_EE ** 2))
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

    #recall_cues = [add_cue_input(example=example, group=E, cue=cue) for cue in cues]

    vars_to_monitor = [E, I,
    SEE, SEI, SIE, SII,
    Pext_E, Pext_I]

    for cue in example.cues:
        ext_E, cue_controller = add_cue_input(group=E, example=example, cue=cue)
        vars_to_monitor.extend([ext_E, cue_controller])

    # first group: 178:223
    # second group 556:645
    #bla_1, cue_controller_1 = add_cue_input(group=E, example=example, cue=example.cues[0])
    # second group Mask: 178: 223
    spike_monitor = SpikeMonitor(E)
    population_rate_monitor = PopulationRateMonitor(E)
    #currents_monitor = StateMonitor(source=E, variables=["I_AMPA", "I_NMDA", "I_GABA", "I_AMPA_cue"],
    #                                record=True)
    currents_monitor = StateMonitor(source=E, variables=["I_AMPA_cue"],
                                    record=True)
    if in_testing:
        np.random.seed(example.seed)
        devices.device.seed(example.seed)
    runtime = sim_time

    vars_to_monitor.extend([spike_monitor, population_rate_monitor, currents_monitor])
    network = Network(vars_to_monitor)
    network.run(runtime, report="text", profile=True)
    #print(recall_cues)
    compte_result = CompteResults(population_rate_monitor=population_rate_monitor, spikes_monitor=spike_monitor,
                                  currents_monitor=currents_monitor, example=example, sim_time=runtime / ms,
                                  dt=defaultclock.dt / ms)
    return compte_result


def search_for_viable_parameter_range():
    execute_compte_experiment(
        AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0, G_EI=0, G_IE=0, G_II=0, NE=800, NI=200,
                            label="Compare subcritical Up for broader vs more compact stimulus"))

if __name__ == "__main__":
    '''
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=1, G_EE_NMDA=0.84, G_EI=0.292, G_IE=2, G_II=0.9, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus"))
                                
                                -> limit unstable, 500 Hz
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=0.84, G_EI=0.292, G_IE=2, G_II=0.9, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> some short, irregular firing
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1, G_EI=0.4, G_IE=1.9, G_II=0.9, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> no activity
    
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1.5, G_EI=0.4, G_IE=1.9, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) 
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1.5, G_EI=0.4, G_IE=1, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> limit unstable, 500 Hz
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1.2, G_EI=0.4, G_IE=1, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> limit unstable, 500 Hz
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1.1, G_EI=0.4, G_IE=1, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> limit unstable, 500 Hz
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1, G_EI=0.4, G_IE=1, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> dieing
                                
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1, G_EI=0.4, G_IE=0.9, G_II=0.5, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> limit unstable, 500 Hz
    execute_compte_experiment(AnExampleExperiment(G_EE_AMPA=0.5, G_EE_NMDA=1, G_EI=0.4, G_IE=0.9, G_II=0.6, NE=800, NI=200,
                                label="Compare subcritical Up for broader vs more compact stimulus")) -> limit unstable, 500 Hz
    '''
    search_for_viable_parameter_range()


