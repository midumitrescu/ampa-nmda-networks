import brian2.devices.device
from brian2 import *
from loguru import logger

from Configuration import Experiment
from iteration_5_nmda.network_with_nmda import wang_model


def simulate_with_1s_step_input(experiment: Experiment, in_testing=True, eq=wang_model, step_length=1*second):

    start_scope()
    if in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C
    w = experiment.synaptic_params.g_ampa

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba

    tau_ampa = 2 * ms
    tau_gaba = 2 * ms

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration
    tau_nmda_decay = 100 * ms
    tau_nmda_rise = 2 * ms
    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model_to_run = eq if experiment.model is None else experiment.model

    neurons = NeuronGroup(experiment.network_params.N,
                          model=model_to_run,
                          threshold="v >= theta",
                          reset="v = V_r",
                          refractory=experiment.neuron_params.tau_rp,
                          method="euler")
    neurons.v[:] = -65 * mV

    excitatory_neurons = neurons[:experiment.network_params.N_E]
    inhibitory_neurons = neurons[experiment.network_params.N_E:]

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="g_e += g_ampa",
                            delay=experiment.synaptic_params.D)
    exc_synapses.connect(p=experiment.network_params.epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="g_i += g_gaba",
                              delay=experiment.synaptic_params.D)
    inhib_synapses.connect(p=experiment.network_params.epsilon)

    nmda_synapses = Synapses(neurons, neurons, on_pre='x += w', method="euler")
    nmda_synapses.connect(p=experiment.network_params.epsilon)

    P_high = PoissonGroup(experiment.network_params.C_ext, rates=experiment.nu_ext)
    P_low = PoissonGroup(experiment.network_params.C_ext, rates=0.1 * experiment.nu_ext)
    S_high = Synapses(P_high, neurons, model="on: 1", on_pre='g_ext_ampa_post += on * g_ampa')
    S_low = Synapses(P_low, neurons, model = "on: 1", on_pre='g_ext_ampa_post += on * g_ampa')

    S_high.connect(p=experiment.network_params.epsilon)
    S_low.connect(i=list(S_high.i), j=list(S_high.j))

    S_high.on[:] = 1
    S_low.on[:] = 0

    external_population_tp_neurons_connectivity = np.vstack((S_high.i, S_high.j))
    logger.debug("Which external neurons connect to neuron # 1? {} ",
                 external_population_tp_neurons_connectivity[:, external_population_tp_neurons_connectivity[1, :] == 1])

    @network_operation(dt=100 * ms)
    def toggle_inputs(t):
        if t / step_length < 1:
            S_high.on[:] = 1
            S_low.on[:] = 0
            logger.debug("at {} we have high active and low inactive", t)
        else:
            S_high.on[:] = 0
            S_low.on[:] = 0
            logger.debug("at {} we have high inactive and low inactive", t)

    rate_monitor = PopulationRateMonitor(neurons)
    spike_monitor = SpikeMonitor(neurons)
    v_monitor = StateMonitor(source=neurons,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=neurons[
                                    experiment.network_params.N_E - experiment.network_params.neurons_to_record: experiment.network_params.N_E + experiment.network_params.neurons_to_record],
                             variables=["g_e", "g_i", "g_nmda", "g_ext_ampa"], record=True)
    internal_states_monitor = StateMonitor(source=neurons, variables=experiment.recorded_hidden_variables, record=True)

    run(experiment.sim_time)

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor