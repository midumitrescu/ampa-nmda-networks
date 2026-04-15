import brian2.devices.device
import numpy as np
from brian2 import SpikeMonitor, StateMonitor, seed, start_scope, \
    defaultclock, kHz, mmole, NeuronGroup, second, run, PoissonInput

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults


def simulate_and_record_essential_variables(experiment: Experiment)-> SimulationResults:

    if experiment.in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)
        seed(0)
        np.random.default_rng(0)

    start_scope()

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba
    g_nmda_max = experiment.synaptic_params.g_nmda
    g_x = experiment.synaptic_params.g_x_nmda

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = experiment.synaptic_params.MG_C  # extracellular magnesium concentration

    tau_ampa = experiment.synaptic_params.tau_ampa
    tau_gaba = experiment.synaptic_params.tau_gaba
    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model = experiment.model
    single_neuron = NeuronGroup(1,
                                model=model,
                                threshold="v >= theta",
                                reset="v = V_r",
                                refractory=experiment.neuron_params.tau_rp,
                                method=experiment.integration_method)
    single_neuron.v[:] = V_r

    order = [0, 1, 2, 3, 4, 5] if experiment.in_testing else [0] * 5

    P_upstate_exc = PoissonInput(target=single_neuron, target_var="g_e", N=experiment.network_params.up_state.N_E,
                                 rate=experiment.network_params.up_state.nu,
                                 weight=g_ampa, order=order[0])
    P_upstate_inh = PoissonInput(target=single_neuron, target_var="g_i", N=experiment.network_params.up_state.N_I,
                                 rate=experiment.network_params.up_state.nu,
                                 weight=g_gaba, order=order[1])
    P_upstate_nmda = PoissonInput(target=single_neuron, target_var="x_nmda",
                                  N=experiment.network_params.up_state.N_NMDA,
                                  rate=experiment.network_params.up_state.nu_nmda, weight=g_x, order=order[4])

    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)


    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=60 * second)

    return SimulationResults(experiment, None, spike_monitor, v_monitor, None, internal_states_monitor,
                             None)