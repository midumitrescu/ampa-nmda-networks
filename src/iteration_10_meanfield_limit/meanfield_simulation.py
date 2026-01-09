import numpy as np
from brian2 import plt, mpl, StateMonitor, mV, start_scope, defaultclock, mmole, kHz, NeuronGroup, run, \
    second, devices as brian2devices, seed, PoissonInput, Synapses, \
    PopulationRateMonitor, SpikeMonitor, siemens, Hz
from loguru import logger

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, SynapticParams, State
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    SimulationResults, MeanField
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import \
    SimulationResultsWithSteadyState, sim_steady_state, plot_raster_and_g_s_in_one_time_range, \
    plot_currents_in_one_time_range, plot_internal_states_in_one_time_range

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

def weak_mean_field(conductance, N, N_reference):
    res = conductance * N_reference / N
    return res

'''
g_ampa = weak_mean_field(experiment.synaptic_params.g_ampa, experiment, 2000)
g_gaba = weak_mean_field(experiment.synaptic_params.g_gaba, experiment, 2000)
g_x = weak_mean_field(1, experiment, 2000)
'''
def prepare_mean_field(experiment: Experiment, N=2000, N_reference=2000):
    state = {
        "N": N,
        "nu": experiment.network_params.up_state.nu / Hz,
        "nu_nmda": experiment.network_params.up_state.nu_nmda / Hz,

        State.KEY_GAMMA: experiment.network_params.up_state.gamma,
        State.KEY_OMEGA: experiment.network_params.up_state.omega,
    }

    result = experiment.with_properties(
        {
            SynapticParams.KEY_G_AMPA: weak_mean_field(experiment.synaptic_params.g_ampa / siemens, N, N_reference),
            SynapticParams.KEY_G_GABA: weak_mean_field(experiment.synaptic_params.g_gaba / siemens, N, N_reference),
            SynapticParams.KEY_X_NMDA: weak_mean_field(experiment.synaptic_params.g_x_nmda, N, N_reference),
            "up_state": state,
            "down_state": state
        })
    return result

def sim_and_plot_meanfield_with_upstate_and_steady_state(experiment: Experiment) -> SimulationResultsWithSteadyState:
    simulation_results = simulate_meanfield_with_up_state_and_steady_state(experiment)
    plot_simulation(simulation_results)
    return simulation_results


def simulate_meanfield_with_up_state_and_steady_state(experiment: Experiment):
    logger.debug("Simulating steady state for {}", experiment)
    steady_up_state_results = sim_steady_state(experiment, state=experiment.network_params.up_state)
    logger.debug("Steady state simulation for {} done! Results are Up State = {}", experiment,
                 steady_up_state_results)
    simulation_results = simulate_one_state_with_meanfield(experiment)
    return SimulationResultsWithSteadyState(simulation_results, steady_up_state_results, None)


def plot_simulation(simulation_results: SimulationResultsWithSteadyState):
    params_t_range = simulation_results.experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_range in params_t_range:
            plot_raster_and_g_s_in_one_time_range(simulation_results,
                                                  time_range=time_range)
            plot_currents_in_one_time_range(simulation_results, time_range=time_range)
            plot_internal_states_in_one_time_range(simulation_results, time_range=time_range)
    else:
        plot_raster_and_g_s_in_one_time_range(simulation_results,
                                              time_range=params_t_range)
        plot_currents_in_one_time_range(simulation_results, time_range=params_t_range)
        plot_internal_states_in_one_time_range(simulation_results, time_range=params_t_range)


def simulate_one_state_with_meanfield(experiment: Experiment):
    """
        g --
        nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
        sim_time -- simulation time
        ax_spikes -- matplotlib axes to plot spikes on
        ax_rates -- matplotlib axes to plot rates on
        rate_tick_step -- step size for rate axis ticks
        """
    if experiment.in_testing:
        np.random.seed(0)
        brian2devices.device.seed(0)
        seed(0)
        np.random.default_rng(0)

    start_scope()

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    theta = 100 * mV
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba
    g_x = experiment.synaptic_params.g_x_nmda
    g_nmda_max = experiment.synaptic_params.g_nmda

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration

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
    single_neuron.v[:] = -65 * mV

    order = [0, 1, 2, 3, 4, 5] if experiment.in_testing else [0] * 5

    P_upstate_exc = PoissonInput(target=single_neuron, target_var="g_e", N=experiment.network_params.up_state.N_E, rate=experiment.network_params.up_state.nu,
                                 weight=g_ampa, order=order[0])
    P_upstate_inh = PoissonInput(target=single_neuron, target_var="g_i", N=experiment.network_params.up_state.N_I, rate=experiment.network_params.up_state.nu,
                                 weight=g_gaba, order=order[1])
    P_upstate_nmda = PoissonInput(target=single_neuron, target_var="x_nmda", N=experiment.network_params.up_state.N_NMDA,
                                  rate=experiment.network_params.up_state.nu_nmda, weight=g_x, order=order[4])

    logger.debug("Poisson Input AMPA {}, Poisson Input GABA {}, Poisson Input NMDA {}",
                 (experiment.network_params.up_state.N_E, experiment.network_params.up_state.nu, g_ampa),
                 (experiment.network_params.up_state.N_I, experiment.network_params.up_state.nu, g_gaba),
                 (experiment.network_params.up_state.N_NMDA, experiment.network_params.up_state.nu_nmda, g_x))

    rate_monitor = PopulationRateMonitor(single_neuron)
    spike_monitor = SpikeMonitor(single_neuron)
    v_monitor = StateMonitor(source=single_neuron,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=single_neuron,
                             variables=experiment.plot_params.recorded_g_s, record=True)

    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=experiment.plot_params.recorded_currents,
                                    record=True)
    reporting = "text" if experiment.in_testing else None
    run(experiment.sim_time, report=reporting, report_period=1 * second)

    return SimulationResults(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
                             currents_monitor, mean_field_values=MeanField(g_ampa=g_ampa, g_gaba=g_gaba, g_x=g_x,
                                                                           N=experiment.network_params.up_state.N,
                                                                           N_reference=2000))


def __weak_mean_field(conductance, experiment, N_reference):
    return conductance * N_reference / experiment.network_params.up_state.N
