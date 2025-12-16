from brian2 import *
from loguru import logger

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import \
    plot_simulation, sim_steady_state, SimulationResultsWithSteadyState

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

def sim_diffusion_process_with_up_down(experiment: Experiment):
    logger.debug("Simulating steady state for {}", experiment)
    steady_up_state_results = sim_steady_state(experiment, state=experiment.network_params.up_state)
    steady_down_state_results = sim_steady_state(experiment, state=experiment.network_params.down_state)
    logger.debug("Steady state simulation for {} done! Results are Up State = {}, Down State = {}", experiment,
                 steady_up_state_results, steady_down_state_results)

    simulation_results = __sim_diffusion_process_with_up_down(experiment)
    return SimulationResultsWithSteadyState(simulation_results = simulation_results,
                                            steady_up_state_results = steady_up_state_results,
                                            steady_down_state_results=steady_down_state_results)

def sim_and_plot_diffusion_process(experiment: Experiment) -> SimulationResultsWithSteadyState:
    simulation_results = sim_diffusion_process_with_up_down(experiment)
    plot_simulation(simulation_results)
    return simulation_results

def __sim_diffusion_process_with_up_down(experiment: Experiment) -> SimulationResults:
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

    E_ampa = experiment.synaptic_params.e_ampa
    E_gaba = experiment.synaptic_params.e_gaba
    E_nmda = experiment.synaptic_params.e_ampa

    MG_C = 1 * mmole  # extracellular magnesium concentration

    tau_ampa = experiment.synaptic_params.tau_ampa
    tau_gaba = experiment.synaptic_params.tau_gaba
    tau_nmda_rise = experiment.synaptic_params.tau_nmda_rise
    tau_nmda_decay = experiment.synaptic_params.tau_nmda_decay

    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    g_e_0_up = experiment.network_params.up_state.effective_timeconstant_estimation.mean_excitatory_conductance()
    g_i_0_up = experiment.network_params.up_state.effective_timeconstant_estimation.mean_inhibitory_conductance()
    x_0_up =  experiment.network_params.up_state.effective_timeconstant_estimation.mean_nmda_activation()

    g_e_0_down = experiment.network_params.down_state.effective_timeconstant_estimation.mean_excitatory_conductance()
    g_i_0_down = experiment.network_params.down_state.effective_timeconstant_estimation.mean_inhibitory_conductance()
    x_0_down = experiment.network_params.down_state.effective_timeconstant_estimation.mean_nmda_activation()

    single_neuron = NeuronGroup(1,
                                    model=experiment.diffusion_model,
                                    threshold="v >= theta",
                                    reset="v = V_r",
                                    refractory=experiment.neuron_params.tau_rp,
                                    method=experiment.integration_method)
    single_neuron.v[:] = experiment.neuron_params.E_leak
    single_neuron.up[:] = 1
    single_neuron.down[:] = 0

    v_monitor = StateMonitor(source=single_neuron,
                             variables=["v", "g_e", "g_i", "g_nmda", "x_nmda", "s_nmda"], record=True)

    rate_monitor = PopulationRateMonitor(single_neuron)
    spike_monitor = SpikeMonitor(single_neuron)
    g_monitor = StateMonitor(source=single_neuron,
                             variables=experiment.plot_params.recorded_g_s, record=True)
    internal_states_monitor = StateMonitor(source=single_neuron, variables=experiment.recorded_hidden_variables,
                                           record=True)
    currents_monitor = StateMonitor(source=single_neuron, variables=experiment.plot_params.recorded_currents,
                                    record=True)

    up_down_monitor = StateMonitor(source=single_neuron, variables=["up", "down"], record=True)

    diffusion_network = Network([single_neuron, v_monitor, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
                             currents_monitor, up_down_monitor])

    reporting = "text" if experiment.in_testing else None

    @network_operation(dt=500 * ms)
    def toggle_inputs(t):
        step = int(t / (0.5 * second))
        if step % 2 == 0:
            single_neuron.up = 1
            single_neuron.down = 0
            logger.debug("at {} we have up=1 and down=0", t / ms)
        else:
            single_neuron.up = 0
            single_neuron.down = 1
            logger.debug("at {} we have up=0 and down=1", t / ms)

    run(experiment.sim_time, report=reporting, report_period=1 * second)
    stop()
    result = SimulationResults(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor,
                             currents_monitor)
    return result

def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba/ nS:.2f}\,n\mathrm{{S}}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda/ nS:.2f}\,n\mathrm{{S}}$]"""