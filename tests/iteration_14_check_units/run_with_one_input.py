import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from brian2 import StateMonitor, ms, mpl, start_scope, \
    defaultclock, kHz, mmole, NeuronGroup, Synapses, second, run, \
    SpikeGeneratorGroup, msecond
from loguru import logger

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults
from utils import ExtendedDict

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

class SimulationResultsWithDV(SimulationResults):

    def __init__(self, experiment: Experiment,
                 v_monitor: StateMonitor, g_monitor: StateMonitor, internal_states_monitor: StateMonitor,
                 currents_monitor: StateMonitor, synapses_active=(False, False, False), time_of_spike=200*msecond):
        super().__init__(experiment=experiment, rate_monitor=None, spike_monitor=None,
                 v_monitor=v_monitor, g_monitor=g_monitor, internal_states_monitor=internal_states_monitor,
                 currents_monitor= currents_monitor, mean_field_values=None)
        self.synapses_active = synapses_active
        self.time_of_spike = time_of_spike

        self.voltages = ExtendedDict({
            "t": self.voltages.t,
            "v": self.voltages.v[0]
        })

        self.dv = self.__compute_dv_s()

        logger.info("Simulation {} has achieved stationarity? {}", self.synapses_active, self.voltages.v[0] == self.voltages.v[-1])

    def __compute_dv_s(self):
        index_before_spike = int(self.time_of_spike / self.experiment.sim_clock) - 2
        base_v = self.voltages.v[index_before_spike]
        input_ampa, input_gaba, input_nmda = self.synapses_active
        if input_ampa or input_nmda:
            peak = np.max(self.voltages.v[index_before_spike:])
        else:
            peak = np.min(self.voltages.v[index_before_spike:])

        return abs(base_v - peak)


def simulate_with_one_presynaptic_spike(
        experiment: Experiment,
        synapse_active=(True, False, False)):
    experiment = experiment.with_properties({
        "t_range": [[0, 1000], [150, 400]],
        Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda"],
        Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_ampa", "I_gaba"],
        Experiment.KEY_G_S_TO_RECORD: ["g_e", "g_i", "g_nmda"]
    })

    start_scope()

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak

    g_ampa = experiment.synaptic_params.g_ampa
    g_gaba = experiment.synaptic_params.g_gaba
    g_nmda_max = experiment.synaptic_params.g_nmda
    g_x_nmda = experiment.synaptic_params.g_x_nmda

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
                                refractory=experiment.neuron_params.tau_rp,
                                method=experiment.integration_method)
    single_neuron.v[:] = E_leak

    synapses = [None] * 3

    one_presynaptic_input = SpikeGeneratorGroup(
        N=1,
        indices=[0],
        times=[200 * ms]
    )
    ampa_on, gaba_on, nmda_on = synapse_active

    if ampa_on:
        S_ampa = Synapses(
            one_presynaptic_input,
            single_neuron,
            on_pre='g_e += g_ampa',
            method=experiment.integration_method
        )
        S_ampa.connect()
        synapses[0] = S_ampa

    if gaba_on:
        S_gaba = Synapses(
            one_presynaptic_input,
            single_neuron,
            on_pre='g_i += g_gaba',
            method=experiment.integration_method
        )
        S_gaba.connect()
        synapses[1] = S_gaba

    if nmda_on:
        S_nmda = Synapses(
            one_presynaptic_input,
            single_neuron,
            on_pre='x_nmda += g_x_nmda',
            method=experiment.integration_method
        )
        S_nmda.connect()
        synapses[2] = S_nmda

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

    return SimulationResultsWithDV(experiment, v_monitor=v_monitor, g_monitor=g_monitor, internal_states_monitor=internal_states_monitor,
                                   currents_monitor=currents_monitor, synapses_active=synapse_active)


def plot_dv_simulation_in_one_time_range(experiment: Experiment, simulation_results: list[SimulationResultsWithDV], time_range=[150, 300]):
    prepare_bigger_fonts()

    with plt.rc_context({'lines.linewidth': 2.5}):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=(14, 12)
        )

        ampa_sim, gaba_sim, nmda_sim = simulation_results

        alpha = 0.6

        [plot_spike_time(ax=ax, t_spike_ms = ampa_sim.time_of_spike / msecond) for ax in axes.flatten()]

        axes[0, 0].set_title("NMDA hidden variables")
        axes[0, 0].set_ylabel("[unitless]")
        axes[0, 1].set_title("Conductances (AMPA/E, GABA/I, NMDA)")
        axes[0, 1].set_ylabel("[nS]")

        axes[1, 0].set_title(f"Currents (AMPA/E, GABA/I, NMDA) \n Total charge [fC]: AMPA={ampa_sim.currents.q_ampa[0] * 1E3:.2f}, GABA={gaba_sim.currents.q_gaba[0] * 1E3: .2f}, NMDA={nmda_sim.currents.q_nmda[0] * 1E3 : .2f}")
        axes[1, 0].set_ylabel("[nA]")

        axes[1, 1].set_title(
            f"Voltage traces (AMPA/E, GABA/I, NMDA) \n $\Delta$V [mV]: AMPA={ampa_sim.dv:.3f}, GABA={gaba_sim.dv: .3f}, NMDA={nmda_sim.dv : .3f}")
        axes[1, 1].set_ylabel("[mV]")


        index_start = int(time_range[0] * msecond /experiment.sim_clock)
        index_end = int(time_range[1] * msecond /experiment.sim_clock)
        axes[0, 0].plot(nmda_sim.internal_states_monitor.t[index_start:index_end], nmda_sim.internal_states_monitor.x_nmda[0][index_start:index_end], label="x_nmda", alpha=alpha)
        axes[0, 0].plot(nmda_sim.internal_states_monitor.t[index_start:index_end], nmda_sim.internal_states_monitor.s_nmda[0][index_start:index_end], label="s_nmda", alpha=alpha)

        axes[0, 1].plot(ampa_sim.g_s.t[index_start:index_end], ampa_sim.g_s.g_e[0][index_start:index_end], label="$g_e$", alpha=alpha)
        axes[0, 1].plot(gaba_sim.g_s.t[index_start:index_end], gaba_sim.g_s.g_i[0][index_start:index_end], label="$g_i$", alpha=alpha)
        axes[0, 1].plot(nmda_sim.g_s.t[index_start:index_end], nmda_sim.g_s.g_nmda[0][index_start:index_end], label="$g_\mathrm{NMDA}$", alpha=alpha)

        axes[1, 0].plot(ampa_sim.currents.t[index_start:index_end], ampa_sim.currents.I_ampa[0][index_start:index_end], label="$I_\mathrm{AMPA}$",
                        alpha=alpha)
        axes[1, 0].plot(gaba_sim.currents.t[index_start:index_end], gaba_sim.currents.I_gaba[0][index_start:index_end], label="$I_\mathrm{GABA}$",
                        alpha=alpha)
        axes[1, 0].plot(nmda_sim.currents.t[index_start:index_end], nmda_sim.currents.I_nmda[0][index_start:index_end],
                        label="$I_\mathrm{NMDA}$", alpha=alpha)

        axes[1, 1].plot(ampa_sim.voltages.t[index_start:index_end], ampa_sim.voltages.v[index_start:index_end],
                        label="$v_\mathrm{AMPA}$",
                        alpha=alpha)
        axes[1, 1].plot(gaba_sim.voltages.t[index_start:index_end], gaba_sim.voltages.v[index_start:index_end],
                        label="$v_\mathrm{GABA}$",
                        alpha=alpha)
        axes[1, 1].plot(nmda_sim.voltages.t[index_start:index_end], nmda_sim.voltages.v[index_start:index_end],
                        label="$v_\mathrm{NMDA}$", alpha=alpha)

        [ax.legend() for ax in axes.flatten()]

        fig.suptitle("Simulation of a single presynaptic spike")
        plt.tight_layout()
        show_plots_non_blocking()


def plot_spike_time(ax, t_spike_ms):
    ax.axvline(x=t_spike_ms, color='blue', linestyle='--', linewidth=2, label="$t_\mathrm{spike}$", alpha=0.4)


def plot_dv_simulation(experiment, simulation_results: list[SimulationResultsWithDV]):
    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_range in params_t_range:
            plot_dv_simulation_in_one_time_range(experiment = experiment, simulation_results=simulation_results, time_range=time_range)
    else:
        plot_dv_simulation_in_one_time_range(experiment = experiment, simulation_results=simulation_results, time_range=params_t_range)



def generate_title(experiment: Experiment):
    up_state_title = f"Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]" \
        if experiment.network_params.up_state is not None else ""
    down_state_title = f"Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]" \
        if experiment.network_params.down_state is not None else ""

    return fr"""{experiment.plot_params.panel}  
    {up_state_title}
    {down_state_title}    
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa:.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba:.2f}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda:.2f}$]"""
