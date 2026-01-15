import unittest
import numpy as np

import brian2.devices.device
from brian2 import plt, mpl, start_scope, defaultclock, mmole, ms, kHz, NeuronGroup, mV, PoissonInput, \
    PopulationRateMonitor, SpikeMonitor, StateMonitor, run, second, Hz, uS
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Configuration import Experiment, NetworkParams, PlotParams
from Plotting import show_plots_non_blocking

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


def sim(experiment: Experiment):
    """
        g --
        nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
        sim_time -- simulation time
        ax_spikes -- matplotlib axes to plot spikes on
        ax_rates -- matplotlib axes to plot rates on
        rate_tick_step -- step size for rate axis ticks
        """
    start_scope()
    np.random.seed(0)
    brian2.devices.device.seed(0)

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r

    g_nmda_max = experiment.synaptic_params.g_nmda

    E_ampa = experiment.synaptic_params.e_ampa

    E_nmda = experiment.synaptic_params.e_ampa
    MG_C = 1 * mmole  # extracellular magnesium concentration
    tau_nmda_decay = 100 * ms
    tau_nmda_rise = 2 * ms
    alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates

    model = experiment.model
    neurons = NeuronGroup(1,
                          model=model,
                          threshold="v >= theta",
                          reset="v = V_r",
                          refractory=experiment.neuron_params.tau_rp,
                          method="euler")
    neurons.v[:] = -65 * mV

    external_poisson_input = PoissonInput(
        target=neurons, target_var="x_nmda", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight="1"
    )

    rate_monitor = PopulationRateMonitor(neurons)
    spike_monitor = SpikeMonitor(neurons)
    v_monitor = StateMonitor(source=neurons,
                             variables="v", record=True)

    g_monitor = StateMonitor(source=neurons,
                             variables=["g_nmda"], record=True)


    internal_states_monitor = StateMonitor(source=neurons, variables=experiment.recorded_hidden_variables, record=True)
    run(experiment.sim_time, report="text", report_period=1 * second)

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor

def plot(experiment: Experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor):
    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_slot in params_t_range:
            plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                              time_range=time_slot)
            plot_internal_states(experiment, internal_states_monitor, time_range=time_slot)
    else:
        plot_internal_states(experiment, internal_states_monitor, time_range=params_t_range)
        plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                          time_range=params_t_range)

def sim_and_plot(experiment: Experiment):
    rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor = sim(experiment)
    plot(experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor, internal_states_monitor)

    return rate_monitor, spike_monitor, v_monitor, g_monitor, internal_states_monitor

def plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor, time_range):
    if experiment.plot_params.show_raster_and_rate():
        fig = plt.figure(figsize=(10, 12))
        fig.suptitle(generate_title(experiment))

        outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2])
        plot_raster_and_rates(experiment, outer[0], rate_monitor, spike_monitor, time_range)
        plot_voltages_and_g_s(experiment, outer[1], g_monitor, spike_monitor, time_range, v_monitor)

        show_plots_non_blocking(show=True)

def plot_internal_states(experiment: Experiment, internal_states_monitor, time_range: tuple[int, int]):
    if experiment.plot_params.show_hidden_variables():
        fig, ax = plt.subplots(len(experiment.plot_params.recorded_hidden_variables), 1, sharex=True, figsize=[10, 8])

        if 1 is len(experiment.plot_params.recorded_hidden_variables):
            ax = [ax]
        neurons_to_plot = [(0, "excitatory")]
        for neuron_i, label in neurons_to_plot:

            for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
                index = hidden_var_plot_details['index']
                curve_to_plot = internal_states_monitor[neuron_i].__getattr__(hidden_var_name)
                start_index = int(time_range[0]  / experiment.sim_clock * ms)
                end_index = int(time_range[1]  / experiment.sim_clock * ms)
                min = np.min(curve_to_plot.data[start_index:end_index])
                max = np.max(curve_to_plot.data[start_index:end_index])

                if min is not None and min < 0:
                    min = min * 1.1
                if max is not None and max > 0:
                    max = max * 1.1

                ax[index].plot(internal_states_monitor.t / ms,
                               curve_to_plot,
                               label=f"Neuron {neuron_i} - {label}")
                ax[index].set_ylim(bottom = min, top = max )


        for hidden_var_name, hidden_var_plot_details in experiment.plot_params.create_hidden_variables_plots_grid().items():
            index = hidden_var_plot_details['index']
            title = hidden_var_plot_details['title']
            y_label = hidden_var_plot_details['y_label']
            ax[index].set_title(title)
            ax[index].set_ylabel(y_label)

        ax[-1].set_xlabel("t [ms]")

        ax[0].legend(loc="right")

        for current_ax in ax:
            current_ax.set_xlim(*time_range)

        fig.suptitle(f"{generate_title(experiment)} \n {neurons_to_plot}")
        fig.tight_layout()

        show_plots_non_blocking()

def plot_raster_and_rates(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    rate_to_plot = rate_monitor.smooth_rate(
        width=experiment.plot_params.smoothened_rate_width) / Hz if experiment.plot_params.plot_smoothened_rate else rate_monitor.rate / Hz
    ax_rates.plot(rate_monitor.t / ms, rate_to_plot)
    ax_spikes.set_yticks([])

    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)

    lims = [0, np.max(rate_to_plot[time_start:time_end]) * 1.1]
    ax_rates.set_ylim(lims)

def plot_voltages_and_g_s(experiment, grid_spec_mother, g_monitor, spike_monitor, time_range, v_monitor):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")
    '''
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    if not np.isnan(v_max_plot):
        ax_voltages.set_ylim([v_min_plot, v_max_plot])
    '''
    for i in [0]:
        plot_v_line(experiment, ax_voltages, v_monitor, spike_monitor, i)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    i = 0
    '''
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_e_syn, label=rf"$g_\mathrm{{ext}}$[{i}]", alpha=0.5)
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_e, label=rf"$g_I$[{i}]", alpha=0.5)
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_i, label=rf"$g_E$[{i}]", alpha=0.5)
    '''
    ax_g_s.plot(g_monitor.t / ms, g_monitor[i].g_nmda, label=rf"$g_\mathrm{{nmda}}$[{i}]", alpha=0.5)
    ax_g_s.legend(loc="best")

def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_monitor: StateMonitor, spike_monitor: SpikeMonitor,
                i: int) -> None:
    lines = ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV,
                             label=f"Neuron {i} - {'Exc' if i <= experiment.network_params.neurons_to_record else 'Inh'}",
                             lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    '''
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)
    '''
    ax_voltages.vlines(x=spike_times_current_neuron, ymin=-70, ymax=-35, color=color, linestyle="-.", label=f"Neuron {i} Spike Time", lw=0.8)

def generate_title(experiment: Experiment):
    return fr"""{experiment.plot_params.panel}
    Network: [N={experiment.network_params.N}, $N_E={experiment.network_params.N_E}$, $N_I={experiment.network_params.N_I}$, $\gamma={experiment.network_params.gamma}$, $\epsilon={experiment.network_params.epsilon}$]
    Input: [$\nu_T={experiment.nu_thr}$, $\frac{{\nu_E}}{{\nu_T}}={experiment.nu_ext_over_nu_thr:.2f}$, $\nu_E={experiment.nu_ext:.2f}$ Hz]
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa/ uS:.2f}\,\mu\mathrm{{S}}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba/ uS:.2f}\,\mu\mathrm{{S}}$, $g={experiment.network_params.g}$]"""

single_compartment_with_nmda_but_without_sigmoid = '''
dv/dt = 1/C * (- I_L - I_nmda): volt (unless refractory)
        
I_L = g_L * (v-E_leak): amp / meter ** 2
I_nmda = g_nmda * (v - E_nmda): amp / meter** 2

g_nmda = g_nmda_max * s_nmda: siemens / meter**2

ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
'''

'''
Test equations for NMDA model. Understand each component!
'''
class NMDAModelTestCases(unittest.TestCase):

    def test_configuration_is_parsed(self):
        config = {

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 100,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-3,
            NetworkParams.KEY_EPSILON: 0.,
            "g_nmda": 5e-07,

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_but_without_sigmoid,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda"],
            "record_N": 10,

            "t_range": [[0, 2000], [1500, 1520]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }
        object_under_test = Experiment(config)

        self.assertAlmostEqual(2000, object_under_test.neuron_params.nu_thr / Hz)
        self.assertAlmostEqual(10, object_under_test.nu_ext / Hz)


    def test_nmda_model_runs(self):
        config = {

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 100,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-3,
            NetworkParams.KEY_EPSILON: 0.,
            "g_nmda": 5e-07,

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_but_without_sigmoid,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda"],
            "record_N": 10,

            "t_range": [[0, 2000], [1500, 1520]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }

        sim(Experiment(config))

    def test_nmda_model_can_be_plotted(self):
        config = {

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 100,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 5 * 1e-3,
            NetworkParams.KEY_EPSILON: 0.,
            "g_nmda": 5e-6,

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_but_without_sigmoid,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "I_nmda", "alpha_x_t"],
            #Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["bla_bla_bar"],
            "record_N": 10,

            "t_range": [[0, 2000], [1500, 1520]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }

        sim_and_plot(Experiment(config))

    def test_nmda_moodel_with_less_input(self):
        config = {

            NetworkParams.KEY_N_E: 1,
            NetworkParams.KEY_C_EXT: 100,
            NetworkParams.KEY_NU_E_OVER_NU_THR: 0.1 * 5 * 1e-3,
            NetworkParams.KEY_EPSILON: 0.,
            "g_nmda": 8e-6,

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_but_without_sigmoid,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "s_drive", "one_minus_s_nmda", "I_nmda", "alpha_x_t"],
            "record_N": 10,

            "t_range": [[0, 2000], [1500, 1520]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        }

        sim_and_plot(Experiment(config))


if __name__ == '__main__':
    unittest.main()
