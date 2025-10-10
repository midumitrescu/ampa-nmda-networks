import brian2.devices.device
import matplotlib.pyplot as plt
from brian2 import *
from brian2 import seed
from loguru import logger
from matplotlib import gridspec

from Configuration import Experiment

'''
Simulation params
g: relative inhibitory to excitatory synaptic strength. g_I / g_E
C: Each neuron receives C randomly chosen connections from other neurons in the network, from which. C = epsilon N
C E = epsilon * N E 
C I = epsilon * N I 
C_ext: It also receives C_ext connections from excitatory neurons outside the network. 

Each neuron gets C_E, C_I, C_EXT connections

'''

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

default_model = """        
    dv/dt = 1/C * (-g_L * (v-E_leak) - g_e * (v-E_ampa) - g_i * (v-E_gaba)) : volt (unless refractory)  
    dg_e/dt = -g_e / tau_ampa : siemens / meter**2
    dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
"""


def sim_and_plot(experiment: Experiment, in_testing=True, eq=default_model):
    rate_monitor, spike_monitor, v_monitor, g_monitor = sim(experiment, in_testing, eq)
    plot_simulation(experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor)
    plot_psd_and_CVs(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor)
    plt.show(block=False)
    plt.close()
    return rate_monitor, spike_monitor, v_monitor, g_monitor


def smoothen_curve(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def compute_cvs(spike_monitor: SpikeMonitor):
    result = np.zeros(len(spike_monitor.spike_trains()))

    for index, spike_train in spike_monitor.spike_trains().items():
        if len(spike_train) > 2:
            isis_s = np.diff(spike_train)
            result[index] = np.std(isis_s) / np.mean(isis_s)

            if np.mean(isis_s) == 0 or np.std(isis_s) / ms > 1000 or np.isnan(result[index]):
                logger.debug("Detected mean == 0 at {}, std={}", index, np.std(isis_s))
            if np.std(isis_s) == 0:
                logger.debug("STD = 0 for {} but we had {} spikes", index, len(spike_train))
        else:
            result[index] = 0

    # args_with_nan = np.argwhere(np.isnan(result))
    result = result[~np.isnan(result)]
    return result[result != 0]

def compute_fft(experiment, rate_monitor):
    sampling_rate = int(second / experiment.sim_clock)
    # Perform the Fast Fourier Transform
    n = len(rate_monitor.t)

    #fft_result = np.fft.fft(extract_rate(experiment, rate_monitor))
    fft_result = np.fft.fft(rate_monitor.rate)
    fft_freq = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_magnitude = np.abs(fft_result)
    positive_frequencies = fft_freq[:n // 2]
    positive_magnitude = fft_magnitude[:n // 2]
    return positive_frequencies, positive_magnitude

def extract_rate(experiment: Experiment, rate_monitor: PopulationRateMonitor):
    if experiment.plot_params.plot_smoothened_rate:
        return rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width) / Hz
    else:
        return rate_monitor.rate / Hz


def sim(experiment: Experiment, in_testing=True, eq=default_model):
    """
    g --
    nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
    sim_time -- simulation time
    ax_spikes -- matplotlib axes to plot spikes on
    ax_rates -- matplotlib axes to plot rates on
    rate_tick_step -- step size for rate axis ticks
    """
    start_scope()
    if in_testing:
        np.random.seed(0)
        brian2.devices.device.seed(0)

    defaultclock.dt = experiment.sim_clock

    C = experiment.neuron_params.C

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

    # dv/dt = - 1/C * g_L * (v-E_leak) - 1/C * g_e * (v-E_ampa) - 1/C * g_i *  (v-E_gaba) : volt (unless refractory)
    # dv/dt = - 1/C * g_L * (v-E_leak) - 1/C * g_e * (v-E_ampa) - 1/C * g_i *  (v-E_gaba) : volt (unless refractory)
    # dg_e/dt = - 1/tau_ampa * g_e: siemens / meter**2

    neurons = NeuronGroup(experiment.network_params.N,
                          eq,
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

    external_poisson_input = PoissonInput(
        target=neurons, target_var="g_e", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight=experiment.synaptic_params.g_ampa
    )

    rate_monitor = PopulationRateMonitor(neurons)
    spike_monitor = SpikeMonitor(neurons)
    v_monitor = StateMonitor(source=neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25],
                             variables="v", record=True)

    g_monitor = StateMonitor(source=neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25],
                             variables=["g_e", "g_i"], record=True)

    run(experiment.sim_time)

    return rate_monitor, spike_monitor, v_monitor, g_monitor


def plot_v_line(experiment: Experiment, ax_voltages: Axes, v_monitor: StateMonitor, spike_monitor: SpikeMonitor,
                i: int) -> None:
    lines = ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV, label=f"Neuron {i} - {'Exc' if i <= 25 else 'Inh'}",
                             lw=1)
    color = lines[0].get_color()
    spike_times_current_neuron = spike_monitor.all_values()['t'][i] / ms

    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)

    ax_voltages.vlines(x=spike_times_current_neuron, ymin=v_min_plot, ymax=v_max_plot, color=color, linestyle="-.",
                       label=f"Neuron {i} Spike Time", lw=0.8)


def plot_simulation(experiment: Experiment, rate_monitor,
                    spike_monitor, v_monitor, g_monitor):
    params_t_range = experiment.plot_params.t_range

    if isinstance(params_t_range[0], list):
        for time_slot in params_t_range:
            plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                              time_range=time_slot)
    else:
        plot_simulation_in_one_time_range(experiment, rate_monitor, spike_monitor, v_monitor, g_monitor,
                                          time_range=params_t_range)


def plot_simulation_in_one_time_range(experiment: Experiment, rate_monitor: PopulationRateMonitor,
                                      spike_monitor, v_monitor, g_monitor, time_range=[100, 200]):
    rate_tick_step = experiment.plot_params.rate_tick_step
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(experiment.gen_plot_title())

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2])
    plot_raster_and_rates(experiment, outer[0], rate_monitor, spike_monitor, time_range)
    plot_voltages_and_g_s(experiment, outer[1], g_monitor, spike_monitor, time_range, v_monitor)


def plot_voltages_and_g_s(experiment, grid_spec_mother, g_monitor, spike_monitor, time_range, v_monitor):
    voltage_and_g_s_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, hspace=0.8)
    ax_voltages, ax_g_s = voltage_and_g_s_examples.subplots(sharex="col")
    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k",
                        label="$\\theta$")
    v_min_plot, v_max_plot = find_v_min_and_v_max_for_plotting(experiment, v_monitor)
    ax_voltages.set_ylim([v_min_plot, v_max_plot])
    for i in [0, 2, 26, 28]:
        plot_v_line(experiment, ax_voltages, v_monitor, spike_monitor, i)
    for ax in [ax_voltages, ax_g_s]:
        ax.set_xlim(*time_range)
    ax_voltages.legend(loc="right")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[0].g_i, label="$g_i$[0]")
    ax_g_s.plot(g_monitor.t / ms, g_monitor[0].g_e, label="$g_e$[0]")
    ax_g_s.legend(loc="best")


def plot_raster_and_rates(experiment, grid_spec_mother, rate_monitor, spike_monitor, time_range):
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec_mother, height_ratios=[4, 1],
                                                             hspace=0)
    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    rate_to_plot = rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width) / Hz if experiment.plot_params.plot_smoothened_rate else rate_monitor.rate / Hz
    ax_rates.plot(rate_monitor.t / ms, rate_to_plot)
    ax_spikes.set_yticks([])
    # ax_rates.set_ylim(*experiment.plot_params.rate_range)
    # ax_rates.set_ylim([0, np.max(rate_monitor.rate[int(len(rate_monitor.t) / 2)] / Hz)])
    # ax_rates.set_ylim([0, np.max(rate_monitor.rate[int(len(rate_monitor.t) / 2)] / Hz)])
    for ax in [ax_spikes, ax_rates]:
        ax.set_xlim(*time_range)
    time_start = int(time_range[0] * ms / experiment.sim_clock)
    time_end = int(time_range[1] * ms / experiment.sim_clock)
    lims = [0, np.max(rate_to_plot[time_start:time_end])* 1.1]
    ax_rates.set_ylim(
        lims)


def find_v_min_and_v_max_for_plotting(experiment, v_monitor):
    if experiment.plot_params.voltage_range is not None:
        lim_down, lim_up = experiment.plot_params.voltage_range
    else:
        lim_down, lim_up = np.min(v_monitor.v) / mvolt, np.max(v_monitor.v) / mvolt
    return lim_down, lim_up


def plot_psd_and_CVs(experiment: Experiment, rate_monitor,
                     spike_monitor, v_monitor, g_monitor):
    sampling_rate = int(second / experiment.sim_clock)

    # Perform the Fast Fourier Transform
    n = len(rate_monitor.t)
    fft_result = np.fft.fft(extract_rate(experiment, rate_monitor))
    fft_freq = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_magnitude = np.abs(fft_result)

    positive_frequencies = fft_freq[:n // 2]
    positive_magnitude = fft_magnitude[:n // 2]

    smoothened_magnitude = smoothen_curve(positive_magnitude, window_size=250)

    fig, (ax_fft, ax_cvs) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(experiment.gen_plot_title())

    dominant_frequencies = positive_frequencies[np.argsort(positive_magnitude)[-5:]]  # top 5 frequencies
    logger.debug("Dominant frequencies: {}", dominant_frequencies)

    cut_off_freq = 1000
    ax_fft.plot(positive_frequencies[positive_frequencies < cut_off_freq], smoothened_magnitude[positive_frequencies < cut_off_freq])
    dominant_frequency = dominant_frequencies[-2]
    ax_fft.set_title(f"Frequency Spectrum \n Dominant frequency {dominant_frequency :.2f} Hz ")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    cvs = compute_cvs(spike_monitor)

    ax_cvs.set_title("CVs")
    ax_cvs.set_xlabel("CV")
    ax_cvs.set_ylabel("Density")
    ax_cvs.hist(cvs, bins=50, density=True)

    # fig.show()
    if len(cvs) > 0:
        logger.debug(f"Information regarding CVs: min={np.min(cvs)}, max={np.max(cvs)}, average={np.average(cvs)}")
        counts, bins = np.histogram(cvs, bins=50, density=True)
        bin_widths = np.diff(bins)
        area = np.sum(counts * bin_widths)
        logger.debug("Estimated area under the histogram: {}", area)

    fig.tight_layout()
