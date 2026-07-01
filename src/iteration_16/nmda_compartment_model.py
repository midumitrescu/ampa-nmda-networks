import math

from brian2 import start_scope, device, seed, defaultclock, NeuronGroup, nS, StateMonitor, run, Synapses, ms, mV, nA, \
    SpikeMonitor, second
import numpy as np
from matplotlib.lines import Line2D

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_16.model import ConductanceDiffusionSimulationConfig
from iteration_16.simulation import create_spike_source

from matplotlib import pyplot as plt

from utils import ExtendedDict

NMDA_COMPARTMENT_MODEL = """
dx_nmda/dt = -x_nmda / tau_nmda_rise : 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_nmda) : 1
last_presyn: integer
spikes_count: integer
"""

__MAGNESIUM_BLOCK_NONLINEARITY__ = '''
sigma_of_v = 1 / (1 + exp(-0.062 * v / mV) * (mg_concentration / 3.57)) : 1
'''

__NO_MAGNESIUM_BLOCK_NONLINEARITY__ = '''
sigma_of_v = 1 : 1
'''

__SOMA_MODEL__ = """
dv/dt = 1/C * (
    -g_L * (v - E_L)
    - i_ampa
    - i_gaba
    - i_nmda_total
) : volt


dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens

i_ampa = g_ampa * (v - e_ampa): amp
i_gaba = g_gaba * (v - e_gaba): amp

i_nmda_total : amp
"""

SOMA_MODEL = f'''
{__SOMA_MODEL__}
{__MAGNESIUM_BLOCK_NONLINEARITY__}
'''

SOMA_MODEL_NO_MG_BLOCK = f'''
{__SOMA_MODEL__}
{__NO_MAGNESIUM_BLOCK_NONLINEARITY__}
'''


def extract_nmda_presynaptic_spikes(nmda_state_monitor: StateMonitor):

    if nmda_state_monitor is None:
        return None, None

    #indexes_of_presyn_input = np.diff(nmda_state_monitor.last_presyn, axis=1) != 0
    #indexes_of_x_jumps = np.diff(nmda_state_monitor.x_nmda, axis=1) > 0
    indexes_of_spikes = np.diff(nmda_state_monitor.spikes_count, axis=1) != 0

    compartment_idx, time_idx = np.where(indexes_of_spikes > 0)
    times = nmda_state_monitor.t[time_idx] / ms
    return compartment_idx, times

def get_connectivity_matrix(nmda_input: Synapses):
    i = nmda_input.i[:]
    j = nmda_input.j[:]

    N_pre = nmda_input.source.N
    N_post = nmda_input.target.N

    W = np.zeros((N_pre, N_post))
    W[i, j] = 1
    return W

'''
data = {
            "config": config,
            "neuron_monitor": neuron_monitor,
            "nmda_monitor": nmda_monitor,
            "current_monitor": currents_monitor,
            "ampa_spikes": ampa_presyn_mon,
            "gaba_spikes": gaba_presyn_mon,
            "nmda_spikes": extract_nmda_presynaptic_spikes(nmda_monitor),
            "compartment_connectivity": get_connectivity_matrix(nmda_input)
        })
'''

class SimulationResults:
    def __init__(self, data):
        self.config = data["config"]

        # spike data
        self.ampa_spikes = SimulationResults.__extract_spikes__(data["ampa_spikes"], self.config, N=self.config.N_E)
        self.gaba_spikes = SimulationResults.__extract_spikes__(data["gaba_spikes"], self.config, N=self.config.N_I)

        # continuous monitors
        self.neuron_monitor = SimulationResults.__extract_neuron_observables__(data["neuron_monitor"])
        self.nmda_monitor = SimulationResults.__extract_nmda_monitor__(data["nmda_monitor"])
        self.current_monitor = SimulationResults.__extract_soma_currents(data["current_monitor"])

        # connectivity (pure numpy)
        self.compartment_connectivity = data["compartment_connectivity"]

        nmda_compartments, nmda_times = data["nmda_spikes"]
        self.nmda_spikes = ExtendedDict({
            "t": nmda_times,
            "compartments": nmda_compartments
        })

    @staticmethod
    def __extract_spikes__(spike_monitor: SpikeMonitor, config, N):
        sim_time = config.simulation_time / second

        if spike_monitor is None:
            return ExtendedDict({})
        return ExtendedDict({
            "t": np.array(spike_monitor.t / ms),
            "i": np.array(spike_monitor.i),
            "all_values": spike_monitor.all_values(),
            "num_spikes": spike_monitor.num_spikes,
            "mean_rate": spike_monitor.num_spikes / (sim_time * N),
        })

    @staticmethod
    def __extract_neuron_observables__(neuron_monitor: StateMonitor):

        if neuron_monitor is None:
            return ExtendedDict({})
        return ExtendedDict({
            "t": np.array(neuron_monitor.t / ms),
            "v": np.array(neuron_monitor.v / mV),
            "g_ampa": np.array(neuron_monitor.g_ampa / nS),
            "g_gaba": np.array(neuron_monitor.g_gaba / nS),
        })

    @staticmethod
    def __extract_nmda_monitor__(nmda_monitor: StateMonitor):

        if nmda_monitor is None:
            return ExtendedDict({})
        return ExtendedDict({
            "t": np.array(nmda_monitor.t / ms),
            "x_nmda": np.array(nmda_monitor.x_nmda),
            "s_nmda": np.array(nmda_monitor.s_nmda),
            "last_presyn": np.array(nmda_monitor.last_presyn),
            "spikes_count": np.array(nmda_monitor.spikes_count)
        })

    @staticmethod
    def __extract_soma_currents(soma_current_monitor: StateMonitor):
        if soma_current_monitor is None:
            return ExtendedDict({})

        return ExtendedDict({
            "t": np.array(soma_current_monitor.t / ms),
            "i_ampa": np.array(soma_current_monitor.i_ampa / nA),
            "i_gaba": np.array(soma_current_monitor.i_gaba / nA),
            "i_nmda_total": np.array(soma_current_monitor.i_nmda_total / nA),
        })

    def mean_variance_soma_v(self):
        return SimulationResults.__mean_var__(self.neuron_monitor.v[0])

    def mean_variance_soma_v_no_nmda(self):
        return SimulationResults.__mean_var__(self.neuron_monitor.v[1])

    @staticmethod
    def __mean_var__(v_array: np.ndarray):
        data_points = len(v_array)
        start_for_mean = int(data_points / 4)
        voltage_roi = v_array[start_for_mean:]
        return np.mean(voltage_roi), np.var(voltage_roi)


class NMDASimulationWangCompartments:

    @staticmethod
    def run(config: ConductanceDiffusionSimulationConfig, detailed_statistics=True):

        start_scope()

        if config.seed is not None:
            device.reinit()
            np.random.seed(config.seed)
            seed(config.seed)

        defaultclock.dt = config.dt

        neuron = NeuronGroup(
            2,
            SOMA_MODEL,
            method="euler",
            namespace={
                "C": config.membrane_capacitance,
                "mg_concentration": config.magnesium_concentration,

                "tau_ampa": config.tau_ampa,
                "tau_gaba": config.tau_gaba,

                "E_L": config.e_L,
                "e_ampa": config.e_ampa,
                "e_gaba": config.e_gaba,
                "e_nmda": config.e_nmda,

                "g_L": config.g_L,
            },
        )

        neuron.v = config.resting_voltage
        neuron.g_ampa = 0 * nS
        neuron.g_gaba = 0 * nS

        nmda_compartments = NeuronGroup(
            config.k_comp,
            NMDA_COMPARTMENT_MODEL,
            method="euler",
            namespace={
                "tau_nmda_rise": config.tau_nmda_rise,
                "tau_nmda_decay": config.tau_nmda_decay,
                "alpha_nmda": config.alpha_nmda,
                "e_nmda": config.e_nmda,
            },
        )

        nmda_compartments.s_nmda = 0
        nmda_compartments.x_nmda = 0
        nmda_compartments.last_presyn = -1
        nmda_compartments.spikes_count = 0

        ampa_source = create_spike_source(
            config.ampa_spike_times,
            config.r_e,
            N=config.N_E
        )

        gaba_source = create_spike_source(
            config.gaba_spike_times,
            config.r_i,
            N=config.N_I
        )

        excitatory_synapses = Synapses(
            ampa_source,
            neuron,
            on_pre="g_ampa += w_ampa",
            namespace={
                "w_ampa": config.w_ampa
            }
        )

        inhibitory_synapse = Synapses(
            gaba_source,
            neuron,
            on_pre="g_gaba += w_gaba",
            namespace={
                "w_gaba": config.w_gaba
            }
        )

        nmda_input = Synapses(
            ampa_source,
            nmda_compartments,
            on_pre='''
                x_nmda += w_x
                last_presyn = _presynaptic_idx
                spikes_count += 1
            ''',
            namespace={
                "w_x": config.w_x
            }
        )

        group_size = config.N_E // config.k_comp
        #nmda_input.connect('j == i // group_size')

        i = np.arange(config.N_E)
        j = np.minimum(i // group_size, config.k_comp - 1)

        nmda_input.connect(i=i, j=j)

        excitatory_synapses.connect()
        inhibitory_synapse.connect()

        nmda_to_soma = Synapses(
            nmda_compartments,
            neuron[0],
            """
            i_nmda_total_post =  g_nmda_max  * s_nmda_pre * sigma_of_v_post * (v_post - e_nmda): amp (summed)
            """,
            namespace={
                "g_nmda_max": config.get_g_nmda_max(),
                "e_nmda": config.e_nmda,
                "mg_concentration": config.magnesium_concentration
            }
        )

        nmda_to_soma.connect()
        neuron_monitor = StateMonitor(
            neuron,
            [
                "v",
                "g_ampa",
                "g_gaba"
            ],
            record=True,
        )

        currents_monitor, nmda_monitor, ampa_presyn_mon, gaba_presyn_mon, connectivity = None, None, None, None, None
        if detailed_statistics:
            currents_monitor = StateMonitor(
                neuron,
                [
                    "i_ampa",
                    "i_gaba",
                    "i_nmda_total",
                ],
                record=True,
            )

            nmda_monitor = StateMonitor(
                nmda_compartments,
                [
                    "s_nmda",
                    "x_nmda",
                    "last_presyn",
                    "spikes_count"
                ],
                record=True,
            )

            ampa_presyn_mon = SpikeMonitor(ampa_source)
            gaba_presyn_mon = SpikeMonitor(gaba_source)
            connectivity = get_connectivity_matrix(nmda_input)

        run(config.simulation_time)

        return SimulationResults(data = {
            "config": config,
            "neuron_monitor": neuron_monitor,
            "nmda_monitor": nmda_monitor,
            "current_monitor": currents_monitor,
            "ampa_spikes": ampa_presyn_mon,
            "gaba_spikes": gaba_presyn_mon,
            "nmda_spikes": extract_nmda_presynaptic_spikes(nmda_monitor),
            "compartment_connectivity": connectivity
        })

    @staticmethod
    def run_and_plot(config: ConductanceDiffusionSimulationConfig, detailed_statistics=True, title=None, testing=False):
        result = NMDASimulationWangCompartments.run(config, detailed_statistics=detailed_statistics)
        plot_nmda_compartments(result, title=title, testing=testing)
        return result

def plot_nmda_compartments(result: SimulationResults, title=None, testing=False):

    config = result.config
    neuron_monitor = result.neuron_monitor
    nmda_monitor = result.nmda_monitor
    current_monitor = result.current_monitor
    ampa_spikes = result.ampa_spikes
    gaba_spikes = result.gaba_spikes

    prepare_bigger_fonts()

    number_of_plots = 5 if testing else 2

    fig, axes = plt.subplots(
        number_of_plots,
        1,
        figsize=(14, 14),
        sharex=True,
    )

    t = neuron_monitor.t

    axes[0].plot(
        t,
        neuron_monitor.v[0],
        color="black",
        label=r"$\bar g_{\mathrm{nmda}}=$"f"{result.config.g_nmda_max / nS : .2f} (nS)"
    )

    axes[0].plot(
        t,
        result.neuron_monitor.v[1],
        label=r"$\bar g_{\mathrm{nmda}}=0$ (nS)",
        linestyle=":",
        linewidth=2,
        alpha=0.6
    )
    mean_vm_nmda, var_vm_nmda = result.mean_variance_soma_v()
    mean_vm_nmda_no_nmda, var_vm_nmda_no_nmda = result.mean_variance_soma_v_no_nmda()
    axes[0].set_ylabel("V (mV)")
    axes[0].set_title("Membrane voltage \n"r"$V_{m} = $"f"{mean_vm_nmda: .2f} (mV)"r", $V_{m, \mathrm{No NMDA}} = $"f"{mean_vm_nmda_no_nmda: .2f} (mV)"
                      r", $\sigma_{v}^2 = $"f"{var_vm_nmda: .2f} (mV)"r", $\sigma_{v, \mathrm{No NMDA}}^2 = $"f"{var_vm_nmda_no_nmda: .2f} (mV)"r", $\Delta V=$"f"{(mean_vm_nmda - mean_vm_nmda_no_nmda): .2f} mV")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        t,
        current_monitor.i_nmda_total[0],
        label="NMDA"
    )
    axes[1].plot(
        t,
        current_monitor.i_ampa[0],
        label="AMPA"
    )
    axes[1].plot(
        t,
        current_monitor.i_gaba[0],
        label = "GABA"
    )
    axes[1].set_ylabel("$I_{\mathrm{NMDA}}$ (nA)")
    axes[1].set_title("Currents at the soma")

    axes[1].legend()

    if testing:
        plotted_nmda_compartments = min(len(nmda_monitor.s_nmda), 10)
        for idx in range(plotted_nmda_compartments):
            axes[2].plot(
                nmda_monitor.t,
                nmda_monitor.s_nmda[idx],
                alpha=0.6,
            )

        axes[2].set_ylabel("$s_{\mathrm{NMDA}}$")
        axes[2].set_xlabel("Time (ms)")
        axes[2].set_title(f"{plotted_nmda_compartments} NMDA compartments")

        axes[3].scatter(
            ampa_spikes.t,
            ampa_spikes.i + config.N_I,
            color="red",
            marker=".",
            s=5,
            label="Excitatory (AMPA)"
        )

        # Inhibitory spikes
        axes[3].scatter(
            gaba_spikes.t,
            gaba_spikes.i,
            color="blue",
            marker=".",
            s=5,
            label="Inhibitory (GABA)"
        )

        exc_center = (config.N_E - 1) / 2
        inh_center = config.N_E + (config.N_I - 1) / 2

        axes[3].set_yticks([exc_center, inh_center])
        axes[3].set_yticklabels(["GABA", "AMPA"])

        axes[3].set_ylabel("Input")
        axes[3].set_xlabel("Time (ms)")
        axes[3].set_title("Presynaptic spike raster")
        axes[3].legend()

        scatter_plot_clusters = axes[4].scatter(
            result.nmda_spikes.t,
            result.nmda_spikes.compartments,
            c=result.nmda_spikes.compartments,
            s=5
        )
        '''
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color=scatter_plot_clusters.cmap(scatter_plot_clusters.norm(cluster)),
                label=f"Cluster {cluster}",
                markersize=6,
            )
            for cluster in np.unique(result.nmda_spikes.compartments)[:10]
        ]
    
        axes[4].legend(handles=handles)
        '''

        axes[4].set_ylabel("NMDA compartment")
        axes[4].set_xlabel("Time (ms)")
        axes[4].set_title("NMDA synaptic events per compartment")

        W = result.compartment_connectivity

    if title is None:
        title = f"Simulation with {result.config.k_comp} NMDA compartments"

    fig.suptitle(title)
    fig.tight_layout()
    show_plots_non_blocking()

    if testing:
        plt.imshow(W, aspect='auto', origin='lower')
        plt.xlabel("Compartment (post)")
        plt.ylabel("Presynaptic neuron")
        plt.title("Connectivity matrix")
        plt.colorbar(label="connection")
        show_plots_non_blocking()