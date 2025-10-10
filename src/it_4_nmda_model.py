from brian2 import *
from loguru import logger
from matplotlib import gridspec
from sympy.printing.pretty.pretty_symbology import line_width

'''
Simulation params
g: relative inhibitory to excitatory synaptic strength. g_I / g_E
C: Each neuron receives C randomly chosen connections from other neurons in the network, from which. C = epsilon N
C E = epsilon * N E 
C I = epsilon * N I 
C_ext: It also receives C_ext connections from excitatory neurons outside the network. 

Each neuron gets C_E, C_I, C_EXT connections

'''


class NetworkParams:
    KEY_G = "g"
    KEY_NU_THR = "nu_thr"
    KEY_NU_E_OVER_NU_THR = "nu_ext_over_nu_thr"

    KEY_N = "N"
    KEY_N_E = "N_E"
    KEY_N_I = "N_I"

    KEY_C_EXT = "C_ext"

    KEY_GAMMA = "GAMMA"
    KEY_EPSILON = "epsilon"

    def __init__(self, params):
        self.g = params.get(NetworkParams.KEY_G, 0)
        #self.nu_thr = params.get(NetworkParams.KEY_NU_THR, 0) * Hz
        self.N_E = params.get(NetworkParams.KEY_N_E, 10_000)
        self.gamma = params.get(NetworkParams.KEY_GAMMA, 0.25)
        self.epsilon = params.get(NetworkParams.KEY_EPSILON, 0.1)

        self.N_I = round(self.gamma * self.N_E)
        self.N = self.N_E + self.N_I

        self.C_E = int(self.epsilon * self.N_E)

        self.C_ext = params.get(NetworkParams.KEY_C_EXT, self.C_E)


class NeuronModelParams:

    C = 1 * ufarad * (cm ** -2)
    g_L = 0.00004 * siemens * (cm ** -2) # TODO: I used 0.004 S/cm^2. This one I used in the simulation for Larkumm lab. However, this results in a tau membrane of 0.2 ms, way too low.
    #tau = 20 * ms
    tau = C / g_L
    theta = -40 * mV
    V_r = -55 * mV
    E_leak = -65 * mV
    tau_rp = 2 * ms

    # synapse parameters
    J = 0.5 * mV
    D = 1.5 * ms

    def __init__(self, network_params: NetworkParams = NetworkParams):
        self.nu_thr = (self.theta - self.E_leak) / (self.J * network_params.C_E * self.tau)

        logger.info("Computed nu threshold = {}", self.nu_thr)


class PlotParams:
    KEY_PANEL = "panel"

    KEY_T_RANGE = "t_range"
    KEY_RATE_RANGE = "rate_range"
    KEY_VOLTAGE_RANGE = "voltage_range"
    KEY_RATE_TICK_STEP = "rate_tick_step"


    def __init__(self, params):
        self.panel = params.get(PlotParams.KEY_PANEL, "")
        self.t_range = params.get(PlotParams.KEY_T_RANGE, [0, 100])
        self.rate_range = params.get(PlotParams.KEY_RATE_RANGE, [0, 150])
        self.voltage_range = params.get(PlotParams.KEY_VOLTAGE_RANGE, None)

        self.rate_tick_step = params.get(PlotParams.KEY_RATE_TICK_STEP, 30)\



class Experiment:
    KEY_SIM_TIME = "sim_time"

    def __init__(self, params):
        self.sim_time = params.get(Experiment.KEY_SIM_TIME, params.get(PlotParams.KEY_T_RANGE, (0, 200))[1]) * ms
        self.network_params = NetworkParams(params)
        self.neuron_params = NeuronModelParams(self.network_params)
        self.plot_params = PlotParams(params)

        self.nu_ext_over_nu_thr = params.get(NetworkParams.KEY_NU_E_OVER_NU_THR, 1)
        self.nu_thr = self.neuron_params.nu_thr
        self.nu_ext = self.nu_ext_over_nu_thr * self.nu_thr

        self.mean_excitatory_input = self.neuron_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext
        self.mean_inhibitory_input = - self.network_params.g * self.neuron_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext


plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


def sim(experiment: Experiment, in_testing=True):
    """
    g --
    nu_ext_over_nu_thr -- ratio of external stimulus rate to threshold rate
    sim_time -- simulation time
    ax_spikes -- matplotlib axes to plot spikes on
    ax_rates -- matplotlib axes to plot rates on
    rate_tick_step -- step size for rate axis ticks
    """
    if in_testing:
        np.random.seed(0)
    defaultclock.dt = 0.05 * ms

    g = experiment.network_params.g
    C = experiment.neuron_params.C

    theta = experiment.neuron_params.theta
    g_L = experiment.neuron_params.g_L
    E_leak = experiment.neuron_params.E_leak
    V_r = experiment.neuron_params.V_r
    J = experiment.neuron_params.J

    g = 0
    #E_leak = -55 * mV
    #theta = -53 * mV

    neurons = NeuronGroup(experiment.network_params.N,
                          """
                          dv/dt = 1/C * (-g_L * (v-E_leak)): volt (unless refractory)                          """,
                          threshold="v >= theta",
                          reset="v = V_r",
                          refractory=experiment.neuron_params.tau_rp,
                          method="exact")
    neurons.v[:] = -65 * mV

    excitatory_neurons = neurons[:experiment.network_params.N_E]
    inhibitory_neurons = neurons[experiment.network_params.N_E:]

    exc_synapses = Synapses(excitatory_neurons, target=neurons, on_pre="v += J", delay=experiment.neuron_params.D)
    exc_synapses.connect(p=experiment.network_params.epsilon)

    inhib_synapses = Synapses(inhibitory_neurons, target=neurons, on_pre="v += -g*J", delay=experiment.neuron_params.D)
    inhib_synapses.connect(p=experiment.network_params.epsilon)

    external_poisson_input = PoissonInput(
        target=neurons, target_var="v", N=experiment.network_params.C_ext, rate=experiment.nu_ext,
        weight=experiment.neuron_params.J
    )

    rate_monitor = PopulationRateMonitor(neurons)

    # record from the first 50 excitatory neurons
    spike_monitor = SpikeMonitor(neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25])
    v_monitor = StateMonitor(source=neurons[experiment.network_params.N_E - 25: experiment.network_params.N_E + 25],
                             variables="v", record=True)

    run(experiment.sim_time, report='text')

    plot_simulation(experiment, rate_monitor,
                    spike_monitor, v_monitor)

    if experiment.sim_time > 1000 * ms:
        experiment.plot_params.t_range = [1000, min(experiment.sim_time/ms, 2000)]
        plot_simulation(experiment, rate_monitor, spike_monitor, v_monitor)

    '''
    cp = params.copy()
    cp.update({PlotParams.KEY_T_RANGE: [1000, 2000]})
    plot_simulation(cp, rate_monitor, spike_monitor, v_monitor)
    '''

    return rate_monitor, spike_monitor, v_monitor


def plot_simulation(experiment: Experiment, rate_monitor,
                    spike_monitor, v_monitor):
    rate_tick_step = experiment.plot_params.rate_tick_step
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(
        f''' {experiment.plot_params.panel}, N = {experiment.network_params.N}, $N_E = {experiment.network_params.N_E}$, $N_I = {experiment.network_params.N_I}$, $\gamma={experiment.network_params.gamma}$
    $\\nu_T = {experiment.nu_thr}$, $\\frac{{\\nu_E}}{{\\nu_T}} = {experiment.nu_ext_over_nu_thr: .2f}$ ''')

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2])
    raster_and_population = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[4, 1], hspace=0)
    voltage_examples = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1], hspace=.05)

    ax_spikes, ax_rates = raster_and_population.subplots(sharex="col")
    ax_spikes.plot(spike_monitor.t / ms, spike_monitor.i, "|")
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz)
    ax_spikes.set_yticks([])
    ax_rates.set_ylim(*experiment.plot_params.rate_range)

    ax_voltages = voltage_examples.subplots()

    ax_voltages.axhline(y=experiment.neuron_params.theta / ms, linestyle="dotted", linewidth="0.3", color="k", label="$\\theta$")
    if experiment.plot_params.voltage_range is not None:
        ax_voltages.set_ylim(*experiment.plot_params.voltage_range)


    for i in range(0, 2):
        ax_voltages.plot(v_monitor.t / ms, v_monitor[i].v / mV, label=f"Exc {i}")
        ax_voltages.plot(v_monitor.t / ms, v_monitor[25 + i].v / mV, label=f"Inh {25 + i}")

    for ax in [ax_spikes, ax_rates, ax_voltages]:
        ax.set_xlim(*experiment.plot_params.t_range)

    ax_voltages.legend(loc="best")
    ax_voltages.set_xlabel("t [ms]")
    ax_voltages.set_ylabel("v [mV]")
    ax_rates.set_yticks(
        np.arange(
            experiment.plot_params.rate_range[0], experiment.plot_params.rate_range[1] + rate_tick_step, rate_tick_step
        )
    )


test_parameters = {
    "D_1": {
        "g": 1,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_2": {
        "g": 2,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_3": {
        "g": 3,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_4": {
        "g": 4,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
    "D_5": {
        "g": 5,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [2800, 3000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    }
}

test_parameters = dict(map(
    lambda nu_e_over_nu_th: (f"Scan {nu_e_over_nu_th}", {
        PlotParams.KEY_PANEL: f"Scan {nu_e_over_nu_th}",
        NetworkParams.KEY_G: 5,
        NetworkParams.KEY_NU_E_OVER_NU_THR: nu_e_over_nu_th,
        PlotParams.KEY_T_RANGE: [0, 50],
        PlotParams.KEY_RATE_RANGE: [0, 250],
        PlotParams.KEY_RATE_TICK_STEP: 50}), np.arange(0.5, 100, step=2.5)
))

parameters = {
    "A": {
        "g": 3,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "A_1": {
        "g": 30,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "A_2": {
        "g": 100,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 6000],
        "rate_tick_step": 1000,
    },
    "B": {
        "g": 6,
        "nu_ext_over_nu_thr": 4,
        "t_range": [100, 200],
        "rate_range": [0, 400],
        "rate_tick_step": 100,
    },
    "C": {
        "g": 5,
        "nu_ext_over_nu_thr": 2,
        "t_range": [100, 200],
        "rate_range": [0, 200],
        "rate_tick_step": 50,
    },
    "D": {
        "g": 4.5,
        "nu_ext_over_nu_thr": 0.9,
        "t_range": [1800, 2000],
        "rate_range": [0, 250],
        "rate_tick_step": 50,
    },
}

if __name__ == "__main__":
    for panel, params in test_parameters.items():
        sim(Experiment(params))
        plt.show()
