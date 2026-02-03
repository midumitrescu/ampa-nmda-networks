import matplotlib.pyplot as plt
import numpy as np
from brian2 import nS, SpikeMonitor, PopulationRateMonitor, StateMonitor, ms, msecond, prefs, defaultclock, nF, mV, Hz, \
    start_scope, NeuronGroup, Synapses, PoissonInput, second, devices, run, network_operation
from brian2.units.allunits import pampere
from mpl_toolkits.axes_grid1 import make_axes_locatable

sim_time = 10 * second

class AnExampleExperiment:

    def __init__(self, G_EE_AMPA, G_EE_NMDA, G_EI, G_IE, G_II, NE, NI, label, gext_E=3.1, gext_I=2.38, nu_ext = 1800, N_ext = 1000, NE_ref=2048, NI_ref=512, seed=None):
        self.G_EE_AMPA = G_EE_AMPA * (NE_ref / NE) * nS
        self.G_EE_NDMA = G_EE_NMDA * (NE_ref / NE) * nS
        self.GEI = G_EI * (NE_ref / NE) * nS
        self.GIE = G_IE * (NI_ref / NI) * nS
        self.GII = G_II * (NI_ref / NI) * nS

        self.gext_E = gext_E * nS
        self.gext_I = gext_I * nS

        self.NE = NE
        self.NI = NI

        self.label = label

        self.in_testing = seed is not None
        self.seed = seed

        self.nu_ext_total = nu_ext * Hz
        self.N_ext = N_ext
        self.nu_ext = self.nu_ext_total / self.N_ext


def add_cue_input(degree, group: NeuronGroup, Jp, Jm, example: AnExampleExperiment,
                  cue_delay=200 * ms, cue_duration=200 * ms, sigma=2, spread=10, cue_rate=50 * Hz):

    theta_E = np.linspace(0, 360, example.NE, endpoint=False)
    dtheta = np.abs((theta_E - degree + 180) % 360 - 180)
    mask = dtheta <= spread

    group_getting_cue = group[np.argwhere(mask)[0][0]:np.argwhere(mask)[-1][0]+1]

    cue_w = Jm + (Jp - Jm) * np.exp(-(dtheta[mask] ** 2) / (2 * sigma ** 2))
    cue_w = cue_w / np.sum(cue_w)

    # ---- Time-dependent Poisson rate ----
    dt = defaultclock.dt
    t_end = cue_delay + cue_duration + 1 * ms
    n_steps = int(t_end / dt)

    rates = np.zeros(n_steps) * Hz
    start = int(cue_delay / dt)
    stop = int((cue_delay + cue_duration) / dt)
    rates[start:stop] = cue_rate

    group_getting_cue.cue_gain = cue_w

    Pext_E = PoissonInput(
        group_getting_cue,
        target_var='gAMPA_cue',
        N=1000,
        weight=example.gext_E,
        rate=cue_rate,
    )

    @network_operation(dt=200 * ms)
    def cue_controller(t):
        if cue_delay <= t < cue_delay + cue_duration:
            Pext_E.active = True
            group_getting_cue.cue_on = 1
        else:
            Pext_E.active = False
            group_getting_cue.cue_on = 0

    return Pext_E, cue_controller


def compute_ordered_firing_rates(example: AnExampleExperiment, spikes_monitor: SpikeMonitor, runtime,
                                 theta_array_assignment):
    bin_size = 10 * ms
    bins = int(runtime / bin_size)

    rates = np.zeros((example.NE, bins))
    for i in range(example.NE):
        spikes = spikes_monitor.spike_trains()[i] / ms
        counts, _ = np.histogram(spikes, bins=bins, range=(0, runtime / ms))
        rates[i] = counts / bin_size

    order = np.argsort(theta_array_assignment)
    rates_sorted = rates[order]
    return rates_sorted


def plot_compte(sim_time, population_rate_monitor: PopulationRateMonitor, spikes_monitor: SpikeMonitor,
                currents_monitor: StateMonitor, theta_array_assignment, example: AnExampleExperiment):
    rates_sorted = compute_ordered_firing_rates(example=example,
                                                spikes_monitor=spikes_monitor,
                                                runtime=sim_time,
                                                theta_array_assignment=theta_array_assignment)

    fig, axs = plt.subplots(
        4, 1,
        figsize=(20, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2, 1, 1]}
    )

    im = axs[0].imshow(
        rates_sorted,
        aspect='auto',
        origin='lower',
        cmap='jet',
        extent=[0, sim_time / msecond, 0, example.NE]
    )

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Firing rate (Hz)")

    axs[0].set_ylabel('Neuron (sorted by cue)')
    axs[0].set_title('Bump attractor dynamics')

    neurons = [0, 200]
    neuron_labels = ["non-cue", "cue"]

    # --- 2) Raster plot ---
    axs[1].plot(
        spikes_monitor.t / ms,
        spikes_monitor.i,
        ".",
        markersize=1,
        color="blue"
    )

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Neuron index')

    raster = to_spike_trains(spikes_monitor, neurons=neurons, sim_time=sim_time)
    rates = raster_to_rates(raster, spikes_monitor.clock.dt)

    axs[2].plot(
        population_rate_monitor.t / ms,
        population_rate_monitor.smooth_rate(width=5 * ms),
        label="Population",
        color="black",
        linewidth=2
    )
    for neuron_index, rate in zip(neurons, rates):
        axs[2].plot(
            population_rate_monitor.t / ms,
            rate,
            label=f"Neuron {neuron_index}"
        )

    axs[2].set_ylabel('Rate (Hz)')
    axs[2].legend(loc="upper right")

    for neuron_index, neuron_label in zip(neurons, neuron_labels):
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_AMPA[neuron_index] / pampere,
            label=f"I AMPA, {neuron_index} - {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_NMDA[neuron_index] / pampere,
            label=f"I NMDA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_GABA[neuron_index] / pampere,
            label=f"I GABA, {neuron_index}- {neuron_label}",
            alpha=0.6
        )
        axs[3].plot(
            currents_monitor.t / ms,
            currents_monitor.I_AMPA_cue[neuron_index] / pampere,
            label=f"I Cue, {neuron_index}- {neuron_label}",
            alpha=0.6
        )

    axs[3].legend()
    fig.suptitle(f"Simulation {example.label} {example.seed if example.in_testing else ''}")

    plt.tight_layout()
    fig.show()


def to_spike_trains(spikes_monitor: SpikeMonitor, neurons: list[int], sim_time):
    neurons = np.array(neurons)  # e.g. [3, 7, 12, 20]
    dt = spikes_monitor.clock.dt
    dt_ms = dt / ms

    T = int(np.ceil(sim_time / dt))
    N = len(neurons)

    raster = np.zeros((N, T), dtype=np.uint8)
    # Spike data
    spikes_neuron_index = np.array(spikes_monitor.i)
    t = np.array(spikes_monitor.t / ms)

    # Convert spike times to indices
    time_idx = (t / dt_ms).astype(np.int32)

    # Map global neuron indices → local indices
    neuron_map = {nid: k for k, nid in enumerate(neurons)}

    # Mask spikes we care about
    mask = np.isin(spikes_neuron_index, neurons)

    # Local neuron indices
    local_i = np.fromiter(
        (neuron_map[n] for n in spikes_neuron_index[mask]),
        dtype=np.int32,
        count=np.sum(mask)
    )

    # Assign spikes
    raster[local_i, time_idx[mask]] = 1
    return raster

def raster_to_rates(raster: np.ndarray, dt: float):
    dt_ms = dt / ms
    W_ms = 50
    W = int(W_ms / dt_ms)

    kernel = np.ones(W, dtype=np.float32) / (W * dt_ms * 1e-3)

    from scipy.signal import fftconvolve

    rate = fftconvolve(
        raster,
        kernel[None, :],
        mode='same',
        axes=1
    )

    return rate

def execute_compte_experiment(example: AnExampleExperiment, seed: int = None):
    in_testing = True

    ################################
    # Global Brian2 preferences
    ################################
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    ################################
    # Network size
    ################################
    # NE = 8000
    # NI = 2000
    NE = example.NE
    NI = example.NI
    N = NE + NI

    ################################
    # Neuron parameters
    ################################
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

    ################################
    # Synaptic reversal potentials
    ################################
    E_AMPA = 0 * mV
    E_NMDA = 0 * mV
    E_GABA = -70 * mV

    ################################
    # Synaptic time constants
    ################################
    tau_AMPA = 2 * ms
    tau_NMDA = 100 * ms
    tau_GABA = 10 * ms

    ################################
    # Recurrent conductances (control set)
    ################################
    GEE_AMPA = example.G_EE_AMPA
    GEE_NMDA = example.G_EE_NDMA
    GEI = example.GEI
    GIE = example.GIE
    GII = example.GII

    ################################
    # Connectivity footprint parameters
    ################################
    Jp_EE = 1.62
    sigma_EE = 18.0  # degrees

    ################################
    # Neuron equations
    ################################
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

    # interesting seeds [8 -> unstable, fires to saturation, 7 -> assync irregular with top 0.35 Hz, 4 -> assync irregular with top 0.5 Hz]
    start_scope()
    ################################
    # Create neuron groups
    ################################
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

    # Assign parameters
    E.v = EL_E
    E.gL = gL_E
    E.Cm = Cm_E
    E.EL = EL_E
    E.cue_gain = 0

    I.v = EL_I
    I.gL = gL_I
    I.Cm = Cm_I
    I.EL = EL_I
    I.cue_gain = 0


    ################################
    # Preferred cue angles
    ################################
    theta_E = np.linspace(0, 360, NE, endpoint=False)
    theta_I = np.linspace(0, 360, NI, endpoint=False)
    dtheta = np.linspace(-180, 180, 360)

    E.theta = theta_E
    I.theta = theta_I

    G = np.exp(-dtheta ** 2 / (2 * sigma_EE ** 2))
    alpha = np.mean(G)  # ≈ sqrt(2π)σ / 360

    Jm = (1 - alpha * Jp_EE) / (1 - alpha)

    integral = (Jm + (Jp_EE - Jm) * np.sqrt(2 * np.pi) * sigma_EE / 360)
    print(f" delta to integral 1 is {1 - integral :.5f}")

    # E → E (NMDA only, structured)
    SEE = Synapses(E, E,
                   model="w: 1",
                   on_pre='''
       gNMDA += w*GEE_NMDA 
       gAMPA += w*GEE_AMPA
    ''')

    SEE.connect(condition='i!=j')

    delta = np.abs(theta_E[:, None] - theta_E[None, :])
    delta = np.minimum(delta, 360 - delta)

    W_EE = Jm + (Jp_EE - Jm) * np.exp(-delta ** 2 / (2 * sigma_EE ** 2))
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

    # mean excitatory conductance
    mean_excitatory_input = tau_AMPA * example.nu_ext_total * example.gext_E
    expected_reversal = (gL_E * EL_E + mean_excitatory_input * E_AMPA) / (gL_E + mean_excitatory_input)
    print(f"Expected reversal of E units {expected_reversal}")

    _, cue_controller = add_cue_input(degree=90, group=E, Jp=Jp_EE, Jm=Jm, example=example)
    cue_controller

    interesting_neurons = [0, 50, 100]
    spike_monitor = SpikeMonitor(E)
    population_rate_monitor = PopulationRateMonitor(E)

    currents_monitor = StateMonitor(source=E, variables=["I_AMPA", "I_NMDA", "I_GABA", "I_AMPA_cue"],
                                    record=True)

    # Run simulation
    runtime = sim_time

    # seeds unstable [0, 1]  seeds stable []
    if in_testing:
        np.random.seed(seed)
        devices.device.seed(seed)

    run(runtime, report="text", profile=True)
    #print(profiling_summary())

    plot_compte(sim_time=runtime, population_rate_monitor=population_rate_monitor, spikes_monitor=spike_monitor,
                currents_monitor=currents_monitor, theta_array_assignment=theta_E, example=example)

default_experiment = AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=0.7, G_EI=0.292, G_IE=1.64, G_II=1, NE=800, NI=200,
                                         label="Experiment showing unstable high activity")

from itertools import product
#for g_IE in np.linspace(1.75, 1.82, num=1):
for g_EE_NMDA, g_IE, seed in product([0.84], np.linspace(1.9, 2.1, 10), [0]):
    current = AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=g_EE_NMDA, G_EI=0.292, G_IE=g_IE, G_II=1, NE=800, NI=200,
                                  label=f"When does dynamics die off?. G_E_NMDA={g_EE_NMDA}, G_IE = {g_IE}", seed=seed)
    execute_compte_experiment(example=current, seed=seed)