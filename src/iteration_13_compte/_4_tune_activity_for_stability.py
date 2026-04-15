import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from brian2.units.allunits import pampere
from mpl_toolkits.axes_grid1 import make_axes_locatable

def to_spike_trains(spikemon: SpikeMonitor, neurons: list[int], sim_time):
    neurons = np.array(neurons)  # e.g. [3, 7, 12, 20]
    dt = spike_monitor.clock.dt
    dt_ms = dt / ms

    T = int(np.ceil(sim_time / dt))
    N = len(neurons)

    raster = np.zeros((N, T), dtype=np.uint8)
    # Spike data
    spikes_neuron_index = np.array(spikemon.i)
    t = np.array(spikemon.t / ms)


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

in_testing = True

################################
# Global Brian2 preferences
################################
prefs.codegen.target = "numpy"
defaultclock.dt = 0.1*ms

################################
# Network size
################################
#NE = 8000
#NI = 2000
NE = 800
NI = 200
N = NE + NI

################################
# Neuron parameters
################################
# Pyramidal cells
Cm_E = 0.5*nF
gL_E = 25*nS
EL_E = -70*mV
Vth_E = -50*mV
Vres_E = -60*mV
tau_ref_E = 2*ms

# Interneurons
Cm_I = 0.2*nF
gL_I = 20*nS
EL_I = -70*mV
Vth_I = -50*mV
Vres_I = -60*mV
tau_ref_I = 1*ms

################################
# Synaptic reversal potentials
################################
E_AMPA = 0*mV
E_NMDA = 0*mV
E_GABA = -70*mV

################################
# Synaptic time constants
################################
tau_AMPA = 2*ms
tau_NMDA = 100*ms
tau_GABA = 10*ms

################################
# External Poisson input
################################
nu_ext = 1800*Hz
gext_E = 3.1*nS
gext_I = 2.38*nS

################################
# Recurrent conductances (control set)
################################
GEE = 0.381*nS * (2048/NE)
#GEI = 0.292*nS * (2048/NE)
GEI = 0.3*nS * (2048/NE)
#GIE = 1.336*nS * (512/NI) -> unstable network like [1.63, 1.62, 1.635] => neurons fire at saturation
#GIE = 1.64*nS * (512/NI) ->  activity almost zero
#GIE = 0.3*nS * (512/NI)
GIE = 0.1*nS * (512/NI)
GII = 1*nS * (512/NI)

################################
# Connectivity footprint parameters
################################
Jp_EE = 1.62
sigma_EE = 18.0  # degrees

################################
# Neuron equations
################################
eqs = '''
dv/dt = ( -gL*(v-EL) - gAMPA*(v-E_AMPA)
          - gNMDA*(v-E_NMDA) - gGABA*(v-E_GABA) ) / Cm : volt (unless refractory)

dgAMPA/dt = -gAMPA/tau_AMPA : siemens
dgNMDA/dt = -gNMDA/tau_NMDA : siemens
dgGABA/dt = -gGABA/tau_GABA : siemens

I_AMPA = gAMPA*(v-E_AMPA): ampere
I_NMDA = gNMDA*(v-E_NMDA): ampere
I_GABA = gGABA*(v-E_GABA): ampere

gL : siemens
Cm : farad
EL : volt
theta: 1
'''

# interesting seeds [8 -> unstable, fires to saturation, 7 -> assync irregular with top 0.35 Hz, 4 -> assync irregular with top 0.5 Hz]
for seed in [8, 7, 4]:
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

    I.v = EL_I
    I.gL = gL_I
    I.Cm = Cm_I
    I.EL = EL_I

    ################################
    # Preferred cue angles
    ################################
    theta_E = np.linspace(0, 360, NE, endpoint=False)
    theta_I = np.linspace(0, 360, NI, endpoint=False)

    E.theta = theta_E
    I.theta = theta_I

    ################################
    # Connectivity footprint
    ################################
    def W(delta, Jp, sigma):
        return Jm + (Jp - Jm)*np.exp(-delta**2/(2*sigma**2))

    # Compute J-
    dtheta = np.linspace(-180, 180, 360)
    Jm = (360 - np.sum(np.exp(-dtheta**2/(2*sigma_EE**2)))) / 360
    Jm = max(Jm, 0.0)

    integral = (Jm + (Jp_EE - Jm) * np.sqrt(2 * np.pi) * sigma_EE / 360)
    print(f" delta to integral 1 is {1 - integral :.5f}")

    ################################
    # Synapses
    ################################
    # E → E (NMDA only, structured)
    SEE = Synapses(E, E,
        model="w: 1",
        on_pre='gNMDA += w*GEE'
    )

    SEE.connect(condition='i!=j')

    delta = np.abs(theta_E[:, None] - theta_E[None, :])
    delta = np.minimum(delta, 360-delta)

    W_EE = Jm + (Jp_EE - Jm)*np.exp(-delta**2/(2*sigma_EE**2))
    W_EE = W_EE / np.mean(W_EE)

    synaptic_weights = W_EE.flatten()
    # np.arange(0, NE) + 800*np.arange(0, NE)
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

    #SEI.connect(p=0.8)
    #SIE.connect(p=0.8)

    # I → I
    SII = Synapses(I, I, on_pre='gGABA += GII')
    SII.connect(condition='i!=j')

    ################################
    # External Poisson input
    ################################
    Pext_E = PoissonInput(E, 'gAMPA', N=1000, rate=1.8*Hz, weight=gext_E)
    Pext_I = PoissonInput(I, 'gAMPA', N=1000, rate=1.8*Hz, weight=gext_I)

    ################################
    # Monitors
    ################################
    interesting_neurons = [0, 50, 100]
    spike_monitor = SpikeMonitor(E)
    population_rate_monitor = PopulationRateMonitor(E)

    currents_monitor = StateMonitor(source=E, variables=["I_AMPA", "I_NMDA", "I_GABA"],
                                           record=True)

    ################################
    # Run simulation
    ################################
    runtime = 2*second
    #runtime=100*ms

    # seeds unstable [0, 1]  seeds stable []
    if in_testing:
        np.random.seed(seed)
        devices.device.seed(seed)

    run(runtime, report="text", profile=True)
    print(profiling_summary())

    ################################
    # Compute firing rates
    ################################
    bin_size = 10*ms
    bins = int(runtime/bin_size)

    rates = np.zeros((NE, bins))
    for i in range(NE):
        spikes = spike_monitor.spike_trains()[i] / ms
        counts, _ = np.histogram(spikes, bins=bins, range=(0, runtime / ms))
        rates[i] = counts / bin_size

    ################################
    # Sort neurons by preferred cue
    ################################
    order = np.argsort(theta_E)
    rates_sorted = rates[order]

    plot_compte(sim_time=runtime, population_rate_monitor=population_rate_monitor, spikes_monitor=spike_monitor, currents_monitor=currents_monitor)

'''
Interesting parameters combinations:

GEE = 0*nS * (2048/NE)
GEI = 0.3*nS * (2048/NE)
GIE = 0.1*nS * (512/NI)
GII = 1*nS * (512/NI)
=> assynnonous irregular
'''
'''
for g_IE, g_EE_NMDA in [(1.9, 0.84),]:
    for seed in [0, 1, 2, 3, 4, 5]:
        current = AnExampleExperiment(G_EE_AMPA=0, G_EE_NMDA=g_EE_NMDA, G_EI=0.292, G_IE=g_IE, G_II=1, NE=800, NI=200,
                                      label=f"Simulating a cue at 90 degrees. G_E_NMDA={g_EE_NMDA}, G_IE = {g_IE}", seed=seed)
        execute_compte_experiment(example=current, seed=seed)
'''