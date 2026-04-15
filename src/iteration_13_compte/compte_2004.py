from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def run_rimulation():
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1*ms

    NE = 8000
    NI = 2000
    N = NE + NI

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

    # reversals
    E_AMPA = 0*mV
    E_NMDA = 0*mV
    E_GABA = -70*mV

    tau_AMPA = 2*ms
    tau_NMDA = 100*ms
    tau_GABA = 10*ms

    # External Poisson input
    nu_ext = 1800*Hz
    gext_E = 3.1*nS
    gext_I = 2.38*nS

    # Recurrent conductances (control set)
    GEE = 0.381*nS * (2048/NE)
    GEI = 0.292*nS * (2048/NE)
    GIE = 1.336*nS * (512/NI)
    GII = 1.024*nS * (512/NI)

    # Connectivity footprint parameters
    Jp_EE = 1.62
    sigma_EE = 18.0  # degrees

    # Neuron equations
    eqs = '''
    dv/dt = ( -gL*(v-EL) - gAMPA*(v-E_AMPA)
              - gNMDA*(v-E_NMDA) - gGABA*(v-E_GABA) ) / Cm : volt (unless refractory)
    
    dgAMPA/dt = -gAMPA/tau_AMPA : siemens
    dgNMDA/dt = -gNMDA/tau_NMDA : siemens
    dgGABA/dt = -gGABA/tau_GABA : siemens
    
    gL : siemens
    Cm : farad
    EL : volt
    '''

    # Create neuron groups
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

    # Preferred cue angles
    theta_E = np.linspace(0, 360, NE, endpoint=False)
    theta_I = np.linspace(0, 360, NI, endpoint=False)

    E.theta = theta_E
    I.theta = theta_I

# Connectivity footprint
def W(delta, Jp, sigma):
    return Jm + (Jp - Jm)*np.exp(-delta**2/(2*sigma**2))

    # Compute J-
    dtheta = np.linspace(-180, 180, 360)
    Jm = (360 - np.sum(np.exp(-dtheta**2/(2*sigma_EE**2)))) / 360
    Jm = max(Jm, 0.0)

################################
# Synapses
################################
# E → E (NMDA only, structured)
SEE = Synapses(E, E,
    on_pre='gNMDA += w*GEE'
)

SEE.connect(condition='i!=j')

delta = np.abs(theta_E[:, None] - theta_E[None, :])
delta = np.minimum(delta, 360-delta)
W_EE = Jm + (Jp_EE - Jm)*np.exp(-delta**2/(2*sigma_EE**2))
W_EE /= np.mean(W_EE)

SEE.w = W_EE.flatten()

# E → I
SEI = Synapses(E, I, on_pre='gNMDA += GEI')
SEI.connect(p=1.0)

# I → E
SIE = Synapses(I, E, on_pre='gGABA += GIE')
SIE.connect(p=1.0)

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
spike_mon = SpikeMonitor(E)
rate_mon = PopulationRateMonitor(E)

################################
# Run simulation
################################
runtime = 2*second
run(runtime)

################################
# Compute firing rates
################################
bin_size = 10*ms
bins = int(runtime/bin_size)

rates = np.zeros((NE, bins))
for i in range(NE):
    spikes = spike_mon.spike_trains()[i]
    counts, _ = np.histogram(spikes, bins=bins, range=(0, runtime))
    rates[i] = counts / bin_size

################################
# Sort neurons by preferred cue
################################
order = np.argsort(theta_E)
rates_sorted = rates[order]

################################
# Plot bump activity
################################
plt.figure(figsize=(10,6))
plt.imshow(
    rates_sorted,
    aspect='auto',
    origin='lower',
    cmap='jet',
    extent=[0, runtime/second, 0, NE]
)
plt.colorbar(label='Firing rate (Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Neuron (sorted by cue)')
plt.title('Bump attractor dynamics')
plt.show()