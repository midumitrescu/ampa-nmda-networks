from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETER SETUP (Wang 2002 values)
# ============================================================================

# NMDA synapse parameters
tau_rise = 2.0 * ms  # NMDA rise time
tau_decay = 100.0 * ms  # NMDA decay time
alpha = 0.5 * kHz  # NMDA saturation rate

# Single spike timing
spike_time = 50 * ms
simulation_duration = 500 * ms

# ============================================================================
# CREATE SPIKE INPUT USING TIMEDARRAY
# ============================================================================

# Create a TimedArray with a single spike at 50ms
spike_times = [spike_time]  # Only one spike at 50ms

# Create a spike generator
spike_indices = [0] * len(spike_times)
spike_generator = SpikeGeneratorGroup(1, spike_indices, spike_times)

# ============================================================================
# STANDALONE NMDA SYNAPSE MODEL
# ============================================================================

# NMDA synapse equations only (no neuron model)
NMDA_synapse = NeuronGroup(1, '''
                           # NMDA receptor dynamics
                           ds/dt = -s / tau_NMDA_decay + alpha_NMDA * x * (1 - s) : 1
                           dx/dt = -x / tau_NMDA_rise : 1

                           # Parameters
                           tau_NMDA_rise : second
                           tau_NMDA_decay : second
                           alpha_NMDA : Hz
                           ''',
                           method='euler')

# Set NMDA parameters
NMDA_synapse.tau_NMDA_rise = tau_rise
NMDA_synapse.tau_NMDA_decay = tau_decay
NMDA_synapse.alpha_NMDA = alpha
NMDA_synapse.x = 0
NMDA_synapse.s = 0

# ============================================================================
# CONNECT SPIKE TO NMDA SYNAPSE
# ============================================================================

# Simple synapse that adds 1 to x when spike arrives
input_synapse = Synapses(spike_generator, NMDA_synapse,
                         on_pre='x_post += 1',
                         method='exact')
input_synapse.connect()


# ============================================================================
# ANALYTICAL SOLUTION FUNCTIONS
# ============================================================================

def s_analytical(t, t0, alpha, tau_rise, tau_decay):
    """
    Exact analytical solution for s(t) after a single spike at t0.
    """
    t_val = float(t / second)
    t0_val = float(t0 / second)
    tau_r_val = float(tau_rise / second)
    tau_d_val = float(tau_decay / second)
    alpha_val = float(alpha / Hz)

    dt_val = t_val - t0_val
    if dt_val < 0:
        return 0.0

    a = alpha_val * tau_r_val
    r = tau_r_val / tau_d_val

    # Integral using scipy
    from scipy.integrate import quad

    lower = a * np.exp(-dt_val / tau_r_val)
    upper = a

    def integrand(u):
        return u ** (-r) * np.exp(-u)

    integral_val, _ = quad(integrand, lower, upper, limit=100)

    return (a ** r) * np.exp(a * np.exp(-dt_val / tau_r_val) - dt_val / tau_d_val) * integral_val


def x_analytical(t, t0, tau_rise):
    """Exact solution for x(t) = exp(-(t-t0)/tau_rise) for t >= t0"""
    if t < t0:
        return 0.0
    return exp(-(t - t0) / tau_rise)


def steady_state_s(x_vals, alpha_val, tau_decay_val):
    """
    CORRECTED steady-state solution for ds/dt = 0:
    s = (α·τ_decay·x) / (1 + α·τ_decay·x)
    """
    numerator = alpha_val * tau_decay_val * x_vals
    return numerator / (1 + numerator)


# ============================================================================
# MONITORS
# ============================================================================

# Monitor NMDA variables with high time resolution
NMDA_monitor = StateMonitor(NMDA_synapse, ['s', 'x'], record=0, dt=0.1 * ms)
spike_monitor = SpikeMonitor(spike_generator)

# ============================================================================
# RUN SIMULATION
# ============================================================================

print("=" * 60)
print("NMDA SYNAPSE - SINGLE SPIKE AT 50ms")
print("=" * 60)
print(f"Spike time: {float(spike_time / ms)} ms")
print(f"tau_rise: {tau_rise}")
print(f"tau_decay: {tau_decay}")
print(f"alpha: {alpha}")
print(f"Product α·τ_decay = {float(alpha * tau_decay):.3f}")
print(f"Simulation duration: {float(simulation_duration / ms)} ms")
print("=" * 60)

run(simulation_duration)

print(f"\nSpike delivered at: {float(spike_monitor.t[0] / ms):.1f} ms")
print(f"Simulation complete.")

# ============================================================================
# ANALYTICAL SOLUTION
# ============================================================================

# Create time points for analytical solution
analytic_t = np.linspace(0, float(simulation_duration / ms), 2000)
s_analytic_vals = []
x_analytic_vals = []

for t_ms in analytic_t:
    t = t_ms * ms
    s_analytic_vals.append(s_analytical(t, spike_time, alpha, tau_rise, tau_decay))
    x_analytic_vals.append(float(x_analytical(t, spike_time, tau_rise)))

# ============================================================================
# SIMULATION RESULTS
# ============================================================================

sim_t = np.array(NMDA_monitor.t / ms)
sim_s = np.array(NMDA_monitor.s[0])
sim_x = np.array(NMDA_monitor.x[0])

# Find analytical values at simulation time points
s_analytic_at_sim = []
for t_ms in sim_t:
    t = t_ms * ms
    s_analytic_at_sim.append(s_analytical(t, spike_time, alpha, tau_rise, tau_decay))

s_analytic_at_sim = np.array(s_analytic_at_sim)

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)
print(f"Max error in s: {np.max(np.abs(sim_s - s_analytic_at_sim)):.2e}")
print(f"RMS error in s: {np.sqrt(np.mean((sim_s - s_analytic_at_sim) ** 2)):.2e}")

# Convert parameters to dimensionless for calculations
tau_r = float(tau_rise / second)
tau_d = float(tau_decay / second)
alpha_val = float(alpha / Hz)

# Check final values
print(f"\nFinal values at t = {sim_t[-1]:.1f} ms:")
print(f"  x = {sim_x[-1]:.6f}")
print(f"  s(simulation) = {sim_s[-1]:.6f}")
print(f"  s(analytical) = {s_analytic_at_sim[-1]:.6f}")
print(f"  Steady-state s for this x: {steady_state_s(sim_x[-1], alpha_val, tau_d):.6f}")

# ODE verification
check_idx = np.argmin(np.abs(sim_t - 400))
s_check = sim_s[check_idx]
x_check = sim_x[check_idx]
t_check = sim_t[check_idx]

# Theoretical ds/dt from ODE
ds_dt_theory = -s_check / tau_d + alpha_val * x_check * (1 - s_check)

# Numerical derivative
dt_sim = float(defaultclock.dt / second)
ds_dt_numeric = (sim_s[check_idx] - sim_s[check_idx - 1]) / dt_sim

print(f"\nODE check at t = {t_check:.1f} ms:")
print(f"  s = {s_check:.6f}, x = {x_check:.6f}")
print(f"  ds/dt (ODE):     {ds_dt_theory:.6e} /s")
print(f"  ds/dt (numeric): {ds_dt_numeric:.6e} /s")
print(f"  Difference:      {abs(ds_dt_theory - ds_dt_numeric) / abs(ds_dt_theory + 1e-12):.2%}")

# ============================================================================
# STEADY-STATE CURVE FOR PHASE PLOT (CORRECTED)
# ============================================================================

# Generate steady-state curve for phase plot
x_range = np.linspace(0, 1.0, 200)  # x from 0 to 1
s_steady = steady_state_s(x_range, alpha_val, tau_d)

print(f"\nSteady-state analysis:")
print(f"  α·τ_decay = {alpha_val * tau_d:.3f}")
print(f"  When x → 0: s_steady → 0")
print(f"  When x → 1: s_steady → {(alpha_val * tau_d) / (1 + alpha_val * tau_d):.3f}")
print(f"  When x = 0.5: s_steady = {(alpha_val * tau_d * 0.5) / (1 + alpha_val * tau_d * 0.5):.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: s(t) - comparison
ax1 = axes[0, 0]
ax1.plot(sim_t, sim_s, 'b-', linewidth=2, label='Simulation')
ax1.plot(analytic_t, s_analytic_vals, 'r--', linewidth=1.5, alpha=0.7, label='Analytical')
ax1.axvline(x=50, color='k', linestyle=':', alpha=0.5, label='Spike at 50ms')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('s(t)')
ax1.set_title('NMDA Synapse Activation (s)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 500])

# Plot 2: x(t)
ax2 = axes[0, 1]
ax2.plot(sim_t, sim_x, 'g-', linewidth=2, label='Simulation')
ax2.axvline(x=50, color='k', linestyle=':', alpha=0.5, label='Spike at 50ms')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('x(t)')
ax2.set_title('Auxiliary Variable (x)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 150])

# Plot 3: Phase plot s vs x
ax3 = axes[1, 0]
# Simulation trajectory
ax3.plot(sim_x, sim_s, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')

# Corrected steady-state curve
ax3.plot(x_range, s_steady, 'k-', linewidth=2,
         label='Steady-state: s = (ατ_decay x)/(1+ατ_decay x)')

# Mark key points
ax3.plot(sim_x[0], sim_s[0], 'go', markersize=8, label='Start (t=0)')
ax3.plot(sim_x[-1], sim_s[-1], 'ro', markersize=8, label=f'End (t={sim_t[-1]:.0f}ms)')

ax3.set_xlabel('x')
ax3.set_ylabel('s')
ax3.set_title('Phase Plot s vs x')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distance from steady-state
ax4 = axes[1, 1]
# Calculate steady-state value for each point in simulation
s_steady_at_sim = steady_state_s(sim_x, alpha_val, tau_d)
distance = np.abs(sim_s - s_steady_at_sim)
ax4.plot(sim_t, distance, 'purple', linewidth=1.5)
ax4.axvline(x=50, color='k', linestyle=':', alpha=0.5)
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('|s - s_steady(x)|')
ax4.set_title('Distance from Steady-State')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 500])
ax4.set_yscale('log')

plt.suptitle(f'Wang 2002 NMDA Dynamics: s_steady = (ατ_decay x)/(1+ατ_decay x), ατ_decay={alpha_val * tau_d:.3f}',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ============================================================================
# QUASI-STEADY-STATE ANALYSIS
# ============================================================================

# For each (x, s) point, check if ds/dt ≈ 0
print("\n" + "=" * 60)
print("QUASI-STEADY-STATE ANALYSIS")
print("=" * 60)

# Calculate ds/dt at each point using the ODE
ds_dt_vals = -sim_s / tau_d + alpha_val * sim_x * (1 - sim_s)

# Find when |ds/dt| < epsilon
epsilon = 1e-3  # Small threshold
quasi_steady = np.abs(ds_dt_vals) < epsilon

if np.any(quasi_steady):
    first_steady_idx = np.where(quasi_steady)[0][0]
    first_steady_time = sim_t[first_steady_idx]
    print(f"First time |ds/dt| < {epsilon}: {first_steady_time:.1f} ms")
    print(f"  s = {sim_s[first_steady_idx]:.6f}, x = {sim_x[first_steady_idx]:.6f}")
    print(f"  s_steady for this x = {s_steady_at_sim[first_steady_idx]:.6f}")
else:
    print(f"System never reaches |ds/dt| < {epsilon}")

# Final check
print(f"\nAt final time t = {sim_t[-1]:.1f} ms:")
print(f"  ds/dt = {ds_dt_vals[-1]:.6e} /s")
print(f"  s - s_steady = {sim_s[-1] - s_steady_at_sim[-1]:.6e}")

# Calculate the "effective time constant" for approaching steady-state
if sim_s[-1] > 0:
    # Approximate time constant from exponential fit
    # Find portion after peak (approximately exponential decay)
    peak_idx = np.argmax(sim_s)
    tail_t = sim_t[peak_idx:] - sim_t[peak_idx]
    tail_s = sim_s[peak_idx:]

    if len(tail_t) > 10:
        # Fit exponential decay: s(t) = A * exp(-t/τ_eff)
        log_s = np.log(tail_s + 1e-12)
        coeffs = np.polyfit(tail_t / 1000, log_s, 1)  # Convert ms to seconds
        tau_eff = -1 / coeffs[0] if coeffs[0] < 0 else np.inf
        print(f"\nEffective time constant from exponential fit: {tau_eff * 1000:.1f} ms")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)