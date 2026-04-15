from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETER SETUP
# ============================================================================

tau_rise = 2.0 * ms
tau_decay = 100.0 * ms
alpha = 0.5 * kHz

spike_time = 50 * ms
simulation_duration = 600 * ms

# ============================================================================
# SETUP SIMULATION
# ============================================================================

# Spike generator
spike_times = [spike_time]
spike_indices = [0] * len(spike_times)
spike_generator = SpikeGeneratorGroup(1, spike_indices, spike_times)

# NMDA synapse
NMDA_synapse = NeuronGroup(1, '''
                           ds/dt = -s / tau_NMDA_decay + alpha_NMDA * x * (1 - s) : 1
                           dx/dt = -x / tau_NMDA_rise : 1
                           tau_NMDA_rise : second
                           tau_NMDA_decay : second
                           alpha_NMDA : Hz
                           ''',
                           method='euler')

NMDA_synapse.tau_NMDA_rise = tau_rise
NMDA_synapse.tau_NMDA_decay = tau_decay
NMDA_synapse.alpha_NMDA = alpha
NMDA_synapse.x = 0
NMDA_synapse.s = 0

# Connect spike
input_synapse = Synapses(spike_generator, NMDA_synapse,
                         on_pre='x_post += 1',
                         method='exact')
input_synapse.connect()

# ============================================================================
# MONITORS AND SIMULATION (WITH CORRECT DATA ACCESS)
# ============================================================================

# Monitor with high resolution
NMDA_monitor = StateMonitor(NMDA_synapse, ['s', 'x'], record=0, dt=0.01 * ms)
run(simulation_duration)

# CORRECT: Get data from monitor, not neuron group
sim_t = np.array(NMDA_monitor.t / ms)  # Time in ms
sim_s = np.array(NMDA_monitor.s[0])  # CORRECTED: From monitor
sim_x = np.array(NMDA_monitor.x[0])  # CORRECTED: From monitor

print("=" * 70)
print("DATA VALIDATION")
print("=" * 70)
print(f"Time array shape: {sim_t.shape}")
print(f"s array shape: {sim_s.shape}")
print(f"x array shape: {sim_x.shape}")
print(f"\nFirst few time points (ms): {sim_t[:5]}")
print(f"First few s values: {sim_s[:5]}")
print(f"First few x values: {sim_x[:5]}")
print(f"\nCheck spike region (t=49.9-50.1ms):")
spike_idx = np.argmin(np.abs(sim_t - 50))
for offset in [-2, -1, 0, 1, 2]:
    idx = spike_idx + offset
    if 0 <= idx < len(sim_t):
        print(f"  t={sim_t[idx]:.2f}ms: x={sim_x[idx]:.6f}, s={sim_s[idx]:.6f}")


# ============================================================================
# AUTOCORRELATION FUNCTIONS (CORRECTED)
# ============================================================================

def compute_autocorrelation(signal_vals, max_lag_ms=200):
    """
    Compute autocorrelation function.
    signal_vals: time series from monitor
    max_lag_ms: maximum lag to compute (in ms)
    """
    # Convert max lag to samples
    dt_ms = sim_t[1] - sim_t[0]  # Time step in ms
    max_lag_samples = int(max_lag_ms / dt_ms)

    # Remove mean for autocorrelation
    signal_mean = np.mean(signal_vals)
    signal_detrended = signal_vals - signal_mean

    # Compute autocorrelation using numpy
    n = len(signal_detrended)
    autocorr = np.correlate(signal_detrended, signal_detrended, mode='full')
    autocorr = autocorr[n - 1:n + max_lag_samples]  # Keep only positive lags

    # Normalize to 1 at lag 0
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    # Create lag array in ms
    lags_ms = np.arange(0, len(autocorr)) * dt_ms

    return lags_ms, autocorr


# Compute autocorrelations
max_lag = 400  # ms
x_lags, x_autocorr = compute_autocorrelation(sim_x, max_lag)
s_lags, s_autocorr = compute_autocorrelation(sim_s, max_lag)

# ============================================================================
# THEORETICAL PREDICTIONS
# ============================================================================

tau_rise_ms = float(tau_rise / ms)
tau_decay_ms = float(tau_decay / ms)

# Theoretical autocorrelations for comparison
x_theoretical = np.exp(-x_lags / tau_rise_ms)
s_theoretical = np.exp(-s_lags / tau_decay_ms)  # Approximation


# ============================================================================
# TIME CONSTANT ESTIMATION
# ============================================================================

def estimate_time_constant(lags, autocorr):
    """Estimate time constant from autocorrelation decay"""
    # Find where autocorr drops to 1/e
    target = 1 / np.e
    idx = np.where(autocorr <= target)[0]
    if len(idx) > 0:
        return lags[idx[0]]
    else:
        # Fit exponential decay to the curve
        mask = (lags > 0) & (autocorr > 0)
        if np.sum(mask) > 10:
            log_autocorr = np.log(autocorr[mask])
            coeffs = np.polyfit(lags[mask], log_autocorr, 1)
            return -1 / coeffs[0] if coeffs[0] < 0 else np.inf
        return np.nan


tau_x_est = estimate_time_constant(x_lags, x_autocorr)
tau_s_est = estimate_time_constant(s_lags, s_autocorr)

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 8))

# Plot 1: x(t) and s(t) time series
ax1 = axes[0]
ax1.plot(sim_t, sim_x, 'g-', linewidth=1.5, label='x(t)', alpha=0.8)
ax1.plot(sim_t, sim_s, 'b-', linewidth=1.5, label='s(t)', alpha=0.8)
ax1.axvline(x=50, color='k', linestyle=':', alpha=0.5, label='Spike at 50ms')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Value')
ax1.set_title('Time Series (from Monitor)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([40, 200])

# Plot 2: x autocorrelation
ax2 = axes[1]
ax2.plot(x_lags, x_autocorr, 'g-', linewidth=2, label='Autocorrelation')
ax2.plot(x_lags, x_theoretical, 'k--', linewidth=1.5, alpha=0.7,
         label=f'Theory: exp(-τ/{tau_rise_ms:.1f}ms)')
ax2.axhline(y=1 / np.e, color='r', linestyle=':', alpha=0.5, label='1/e level')
if not np.isnan(tau_x_est):
    ax2.axvline(x=tau_x_est, color='g', linestyle=':', alpha=0.5,
                label=f'τ_est={tau_x_est:.1f}ms')
ax2.set_xlabel('Lag τ (ms)')
ax2.set_ylabel('R_x(τ)')
ax2.set_title(f'x Autocorrelation\n(τ_rise={tau_rise_ms:.1f}ms)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, max_lag])

# Plot 3: s autocorrelation
ax3 = axes[2]
ax3.plot(s_lags, s_autocorr, 'b-', linewidth=2, label='Autocorrelation')
ax3.plot(s_lags, s_theoretical, 'k--', linewidth=1.5, alpha=0.7,
         label=f'Theory: exp(-τ/{tau_decay_ms:.1f}ms)')
ax3.axhline(y=1 / np.e, color='r', linestyle=':', alpha=0.5, label='1/e level')
if not np.isnan(tau_s_est):
    ax3.axvline(x=tau_s_est, color='b', linestyle=':', alpha=0.5,
                label=f'τ_est={tau_s_est:.1f}ms')
ax3.set_xlabel('Lag τ (ms)')
ax3.set_ylabel('R_s(τ)')
ax3.set_title(f's Autocorrelation\n(τ_decay={tau_decay_ms:.1f}ms)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, max_lag])

plt.suptitle(
    f'Autocorrelation Analysis (Corrected Data Access)\nτ_rise={tau_rise_ms:.1f}ms, τ_decay={tau_decay_ms:.1f}ms, α={float(alpha / Hz):.0f}Hz',
    fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ============================================================================
# DETAILED ANALYSIS WITH CORRECTED DATA
# ============================================================================

print("\n" + "=" * 70)
print("CORRECTED AUTOCORRELATION ANALYSIS")
print("=" * 70)

print(f"\nUsing data from StateMonitor (correct):")
print(f"  Number of time points: {len(sim_t)}")
print(f"  Time step: {sim_t[1] - sim_t[0]:.2f} ms")
print(f"  Time range: {sim_t[0]:.1f} to {sim_t[-1]:.1f} ms")

print(f"\nSignal statistics:")
print(f"  x: mean={np.mean(sim_x):.6f}, std={np.std(sim_x):.6f}, max={np.max(sim_x):.6f}")
print(f"  s: mean={np.mean(sim_s):.6f}, std={np.std(sim_s):.6f}, max={np.max(sim_s):.6f}")

print(f"\nAutocorrelation time constants:")
print(f"  x: τ_est = {tau_x_est:.2f} ms (expected: τ_rise = {tau_rise_ms:.1f} ms)")
print(f"  s: τ_est = {tau_s_est:.2f} ms (expected: ~τ_decay = {tau_decay_ms:.1f} ms)")

# Compute correlation with theoretical
if len(x_autocorr) == len(x_theoretical):
    corr_x = np.corrcoef(x_autocorr, x_theoretical)[0, 1]
else:
    corr_x = np.corrcoef(x_autocorr, x_theoretical[:len(x_autocorr)])[0, 1]

if len(s_autocorr) == len(s_theoretical):
    corr_s = np.corrcoef(s_autocorr, s_theoretical)[0, 1]
else:
    corr_s = np.corrcoef(s_autocorr, s_theoretical[:len(s_autocorr)])[0, 1]

print(f"\nCorrelation with theoretical exponential:")
print(f"  x: r = {corr_x:.4f}")
print(f"  s: r = {corr_s:.4f}")


# Integral timescale
def integral_timescale(lags, autocorr):
    """Compute ∫₀^∞ R(τ) dτ using trapezoidal rule"""
    return np.trapz(autocorr, x=lags)


int_time_x = integral_timescale(x_lags, x_autocorr)
int_time_s = integral_timescale(s_lags, s_autocorr)

print(f"\nIntegral timescale (area under autocorrelation):")
print(f"  x: {int_time_x:.2f} ms")
print(f"  s: {int_time_s:.2f} ms")

# For exponential R(τ)=exp(-τ/τ₀), integral timescale = τ₀
print(f"  Expected for perfect exponential: τ_rise={tau_rise_ms:.1f}ms, τ_decay={tau_decay_ms:.1f}ms")


# Half-life comparison
def find_half_life(lags, autocorr):
    """Find time when autocorr drops to 0.5"""
    idx = np.where(autocorr <= 0.5)[0]
    return lags[idx[0]] if len(idx) > 0 else np.nan


half_life_x = find_half_life(x_lags, x_autocorr)
half_life_s = find_half_life(s_lags, s_autocorr)

print(f"\nHalf-life (time to decay to 0.5):")
print(f"  x: {half_life_x:.2f} ms")
print(f"  s: {half_life_s:.2f} ms")
print(f"  Expected: τ×ln(2) = {tau_rise_ms * np.log(2):.2f}ms (x), {tau_decay_ms * np.log(2):.2f}ms (s)")

# Analyze spike response
print(f"\nSpike response analysis:")
spike_idx = np.argmin(np.abs(sim_t - 50))
print(f"  At t={sim_t[spike_idx]:.2f}ms: x={sim_x[spike_idx]:.6f}, s={sim_s[spike_idx]:.6f}")
print(f"  Peak s: {np.max(sim_s):.6f} at t={sim_t[np.argmax(sim_s)]:.2f}ms")
print(f"  Peak x: {np.max(sim_x):.6f} at t={sim_t[np.argmax(sim_x)]:.2f}ms")

# Check if autocorrelation looks reasonable
print(f"\nAutocorrelation quality check:")
print(f"  x autocorr[0] = {x_autocorr[0]:.6f} (should be 1.0)")
print(f"  s autocorr[0] = {s_autocorr[0]:.6f} (should be 1.0)")
print(f"  x autocorr at τ=20ms: {x_autocorr[np.argmin(np.abs(x_lags - 20))]:.6f}")
print(f"  s autocorr at τ=20ms: {s_autocorr[np.argmin(np.abs(s_lags - 20))]:.6f}")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. x has short memory (τ≈2ms) → fast response to spikes")
print("2. s has long memory (τ≈100ms) → enables temporal integration")
print("3. Correct data access (StateMonitor) is critical for analysis")
print("4. Autocorrelation quantifies the 'memory' timescales")
print("5. NMDA's slow s dynamics support working memory in Wang 2002 model")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)