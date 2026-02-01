import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.5  # per ms
tau_rise = 2.0  # ms
tau_decay = 100.0  # ms


# Convert to consistent units (ms)
# α = 0.5/ms, τ_decay = 100 ms, so α·τ_decay = 50

def simulate_single_spike():
    """Simulate s(t) after a spike at t=0"""
    dt = 0.01  # ms
    t_max = 500  # ms
    n_steps = int(t_max / dt)

    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    s = np.zeros(n_steps)

    # Initial conditions at t=0⁺ (just after spike)
    t[0] = 0
    x[0] = 1.0  # Spike adds 1
    s[0] = 0.0  # s starts at 0

    # Euler integration
    for i in range(1, n_steps):
        t[i] = t[i - 1] + dt

        # x decays exponentially
        dx = -x[i - 1] * dt / tau_rise
        x[i] = x[i - 1] + dx

        # s follows ODE
        ds = (-s[i - 1] / tau_decay + alpha * x[i - 1] * (1 - s[i - 1])) * dt
        s[i] = s[i - 1] + ds

    return t, x, s


t, x, s = simulate_single_spike()

# Print values at specific times
print("Time (ms) | x       | s")
print("-" * 30)
for time_ms in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]:
    idx = np.argmin(np.abs(t - time_ms))
    print(f"{time_ms:8.1f} | {x[idx]:7.4f} | {s[idx]:7.4f}")


# Steady-state function
def s_steady(x_val):
    return (alpha * tau_decay * x_val) / (1 + alpha * tau_decay * x_val)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# s(t) over time
ax1 = axes[0]
ax1.plot(t, s, 'b-', linewidth=2)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('s(t)')
ax1.set_title('s(t) after spike at t=0')
ax1.grid(True, alpha=0.3)

# Phase plot
ax2 = axes[1]
ax2.plot(x, s, 'b-', linewidth=2, label='Trajectory')

# Steady-state curve
x_range = np.linspace(0, 1, 100)
ax2.plot(x_range, s_steady(x_range), 'k--', linewidth=1.5, alpha=0.7, label='Steady-state')

ax2.set_xlabel('x')
ax2.set_ylabel('s')
ax2.set_title(f'Phase plot: α={alpha}/ms, τ_rise={tau_rise}ms, τ_decay={tau_decay}ms')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze initial behavior
print("\n" + "=" * 60)
print("INITIAL BEHAVIOR ANALYSIS")
print("=" * 60)
print(f"At t=0⁺: x=1.0, s=0.0")
print(f"Initial ds/dt = α·x·(1-s) - s/τ_decay")
print(f"              = {alpha}×1.0×(1-0) - 0/{tau_decay}")
print(f"              = {alpha} /ms")
print(f"              = {alpha * 1000} /s")

# When does s peak?
peak_idx = np.argmax(s)
peak_t = t[peak_idx]
peak_s = s[peak_idx]
peak_x = x[peak_idx]

print(f"\ns peaks at t={peak_t:.1f} ms:")
print(f"  s_peak = {peak_s:.4f}")
print(f"  x_at_peak = {peak_x:.4f}")
print(f"  Steady-state for this x: s_steady = {s_steady(peak_x):.4f}")
print(f"  Difference: {peak_s - s_steady(peak_x):.4f}")

# Compare with table values
print("\n" + "=" * 60)
print("COMPARISON WITH YOUR TABLE")
print("=" * 60)
print("Your table (shifted by 50ms for spike at t=50ms):")
print("Time after spike | Your s | My simulation")
print("-" * 50)

table_data = {
    0: 0.60,  # At spike time
    50: 0.30,  # 50ms after spike
    100: 0.15,
    150: 0.08,
    200: 0.05,
    250: 0.03,
    300: 0.02,
    350: 0.01,
    400: 0.00
}

for time_after in sorted(table_data.keys()):
    idx = np.argmin(np.abs(t - time_after))
    print(f"{time_after:15.0f} | {table_data[time_after]:6.2f} | {s[idx]:6.2f}")

print("\nDISCREPANCY: Your table shows s=0.60 IMMEDIATELY at spike time,")
print("but s should start at 0 and rise to a peak around 0.98!")
print("\nPossible explanations:")
print("1. Your table has different parameters (maybe α much smaller?)")
print("2. Your table shows values at discrete 50ms intervals, not continuous")
print("3. There's a time-averaging or different measurement in your table")