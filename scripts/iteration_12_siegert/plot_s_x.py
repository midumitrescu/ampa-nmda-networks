import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, gammainc


def s_t(t, t0=0, alpha=1.0, tau_rise=2.0, tau_decay=100.0):
    """
    Exact solution for s(t) with a single spike at t0
    """
    dt = t - t0
    if dt < 0:
        return 0.0

    a = alpha * tau_rise
    r = tau_rise / tau_decay

    # Use incomplete Gamma function: γ(b,z) = Γ(b) * P(b,z)
    # where P(b,z) is the regularized lower incomplete Gamma
    b = 1 - r

    # Lower integration limit
    z_low = a * np.exp(-dt / tau_rise)
    z_high = a

    # Compute γ(b, z_high) - γ(b, z_low) using scipy's gammainc
    # γ(b,z) = Γ(b) * gammainc(b, z)
    gamma_b = gamma(b)
    term = gamma_b * (gammainc(b, z_high) - gammainc(b, z_low))

    # Prefactor
    prefactor = a ** r

    # Exponential term
    exp_term = np.exp(a * np.exp(-dt / tau_rise) - dt / tau_decay)

    return prefactor * exp_term * term


# Time array
t = np.linspace(-5, 50, 5000)
t0 = 0
alpha = 0.5
tau_rise = 2.0
tau_decay = 100.0

# Compute s(t)
s_values = np.array([s_t(ti, t0, alpha, tau_rise, tau_decay) for ti in t])

# Also compute x(t) for comparison
x_values = np.exp(-(t - t0) / tau_rise) * (t >= t0)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(t, x_values, 'b--', linewidth=2, alpha=0.7, label=r'$x(t) = e^{-(t-t_0)/\tau_{\mathrm{rise}}}$')
plt.plot(t, s_values, 'r-', linewidth=3, label=r'$s(t)$ (exact solution)')

plt.axvline(x=t0, color='k', linestyle=':', alpha=0.5, label=f'Spike at $t_0={t0}$')
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('$x(t)$, $s(t)$', fontsize=14)
plt.title(
    f'Exact solution: $\\tau_\\mathrm{{rise}}={tau_rise}$, $\\tau_\\mathrm{{decay}}={tau_decay}$, $\\alpha={alpha}$',
    fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim([-5, 50])
plt.tight_layout()
plt.show()

# Optional: Also plot ds/dt to verify ODE
dt = t[1] - t[0]
# Numerical derivative
ds_dt = np.gradient(s_values, dt)
# Theoretical ds/dt from ODE
ds_dt_theory = -s_values / tau_decay + alpha * x_values * (1 - s_values)

plt.figure(figsize=(10, 6))
plt.plot(t, ds_dt, 'b-', linewidth=2, label='$ds/dt$ (numerical derivative)')
plt.plot(t, ds_dt_theory, 'r--', linewidth=2, label='$ds/dt$ (from ODE)')
plt.xlabel('Time $t$', fontsize=14)
plt.ylabel('$ds/dt$', fontsize=14)
plt.title('Verification: derivative matches ODE', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim([-5, 50])
plt.tight_layout()
plt.show()