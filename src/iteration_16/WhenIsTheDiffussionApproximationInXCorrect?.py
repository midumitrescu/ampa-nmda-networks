import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau_r = 2e-3  # 2 ms in seconds

# Rate range (Hz)
r = np.logspace(0, 5, 1000)  # 1 Hz to 100 kHz

# epsilon(r)
epsilon = 1 / np.sqrt(2 * r * tau_r)

# Thresholds
eps_levels = [0.1, 0.2, 0.3]
colors = ['red', 'orange', 'green']

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r, epsilon, lw=2, label=r'$\epsilon(r)=1/\sqrt{2 \cdot \tau_r \cdot r_N}$')

# Add threshold lines and corresponding r values
for eps, col in zip(eps_levels, colors):
    r_thresh = 1 / (2 * tau_r * eps**2)

    # horizontal line
    plt.axhline(eps, color=col, ls='--', lw=1.5)
    plt.text(2e4, eps*1.05, rf'$\epsilon={eps}$', color=col)

    # vertical line
    plt.axvline(r_thresh, color=col, ls=':', lw=1.5)
    plt.text(r_thresh*1.1, 2, f"{r_thresh:.0f} Hz",
             rotation=90, color=col)

# Axes formatting
plt.xscale('log')
plt.yscale('log')

plt.xlabel('log Input rate $r$ (Hz)')
plt.ylabel(r'log $\epsilon = \sigma_x / \langle x \rangle$')
plt.title(r'For which rate is the diffusion approximation of x valid? ($\tau_r = 2$ ms)')

plt.legend()

plt.tight_layout()
plt.show()