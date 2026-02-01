import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
import argparse


def exact_solution(t, t0=0, alpha=1.0, tau_rise=2.0, tau_decay=10.0):
    """
    Exact solution for s(t) with a single spike at t0
    Returns both s(t) and x(t)
    """
    dt = t - t0
    x = np.exp(-dt / tau_rise) if dt >= 0 else 0.0

    if dt < 0:
        return x, 0.0

    a = alpha * tau_rise
    r = tau_rise / tau_decay
    b = 1 - r

    # Lower integration limit
    z_low = a * np.exp(-dt / tau_rise)
    z_high = a

    # Compute γ(b, z_high) - γ(b, z_low)
    gamma_b = gamma(b)
    term = gamma_b * (gammainc(b, z_high) - gammainc(b, z_low))

    # Prefactor
    prefactor = a ** r

    # Exponential term
    exp_term = np.exp(a * np.exp(-dt / tau_rise) - dt / tau_decay)

    s = prefactor * exp_term * term
    return x, s


def simulate_euler(t_span, dt, t0, alpha, tau_rise, tau_decay):
    """Forward Euler (Newton) method"""
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros_like(t_values)
    s = np.zeros_like(t_values)

    # Find index closest to t0
    idx_spike = np.argmin(np.abs(t_values - t0))

    for i in range(1, len(t_values)):
        # x dynamics: dx/dt = -x/tau_rise + delta at t0
        if i == idx_spike:
            x[i] = x[i - 1] + 1  # Dirac delta impulse
        else:
            x[i] = x[i - 1] - dt * x[i - 1] / tau_rise

        # s dynamics: ds/dt = -s/tau_decay + alpha * x * (1 - s)
        s[i] = s[i - 1] + dt * (-s[i - 1] / tau_decay + alpha * x[i - 1] * (1 - s[i - 1]))

    return t_values, x, s


def simulate_rk4(t_span, dt, t0, alpha, tau_rise, tau_decay):
    """RK4 method with impulse handling"""
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros_like(t_values)
    s = np.zeros_like(t_values)

    # Find spike index
    idx_spike = np.argmin(np.abs(t_values - t0))

    # Create x(t) analytically for RK4 (since x has delta)
    x_analytic = np.exp(-(t_values - t0) / tau_rise) * (t_values >= t0)

    for i in range(1, len(t_values)):
        # Use analytic x(t) for s dynamics
        x_current = x_analytic[i - 1]
        s_current = s[i - 1]

        # RK4 steps for s
        k1 = -s_current / tau_decay + alpha * x_current * (1 - s_current)

        x_mid = x_analytic[i - 1]  # x at t + dt/2 (use analytic)
        s_mid = s_current + 0.5 * dt * k1
        k2 = -s_mid / tau_decay + alpha * x_mid * (1 - s_mid)

        s_mid = s_current + 0.5 * dt * k2
        k3 = -s_mid / tau_decay + alpha * x_mid * (1 - s_mid)

        x_next = x_analytic[i]  # x at t + dt
        s_next = s_current + dt * k3
        k4 = -s_next / tau_decay + alpha * x_next * (1 - s_next)

        s[i] = s_current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, x_analytic, s


def plot_results(config, t_span, dt):
    """Main plotting function"""
    # Generate exact solution
    t_exact = np.linspace(t_span[0], t_span[1], 1000)
    x_exact = np.zeros_like(t_exact)
    s_exact = np.zeros_like(t_exact)

    for i, t in enumerate(t_exact):
        x_exact[i], s_exact[i] = exact_solution(t, config.t0, config.alpha,
                                                config.tau_rise, config.tau_decay)

    # Numerical simulations
    t_euler, x_euler, s_euler = simulate_euler(t_span, dt, config.t0, config.alpha,
                                               config.tau_rise, config.tau_decay)
    t_rk4, x_rk4, s_rk4 = simulate_rk4(t_span, dt, config.t0, config.alpha,
                                       config.tau_rise, config.tau_decay)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: x(t) comparison
    ax = axes[0, 0]
    ax.plot(t_exact, x_exact, 'k-', linewidth=2, label='Exact')
    ax.plot(t_euler, x_euler, 'b--', linewidth=1.5, alpha=0.7, label=f'Euler (dt={dt})')
    ax.plot(t_rk4, x_rk4, 'r:', linewidth=1.5, alpha=0.7, label=f'RK4 (dt={dt})')
    ax.axvline(x=config.t0, color='g', linestyle=':', alpha=0.5, label=f'Spike at t={config.t0}')
    ax.set_xlabel('Time')
    ax.set_ylabel('$x(t)$')
    ax.set_title('$x(t)$: Comparison of methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: s(t) comparison
    ax = axes[0, 1]
    ax.plot(t_exact, s_exact, 'k-', linewidth=3, label='Exact')
    ax.plot(t_euler, s_euler, 'b--', linewidth=1.5, alpha=0.7, label='Euler')
    ax.plot(t_rk4, s_rk4, 'r:', linewidth=1.5, alpha=0.7, label='RK4')
    ax.axvline(x=config.t0, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('$s(t)$')
    ax.set_title('$s(t)$: Comparison of methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Errors for s(t)
    ax = axes[1, 0]
    # Interpolate exact solution to numerical time points
    s_exact_interp_euler = np.array([exact_solution(t, config.t0, config.alpha,
                                                    config.tau_rise, config.tau_decay)[1]
                                     for t in t_euler])
    s_exact_interp_rk4 = np.array([exact_solution(t, config.t0, config.alpha,
                                                  config.tau_rise, config.tau_decay)[1]
                                   for t in t_rk4])

    ax.plot(t_euler, np.abs(s_euler - s_exact_interp_euler), 'b-', label='Euler error')
    ax.plot(t_rk4, np.abs(s_rk4 - s_exact_interp_rk4), 'r-', label='RK4 error')
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute error')
    ax.set_title('Absolute error in $s(t)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Phase portrait
    ax = axes[1, 1]
    ax.plot(x_exact, s_exact, 'k-', label='Exact')
    ax.plot(x_euler, s_euler, 'b--', alpha=0.5, label='Euler')
    ax.plot(x_rk4, s_rk4, 'r:', alpha=0.5, label='RK4')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$s$')
    ax.set_title('Phase portrait: $s$ vs $x$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: ds/dt verification
    ax = axes[2, 0]
    # Compute ds/dt from exact solution (numerical derivative)
    ds_exact = np.gradient(s_exact, t_exact[1] - t_exact[0])
    # Compute RHS of ODE
    ds_ode = -s_exact / config.tau_decay + config.alpha * x_exact * (1 - s_exact)

    ax.plot(t_exact, ds_exact, 'k-', label='$ds/dt$ (numerical)')
    ax.plot(t_exact, ds_ode, 'm--', alpha=0.7, label='ODE RHS')
    ax.set_xlabel('Time')
    ax.set_ylabel('$ds/dt$')
    ax.set_title('Verification: $ds/dt$ matches ODE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Parameter summary
    ax = axes[2, 1]
    ax.axis('off')
    param_text = (
        f'Parameters:\n'
        f'$\\alpha = {config.alpha}$\n'
        f'$\\tau_{{rise}} = {config.tau_rise}$\n'
        f'$\\tau_{{decay}} = {config.tau_decay}$\n'
        f'$t_0 = {config.t0}$\n'
        f'$\\Delta t = {dt}$\n'
        f'Time span: [{t_span[0]}, {t_span[1]}]\n'
        f'Ratio $\\tau_{{rise}}/\\tau_{{decay}} = {config.tau_rise / config.tau_decay:.3f}$'
    )
    ax.text(0.1, 0.5, param_text, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Dynamics: $\\tau_{{rise}}={config.tau_rise}$, '
                 f'$\\tau_{{decay}}={config.tau_decay}$, $\\alpha={config.alpha}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print error summary
    print("\n" + "=" * 60)
    print("ERROR SUMMARY:")
    print("=" * 60)
    print(f"Maximum absolute error in s(t):")
    print(f"  Euler:  {np.max(np.abs(s_euler - s_exact_interp_euler)):.2e}")
    print(f"  RK4:    {np.max(np.abs(s_rk4 - s_exact_interp_rk4)):.2e}")
    print(f"\nRMS error in s(t):")
    print(f"  Euler:  {np.sqrt(np.mean((s_euler - s_exact_interp_euler) ** 2)):.2e}")
    print(f"  RK4:    {np.sqrt(np.mean((s_rk4 - s_exact_interp_rk4) ** 2)):.2e}")


def main():
    parser = argparse.ArgumentParser(description='Simulate synaptic dynamics with exact and numerical solutions')
    parser.add_argument('--alpha', type=float, default=5, help='Coupling strength')
    parser.add_argument('--tau_rise', type=float, default=2, help='Rise time constant')
    parser.add_argument('--tau_decay', type=float, default=100.0, help='Decay time constant')
    parser.add_argument('--t0', type=float, default=0.0, help='Spike time')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step for numerical methods')
    parser.add_argument('--t_start', type=float, default=-5.0, help='Start time')
    parser.add_argument('--t_end', type=float, default=500.0, help='End time')

    config = parser.parse_args()

    t_span = (config.t_start, config.t_end)

    print("=" * 60)
    print("SYNAPTIC DYNAMICS SIMULATION")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  α = {config.alpha}")
    print(f"  τ_rise = {config.tau_rise}")
    print(f"  τ_decay = {config.tau_decay}")
    print(f"  t0 = {config.t0}")
    print(f"  dt = {config.dt}")
    print(f"  Time span: [{t_span[0]}, {t_span[1]}]")
    print("=" * 60)

    plot_results(config, t_span, config.dt)


if __name__ == "__main__":
    main()