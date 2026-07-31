import unittest
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import cm, uF, ohm, um, Quantity, is_dimensionless, get_dimensions, volt, have_same_dimensions, mV, uamp, \
    meter
from brian2.units import second, ms
from brian2.units.allunits import ampere, mampere, pampere
from joblib import Parallel, delayed
from numpy.ma.testutils import assert_array_equal
from scipy.sparse import diags

from numpy.testing import assert_allclose


# TODO: Normalize the way Dayan has it. ie = Ie τm δ(x)δ(t)/ 2πa. A is the radius! We have a slightly different formulation. But Still
def dirac_delta(x0:Quantity, t0:Quantity, tau_m: Quantity, I_e:Quantity, x: np.ndarray[Quantity], r_of_x: Quantity, t:Quantity, dx: Quantity, dt:Quantity) -> np.ndarray[Quantity]:

    if is_dimensionless(t0):
        t0 = t0 * ms
    if is_dimensionless(x0):
        x0 = x0 * um

    assert have_same_dimensions(t0, ms)
    assert have_same_dimensions(t, ms)
    assert have_same_dimensions(dt, ms)
    assert have_same_dimensions(tau_m, ms)
    assert have_same_dimensions(x0, um)
    assert have_same_dimensions(x[0], um)
    assert have_same_dimensions(dx, um)
    assert have_same_dimensions(r_of_x, um)

    if t0 - t < 0 or t0 - t >= dt:
        return np.zeros(len(x)) * mampere / cm ** 2

    print(f"[t={t} Finally inserting input at {t0}. t - t0 = {t-t0}")
    result = np.zeros(len(x)) * mampere / cm ** 2
    result_other_units = np.zeros(len(x)) * ampere / meter ** 2

    delta_xt = 1 / (dx * dt)

    i_e = I_e * tau_m / (2 * np.pi * r_of_x) * delta_xt

    if x0 <= x[0]:
        result[0] = i_e

    elif x0 >= x[-1]:
        result[-1] = i_e

    else:
        i_e_index = np.searchsorted(x, x0) - 1
        alpha = (x0 - x[i_e_index]) / dx

        result[i_e_index] = i_e * (1.0 - alpha)
        result[i_e_index + 1] = i_e * alpha

    assert have_same_dimensions(result[0], 1 * mampere / cm ** 2)
    return result

def plot_difussion_solution(times, x, r_of_x, V_s, verbose=True):

    if verbose:
        assert have_same_dimensions(x[0], 1*um)
        assert have_same_dimensions(r_of_x[0], 1*um)
        assert have_same_dimensions(V_s[0][0], 1*mV)
        assert have_same_dimensions(times[0], 1*ms)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(8, 9),
        gridspec_kw={'height_ratios': [3, 2, 1]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = x / um
    r_of_x = r_of_x / um
    times = times / ms
    V_s = V_s / mV

    print("x:", x[0], x[-1], len(x))
    print("times:", times[0], times[-1], len(times))
    print("V_s:", V_s.shape)

    im = ax1.imshow(
        V_s ,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        #vmax=0.05
    )

    fig.colorbar(im, ax=ax1, label="Voltage (mV)")

    ax1.set_title(
        f"Cable equation. Max V = {np.max(V_s):.4f} mV"
    )
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("t [ms]")

    indices = [0, 3, 20, 50, 80, 99]
    for i in indices:
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} μm"
        )

    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("V [mV]")
    ax2.set_title("Voltage at selected positions")
    ax2.legend()

    # ============================================
    # Bottom: cone geometry
    # ============================================

    r = r_of_x

    # Cone walls
    ax3.plot(x, r, 'k', linewidth=2)
    ax3.plot(x, -r, 'k', linewidth=2)

    # Fill cone
    ax3.fill_between(
        x,
        -r,
        r,
        color='gray',
        alpha=0.3
    )

    # Center axis y=0
    ax3.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )

    # Vertical line at x=0
    ax3.axvline(
        0,
        color='gray',
        linestyle='--',
        linewidth=1.5
    )

    ax3.set_xlabel("x")
    ax3.set_ylabel("radius")
    ax3.set_title("Cable geometry")

    #ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

class CylindricalDendriticTreePDECase(unittest.TestCase):

    def test_difussion_pde_triagonal_matrix(self, verbose=False, x_N=101, t_max = 1 * ms):

        c_m = 1 * uF / cm **2
        Rm = 2 * 1E4 * ohm * cm**2
        gL = 1/Rm
        ra = 100 * ohm * cm

        # 1. Parameters
        L = 500.0 * um
        N = x_N
        dx = L / (N - 1) # um

        tau = c_m / gL # ms

        assert have_same_dimensions(tau, 1*second)

        print("tau=", tau)

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)

        r_0 = 2 * um

        r_of_x = np.ones(len(x)) * r_0
        b = r_0 / (2 * c_m * ra)
        assert have_same_dimensions(b, meter ** 2 / second)

        # TODO: find proper dt
        # CFL condition: delta t <= 1/2 (delta x) ^2 / alpha. Alpha is the prefactor of alpha dV^2 / d^2x
        dt = 0.5 * dx ** 2 / b

        difussion = np.ones(len(x) - 1) * b / dx ** 2
        difussion_decay = np.ones(len(x)) * (-1 / tau - 2 * b / dx ** 2)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[difussion, difussion_decay, difussion],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b / dx ** 2
        A[0, 1] = 2 * b / dx ** 2
        A[-1, -2] = 2 * b / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b / dx ** 2
        # empirically, this does not work! Stiffness computed when boundary conditions are applied: 306 030  = 3*1E6 vs 7E4 when conditions are not applied

        #analyse_eigenvalues_generalized_locally_toeplitz_matrix(A)

        A = A.tocsr().toarray() * (1 / second)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=300 * ms):

            if is_dimensionless(t_max):
                t_max = t_max * ms

            # u0_1 = np.exp(-100 * (x - 3.5) ** 2) / c_m
            u0_1 = np.zeros(x.shape) * mV
            sigma_v = 10 * um
            u0_2 = 100 * np.exp(- 0.5 * (((x - x_2) / sigma_v) ** 2))  * mV

            u0 = u0_1 + u0_2

            print(f"{u0[0] / volt}. Dimension {get_dimensions(u0[0] / volt)}")

            if verbose:
                assert have_same_dimensions(1 * volt / second, 1 * ampere / uF)
                assert have_same_dimensions(u0, 1 * volt)
                assert have_same_dimensions(u0, 1 * volt)

            def synaptic_input_profile(t):

               return dirac_delta(x0=250*um, t0=0.5 * ms, I_e=1.5 * pampere, x=x, t=t, dx=dx, dt=dt, tau_m=tau, r_of_x=r_0)

            def cylindrical_cable_equation(t, V):
                """
                Computes dV/dt = A @ V + I_syn/c_m
                """
                I_syn = synaptic_input_profile(t)

                if verbose:
                    print(f"A: {get_dimensions(A)}")
                    print(f"V: {get_dimensions(V)}")
                    print(f"I Syn: {get_dimensions(I_syn)}")

                    print(f"A @ V: {get_dimensions(A @ V)}")

                    assert have_same_dimensions(A[0, 0], 1 / second)
                    assert have_same_dimensions(V[0], 1*volt)
                    assert have_same_dimensions(I_syn[0] / c_m, 1*volt/second)
                result = A @ V + I_syn / c_m

                if verbose:
                    assert have_same_dimensions(result[0], 1 * volt / second)

                return result

            def forward_euler(f: Callable[[float, np.ndarray], np.ndarray], t_span, V0: np.ndarray, dt: Quantity, saved_frames=1):

                t0, tf = t_span

                # Number of Euler steps
                num_steps = int(np.ceil((tf - t0) / dt))

                # Number of saved states
                save_every = int(np.ceil(num_steps / saved_frames))
                num_save = num_steps//save_every + 1

                # Preallocate
                times = np.zeros(num_save) * ms
                sol = np.zeros((num_save, len(V0))) * mV
                t = t0
                V = V0.copy()

                times[0] = t0
                sol[0] = V

                if verbose:
                    assert have_same_dimensions(V0[0], 1*mV)
                    assert have_same_dimensions(V, 1*mV)
                    assert have_same_dimensions(sol[0][0], 1*mV)

                for step in range(1, num_steps + 1):
                        dt_step = min(dt, tf - t)

                        # Forward Euler step
                        V = V + dt_step * f(t, V)

                        t += dt_step

                        # If another save_every bunch
                        if step % save_every == 0:
                            iteration = step // save_every
                            print(f"[f {iteration}/{num_save}]: Reached step {step} from {num_steps} ({100 * step / num_steps:.2f}%)")
                            times[iteration] = t
                            sol[iteration] = V

                if verbose:
                    assert have_same_dimensions(sol[0], 1 * mV)
                    assert have_same_dimensions(sol[0][0], 1 * mV)
                    assert have_same_dimensions(V0[0], 1 * mV)
                    assert have_same_dimensions(V, 1 * mV)

                return times, sol

            # 3. Solve the ODE system using manual runge kutta
            times, V_s = forward_euler(cylindrical_cable_equation, t_span=(0 * ms, t_max), V0=u0, dt=dt, saved_frames=400)

            if verbose:
                assert have_same_dimensions(times[0], 1 * ms)
                assert have_same_dimensions(V_s[0][0], 1 * mV)
                assert have_same_dimensions(V_s[0], 1 * mV)

            if plot:
                plot_difussion_solution(times=times, x=x, r_of_x=r_of_x, V_s=V_s)

            return np.max(V_s[1]), np.argmax(V_s[1])

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1 * um, L - 0.1 * um, splits)
            x2_values = [6.5 * um, 490*um]
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=t_max, plot=True) for x2 in x2_values
            )

            max_vals, argmax_vals = zip(*results)
            max_vals = np.array(max_vals)
            argmax_vals = np.array(argmax_vals)
            argmax_x = np.asarray(argmax_vals) * dx

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # x2 vs maximum value
            axes[0].plot(x2_values, max_vals, lw=2)
            axes[0].set_xlabel(r"$x_2$")
            axes[0].set_ylabel(r"$\max(V)$")
            axes[0].set_title("Peak voltage")

            # x2 vs argmax
            axes[1].plot(x2_values, argmax_x, lw=2)
            axes[1].set_xlabel(r"$x_2$")
            axes[1].set_ylabel(r"$\arg\max(V)$")
            axes[1].set_title("Peak location")

            # argmax vs max
            axes[2].scatter(argmax_x, max_vals, s=8, alpha=0.6)
            axes[2].set_xlabel(r"$\arg\max(V)$")
            axes[2].set_ylabel(r"$\max(V)$")
            axes[2].set_title("Peak location vs peak voltage")

            fig.suptitle(f"{splits} splits")

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    unittest.main()
