import unittest
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import cm, uF, ohm, um, Quantity, is_dimensionless, get_dimensions, volt, have_same_dimensions, mV
from brian2.units import second, ms
from brian2.units.allunits import ampere, mampere
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.sparse import diags


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

    indices = [0, 3, 20, 80, 99]
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

def analyse_eigenvalues_generalized_locally_toeplitz_matrix(A):
    lam = np.linalg.eigvals(A.toarray())
    print("Eigenvalues")
    # print(lam)
    lam_pos = np.abs(lam)
    stiffness = np.max(lam_pos) / np.min(lam_pos)
    idx = np.argsort(np.abs(lam))
    # yplt.plot(np.arange(len(lam.real)), np.abs(lam.real))
    plt.semilogy(np.abs(lam[idx]))
    plt.xlabel("index")
    plt.ylabel("$\lambda$")
    plt.yscale("log")
    plt.title(f"Eigenvalues of Jacobian matrix. Stiffness = {stiffness: .5f}")
    print(f"Stiffness = {stiffness}. Max eig={np.max(lam_pos)}, min eig={np.min(lam_pos)}")
    idx = np.argsort(np.abs(lam))
    np.set_printoptions(suppress=True, precision=10)
    print("Smallest |lambda|:")
    print(lam[idx[:10]])
    print("\nLargest |lambda|:")
    print(lam[idx[-10:]])
    plt.show()

class TaperedDendritesPDECase(unittest.TestCase):
    def test_heat_difussion(self):

        # 1. Parameters
        alpha = 0.01  # Thermal diffusivity
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        u0 = np.exp(-100 * (x - 3.5) ** 2)  # Gaussian peak in the center

        # 2. Define the PDE as a system of ODEs
        def heat_equation(t, u):
            # Initialize derivative array
            du_dt = np.zeros_like(u)

            # Interior points using central difference
            du_dt[1:-1] = alpha * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2

            # Boundary conditions (Dirichlet: u = 0 at x=0 and x=L)
            du_dt[0] = 0
            du_dt[-1] = 0

            return du_dt

        # 3. Solve the ODE system using SciPy's solve_ivp
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
        sol = solve_ivp(heat_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

        # 4. Visualize the results
        plt.figure(figsize=(8, 5))
        for i in range(len(sol.t)):
            plt.plot(sol.y[:, i], label=f't = {sol.t[i]:.1f}')

        plt.title("1D Heat Equation via Method of Lines")
        plt.xlabel("Spatial grid (x)")
        plt.ylabel("Temperature (u)")
        plt.legend()
        plt.show()

    def test_heat_difussion_on_cone(self):

        # 1. Parameters
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.05
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)

        for x_2 in [0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 6.5, 8.5, 9.5, 9.9]:

            #u0_1 = np.exp(-100 * (x - 3.5) ** 2)
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2)

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            cm = 10
            gL = 0.1
            ra = 10
            I_syn_func = None
            #def linear_taper_cable_equation(t, V, x, dx, cm, gL, ra, rho_func, r_func, m, I_syn_func):
            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=t)

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]
                rho_mid = rho[1:-1]

                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term
                # r^2 * d2V/dx2 + 2 * m * r * dV/dx
                diffusion_term = (r_mid ** 2 * dV2_dx2) + (2 * k * r_mid * dV_dx)

                # 5. Assemble full right-hand side equation
                prefactor = 1.0 / (2.0 * ra * rho_mid)
                dV_dt[1:-1] = (-gL * V_mid + prefactor * diffusion_term + I_syn[1:-1]) / cm

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                print()
                return dV_dt

            # 3. Solve the ODE system using SciPy's solve_ivp
            t_span = (0, 10.0)
            t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
            sol = solve_ivp(linear_taper_cable_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

            # 4. Visualize the results
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(8, 6),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=False
            )

            # Top plot: solution over time
            for i in range(len(sol.t)):
                ax1.plot(sol.y[:, i], label=f't = {sol.t[i]:.1f}')

            ax1.set_title("Cable eq")
            ax1.set_xlabel("Spatial grid (x)")
            ax1.set_ylabel("Voltage (mV)")
            ax1.set_ylim((0, 1.1))
            ax1.legend()

            # Bottom plot: x vs r_of_x
            ax2.plot(x, r_of_x, 'k-')
            ax2.set_xlabel("x")
            ax2.set_ylabel("r(x)")
            ax2.set_title("Radius profile")

            plt.tight_layout()
            plt.show()

    # there is a formula for stability of the simulation delta t <= 0.5 (delta x)^2 / alpha
    # alpha is the prefactor of the second order deriv
    def test_solve_heat_eq_from_youtube_tutorial(self):
        a = 110 # diffusivity coefficient
        length = 50 # mm
        time = 4 # seconds
        nodes = 10
        dx = length / nodes
        dt = 0.5 * dx**2 / a

        u = np.zeros(nodes) + 20 # 20 is initial condition
        u[0] = 100 # Degrees. Initial condition.
        u[-1] = 100

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))
        #plt.show(block=False)

        counter = 0
        # simulation
        while counter < time:
            w = u.copy()

            for i in range(1, nodes - 1):
                u[i] = dt * a * (w[i-1] - 2 * w[i] + w[i+1]) / dx**2 + w[i]

            counter += dt
            print(f"t: {counter: .3f} [s], Average temperature: {np.mean(u): .2f} C")
            pcm.set_array([u])
            axis.set_title(f"Distribution at t: {counter: .3f}")
            #plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        plt.pause(1)
        plt.ioff()

    def test_solve_heat_eq_from_youtube_tutorial_improved(self):
        a = 110 # diffusivity coefficient
        length = 50 # mm
        time = 4 # seconds
        nodes = 10
        dx = length / nodes
        dt = 0.5 * dx**2 / a

        u = np.zeros(nodes) + 20 # 20 is initial condition
        u[0] = 100 # Degrees. Initial condition.
        u[-1] = 100

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))
        #plt.show(block=False)

        counter = 0
        # simulation
        while counter < time:
            w = u.copy()

            u[1:-1] = dt * a * (w[0:-2] - 2 * w[1:-1] + w[2:]) / dx**2 + w[1:-1]

            counter += dt
            print(f"t: {counter: .3f} [s], Average temperature: {np.mean(u): .2f} C")
            pcm.set_array([u])
            axis.set_title(f"Distribution at t: {counter: .3f}")
            #plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        plt.pause(1)
        plt.ioff()

    def test_difussion_on_cone_colormap(self):
        # 1. Parameters
        L = 10.0  # Length of the rod
        N = 50  # Number of spatial grid points
        dx = L / N  # Spatial step size

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.05
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k ** 2 * r_0 ** 2)

        x_2 = 6.5

        u0_1 = np.exp(-100 * (x - 3.5) ** 2)
        u0_2 = np.exp(-100 * (x - x_2) ** 2)

        u0 = u0_1 + u0_2

        import matplotlib
        matplotlib.use("TkAgg")
        plt.ion()
        fig, axis = plt.subplots()
        pcm = axis.pcolormesh([np.zeros_like(x)], cmap=plt.cm.jet, vmin=0, vmax=0.001)
        plt.colorbar(pcm, ax=axis)
        axis.set_ylim((-2, 3))

        def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
            """
            Models a spatio-temporal Dirac delta impulse input.
            """
            # 1. Temporal component: Smooth Gaussian regularized delta
            temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

            # 2. Spatial component: Step indicator normalized by mesh size
            spatial_delta = np.zeros_like(x)
            closest_node_index = np.argmin(np.abs(x - x0))
            spatial_delta[closest_node_index] = 1.0 / dx

            # Combined current injection vector
            return I0 * temporal_delta * spatial_delta

        cm = 10
        gL = 0.1
        ra = 10
        I_syn_func = None

        # def linear_taper_cable_equation(t, V, x, dx, cm, gL, ra, rho_func, r_func, m, I_syn_func):
        def linear_taper_cable_equation(t, V):
            """
            Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
            Uses standard analytical expansion and central differences.
            """
            dV_dt = np.zeros_like(V)

            # 1. Compute profiles at standard grid points
            r = r_of_x
            rho = rho_of_x
            I_syn = synaptic_input_profile(t=t)

            # 2. Slice variables for interior nodes (index 1 to N-2)
            V_mid = V[1:-1]
            V_left = V[:-2]
            V_right = V[2:]

            r_mid = r[1:-1]
            rho_mid = rho[1:-1]

            # 3. Compute central differences for the derivatives
            dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
            dV_dx = (V_right - V_left) / (2 * dx)

            # 4. Reconstruct the expanded diffusion term
            # r^2 * d2V/dx2 + 2 * k * r * dV/dx
            diffusion_term = (r_mid ** 2 * dV2_dx2) - (2 * k * r_mid * dV_dx)

            # 5. Assemble full right-hand side equation
            prefactor = 1.0 / (2.0 * ra * rho_mid)
            dV_dt[1:-1] = (-gL * V_mid + prefactor * diffusion_term + I_syn[1:-1]) / cm

            # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
            # Simple zero-flux boundary condition approximation
            dV_dt[-1] = 0

            print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
            print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
            print()

            pcm.set_array([V])
            # plt.pause(0.005)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

            return dV_dt

        # 3. Solve the ODE system using SciPy's solve_ivp
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10.0, 6)  # Save output at 6 specific times
        sol = solve_ivp(linear_taper_cable_equation, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)

        plt.pause(1)
        plt.ioff()

    # this shows that this is a current based model. We definitely need conductance based model for this!
    def test_heat_difussion_on_cone_runge_kutta(self):

        cm = 2
        gL = 0.1
        ra = 1

        # 1. Parameters
        L = 100.0  # Length of the rod
        N = 500  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        tau = cm / gL

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.0098
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)
        prefactor = 1 / (cm * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        min_prefactor_difussion = rho_of_x[-1] / (2 * prefactor)
        max_prefactor_difussion = rho_of_x[0] / (2 * prefactor)
        # stability requirements
        # second order PDE induces a fourier grid number. Defined as 0.5 delta_x^2 / (max diffussion coeff)
        grid_fourier_number = 0.5 * dx**2 / max_prefactor_difussion
        # first order PDE constraint. Courant-Friedrics-Lewy number. delta x / (1st order diffusion coef)
        cfl_constraint = dx / (k * r_0 / prefactor)

        #dt = min(grid_fourier_number, cfl_constraint)

        dt = 0.25 * dx ** 2 * r_0 * min(min_prefactor_difussion, 1)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=10):
            #u0_1 = np.exp(-100 * (x - 3.5) ** 2) / cm
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2) / cm

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=-1) / cm

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]


                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term

                # 5. Assemble full right-hand side equation
                dV_dt[1:-1] = ( -V_mid / tau  - k * r_0 / prefactor * dV_dx + r_mid / (2 * prefactor) * dV2_dx2 + I_syn[1:-1])

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                #print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                #print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                #print()
                return dV_dt

            def rk4(f, t_span, y0, dt, save_times=None):
                t0, tf = t_span

                t = t0
                y = y0.copy()

                if save_times is None:
                    save_times = np.arange(t0, tf + dt, dt)

                save_times = np.asarray(save_times)

                sol = np.empty((len(save_times), len(y0)))
                sol[0] = y

                save_idx = 1

                while t < tf:
                    delta_t = min(dt, tf - t)

                    k1 = f(t, y)
                    k2 = f(t + delta_t / 2, y + delta_t * k1 / 2)
                    k3 = f(t + delta_t / 2, y + delta_t * k2 / 2)
                    k4 = f(t + delta_t, y + delta_t * k3)

                    y += delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    t += delta_t

                    while save_idx < len(save_times) and t >= save_times[save_idx]:
                        sol[save_idx] = y
                        save_idx += 1

                return save_times, sol

            # 3. Solve the ODE system using manual runge kutta
            t_eval = np.linspace(0, 0.1, 5)  # Save output at 6 specific times
            t_eval = np.arange(0, 5) * 5 * dt
            saved_sols, sol = rk4(linear_taper_cable_equation, t_span=(0, t_max), y0=u0, dt=dt, save_times=t_eval)

            if plot:
                # 4. Visualize the results
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(8, 6),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=False
                )

                # Top plot: solution over time
                for i in range(len(saved_sols)):
                    ax1.plot(x, sol[i], label=f't = {saved_sols[i]:.1f}')

                ax1.set_title(f"Cable eq. Second input at {x_2: .1f}. Max V = {np.max(sol[0]): .4f} mV locally at index {np.argmax(sol[0]) * dx : .2f}")
                ax1.set_xlabel("Spatial grid (x)")
                ax1.set_ylabel("Voltage (mV)")
                #ax1.set_ylim((0, 0.5))
                ax1.legend()

                # Bottom plot: x vs r_of_x
                ax2.plot(x, r_of_x, 'k-')
                ax2.set_xlabel("x")
                ax2.set_ylabel("r(x)")
                ax2.set_title("Radius profile")

                plt.tight_layout()
                fig.show()

            return np.max(sol[0]), np.argmax(sol[0])

        #for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1, L - 0.1, splits)
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=0.3, plot=True) for x2 in x2_values
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


    def test_heat_difussion_on_cone_forward_euler(self):

        cm = 2
        gL = 0.1
        ra = 1

        # 1. Parameters
        L = 100.0  # Length of the rod
        N = 500  # Number of spatial grid points
        dx = L / (N - 1)  # Spatial step size

        tau = cm / gL

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)
        k = 0.0098
        r_0 = 1
        r_of_x = r_0 * (1 - k * x)
        rho_of_x = r_of_x * np.sqrt(1 + k**2 * r_0**2)
        prefactor = 1 / (cm * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        min_prefactor_difussion = rho_of_x[-1] / (2 * prefactor)
        dt = 0.25 * dx ** 2 * r_0 * min(min_prefactor_difussion, 1)

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        def solve(x_2, plot=True, t_max=10):
            #u0_1 = np.exp(-100 * (x - 3.5) ** 2) / cm
            u0_1 = np.zeros_like(x)
            u0_2 = np.exp(-100 * (x - x_2) ** 2) / cm

            u0 = u0_1 + u0_2

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                return I0 * temporal_delta * spatial_delta

            def linear_taper_cable_equation(t, V):
                """
                Evaluates dV/dt for a cable with a perfectly linear radius profile r(x) = m*x + b.
                Uses standard analytical expansion and central differences.
                """
                dV_dt = np.zeros_like(V)

                # 1. Compute profiles at standard grid points
                r = r_of_x
                rho = rho_of_x
                I_syn = synaptic_input_profile(t=-1) / cm

                # 2. Slice variables for interior nodes (index 1 to N-2)
                V_mid = V[1:-1]
                V_left = V[:-2]
                V_right = V[2:]

                r_mid = r[1:-1]

                # 3. Compute central differences for the derivatives
                dV2_dx2 = (V_right - 2 * V_mid + V_left) / (dx ** 2)
                dV_dx = (V_right - V_left) / (2 * dx)

                # 4. Reconstruct the expanded diffusion term

                # 5. Assemble full right-hand side equation
                dV_dt[1:-1] = ( -V_mid / tau  - k * r_0 / prefactor * dV_dx + r_mid / (2 * prefactor) * dV2_dx2 + I_syn[1:-1])

                # 6. Apply Boundary Conditions (Example: Sealed / insulated ends)
                # Simple zero-flux boundary condition approximation
                dV_dt[-1] = 0

                #print(f"t = {t: .3f}, V[14, 18]={V[14: 18]}")
                #print(f"t = {t: .3f}, dV/dt[14, 18]={dV_dt[15: 19]}")
                #print()
                return dV_dt

            def rk4(f, t_span, y0, dt, save_times=None):
                t0, tf = t_span

                t = t0
                y = y0.copy()

                if save_times is None:
                    save_times = np.arange(t0, tf + dt, dt)

                save_times = np.asarray(save_times)

                sol = np.empty((len(save_times), len(y0)))
                sol[0] = y

                save_idx = 1

                while t < tf:
                    delta_t = min(dt, tf - t)

                    k1 = f(t, y)
                    k2 = f(t + delta_t / 2, y + delta_t * k1 / 2)
                    k3 = f(t + delta_t / 2, y + delta_t * k2 / 2)
                    k4 = f(t + delta_t, y + delta_t * k3)

                    y += delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    t += delta_t

                    while save_idx < len(save_times) and t >= save_times[save_idx]:
                        sol[save_idx] = y
                        save_idx += 1

                return save_times, sol

            # 3. Solve the ODE system using manual runge kutta
            t_eval = np.linspace(0, 0.1, 5)  # Save output at 6 specific times
            t_eval = np.arange(0, 5) * 5 * dt
            saved_sols, sol = rk4(linear_taper_cable_equation, t_span=(0, t_max), y0=u0, dt=dt, save_times=t_eval)

            if plot:
                # 4. Visualize the results
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(8, 6),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=False
                )

                # Top plot: solution over time
                for i in range(len(saved_sols)):
                    ax1.plot(x, sol[i], label=f't = {saved_sols[i]:.1f}')

                ax1.set_title(f"Cable eq. Second input at {x_2: .1f}. Max V = {np.max(sol[0]): .4f} mV locally at index {np.argmax(sol[0]) * dx : .2f}")
                ax1.set_xlabel("Spatial grid (x)")
                ax1.set_ylabel("Voltage (mV)")
                #ax1.set_ylim((0, 0.5))
                ax1.legend()

                # Bottom plot: x vs r_of_x
                ax2.plot(x, r_of_x, 'k-')
                ax2.set_xlabel("x")
                ax2.set_ylabel("r(x)")
                ax2.set_title("Radius profile")

                plt.tight_layout()
                fig.show()

            return np.max(sol[0]), np.argmax(sol[0])

        #for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:
        for splits in [10]:
            x2_values = np.linspace(0.1, L - 0.1, splits)
            results = Parallel(
                n_jobs=-3 if len(x2_values) > 2 else 1,  # use all CPU cores
                backend="loky",  # process-based (default)
                verbose=10
            )(
                delayed(solve)(x2, t_max=0.3, plot=True) for x2 in x2_values
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

    def test_pde_triagonal_matrix(self, verbose=False, x_N=100, t_max = 50 * ms):

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
        r_L = 0.5 * um

        k = (1 - r_L / r_0) / L
        r_of_x = r_0 * (1 - k * x)

        a = k * r_0 / (2 * c_m * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))
        b = r_0 * (1 - k * x) / (2 * c_m * ra * np.sqrt(1 + r_0 ** 2 * k ** 2))

        # TODO: find proper dt
        # CFL condition: delta t <= 1/2 (delta x) ^2 / alpha. Alpha is the prefactor of alpha dV^2 / d^2x
        dt = 0.5 * dx ** 2 / b[0]

        lower = b[1:] / dx ** 2 + a / (2 * dx)
        main = -1 / tau - 2 * b / dx ** 2
        upper = b[:-1] / dx ** 2 - a / (2 * dx)

        # Sparse tridiagonal matrix
        A = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="lil"
        )

        # Sparse tridiagonal matrix
        A_only_tau_decay = diags(
            diagonals=[np.ones(len(x)) * (-1/tau)],
            offsets=[0],
            format="lil"
        )

        A_no_neumann_conditions = diags(
            diagonals=[lower, main, upper],
            offsets=[-1, 0, 1],
            format="csr"
        )

        # ensure boundary conditions automatically in A matrix
        A[0, 0] = -1 / tau - 2 * b[0] / dx ** 2
        A[0, 1] = 2 * b[0] / dx ** 2
        A[-1, -2] = 2 * b[-1] / dx ** 2
        A[-1, -1] = -1 / tau - 2 * b[-1] / dx ** 2
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

            def synaptic_input_profile(t, t0=3.5, x0=6.5, I0=1.0, sigma=0.05):

                #TODO: attention. np.zeroes_like(x) has units of distance
                result = np.zeros(len(x)) * mampere / cm ** 2
                assert have_same_dimensions(result[0], 1 * mampere / cm**2)
                return result
                """
                Models a spatio-temporal Dirac delta impulse input.
                """
                # 1. Temporal component: Smooth Gaussian regularized delta
                temporal_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))

                # 2. Spatial component: Step indicator normalized by mesh size
                spatial_delta = np.zeros_like(x)
                closest_node_index = np.argmin(np.abs(x - x0))
                spatial_delta[closest_node_index] = 1.0 / dx

                # Combined current injection vector
                I_injected = I0 * temporal_delta * spatial_delta * mampere / cm**2
                return I_injected

            def linear_taper_cable_equation(t, V):
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
                """
                Forward Euler solver.

                Parameters
                ----------
                f : callable
                    RHS function f(t, V)
                t_span : tuple
                    (t0, tf)
                V0 : array
                    Initial condition
                dt : float
                    Time step
                save_every : int
                    Save every N iterations

                Returns
                -------
                times : array
                    Saved times
                sol : array
                    Saved solutions, shape = (time, space)
                """

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
            times, V_s = forward_euler(linear_taper_cable_equation, t_span=(0 * ms, t_max), V0=u0, dt=dt, saved_frames=400)

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
            #x2_values = [6.5 * um, 490*um]
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
