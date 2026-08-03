import itertools
import unittest
from http.cookiejar import lwp_cookie_str
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import cm, uF, ohm, um, Quantity, is_dimensionless, get_dimensions, volt, have_same_dimensions, mV, meter, \
    nsecond
from brian2.units import second, ms
from brian2.units.allunits import ampere, mampere, pampere, nsecond
from joblib import Parallel, delayed
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

from iteration_19_tapered_dendrites.data import CableParameters, to_SI, NumericalCableParameters

Rm = 2 * 1E4 * ohm * cm ** 2

default_params = CableParameters(c_m=1 * uF / cm ** 2,
                                 Rm=Rm,
                                 gL=1 / Rm,
                                 ra=100 * ohm * cm,
                                 L=500.0 * um,
                                 N=101,
                                 r0=2 * um,
                                 I_e=150 * pampere)

def is_debugging():
    import sys

    if sys.gettrace() is not None:
        print("Running under a debugger")
        debugging = True
    else:
        debugging = False
        print("Normal execution")

    return debugging

def run_simulation_unitless(x_N, t_max,verbose=False):
    simulation_params = default_params.with_property(t=t_max, N=x_N)
    N = simulation_params.N
    si_units = simulation_params.to_numerical()

    # 1. Parameters
    c_m = si_units.c_m
    dx = si_units.dx
    tau = si_units.tau

    # Spatial domain and initial condition
    x = si_units.x
    r_0 = si_units.r0

    r_of_x = np.ones(N) * r_0
    b = si_units.b

    difussion = np.ones(N - 1) * b / dx ** 2
    difussion_decay = np.ones(N) * (-1 / tau - 2 * b / dx ** 2)

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

    dts = [2 * dx ** 2 / b, dx ** 2 / b, 0.5 * dx ** 2 / b,
           0.4 / (2 * b / dx ** 2 + 1 / tau),
           0.3 / (2 * b / dx ** 2 + 1 / tau),
           0.2 / (2 * b / dx ** 2 + 1 / tau),
           0.1 / (2 * b / dx ** 2 + 1 / tau),
           0.01 / (2 * b / dx ** 2 + 1 / tau)]
    dts = [0.3 / (2 * b / dx ** 2 + 1 / tau)]

    if verbose:
        assert is_dimensionless(dts)
        assert is_dimensionless(r_of_x)
        assert is_dimensionless(b)
        assert is_dimensionless(x)
        assert is_dimensionless(dx)

    def solve(x0, plot=True, t_max=300, dt=0.01):
        print(f"tau = {tau}")
        print(f"dt = {dt}")

        u0_1 = np.zeros(x.shape)
        u0_2 = np.zeros(len(x))

        u0 = u0_1 + u0_2

        print(f"{u0[0]}. Dimension {get_dimensions(u0[0])}")

        if verbose:
            assert is_dimensionless(t_max)
            assert is_dimensionless(u0_1)
            assert is_dimensionless(u0_2)
            assert is_dimensionless(u0)

        def synaptic_input_profile(t):
            return dirac_delta_unitless(x0=x0, t0=to_SI(0.1 * ms, second), I_e=to_SI(1.5 * pampere, ampere), x=x, t=t, dx=dx, dt=dt, tau_m=tau, r_of_x=r_0)

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

                assert is_dimensionless(A[0, 0])
                assert is_dimensionless(V[0])
                assert is_dimensionless(I_syn[0] / c_m)
            result = A @ V + I_syn / c_m

            if verbose:
                assert is_dimensionless(result[0])

            return result

        def forward_euler(f: Callable[[float, np.ndarray], np.ndarray], t_span, V0: np.ndarray, dt: float,
                          saved_frames=1):

            t0, tf = t_span

            # Number of Euler steps
            num_steps = int(np.ceil((tf - t0) / dt))

            # Number of saved states
            save_every = int(np.ceil(num_steps / saved_frames))
            num_save = num_steps // save_every + 1

            # Preallocate
            times = np.zeros(num_save)
            sol = np.zeros((num_save, len(V0)))
            t = t0
            V = V0.copy()

            times[0] = t0
            sol[0] = V

            if verbose:
                assert is_dimensionless(V0[0])
                assert is_dimensionless(V)
                assert is_dimensionless(sol[0][0])

            for step in range(1, num_steps + 1):
                dt_step = min(dt, tf - t)

                # Forward Euler step
                V = V + dt_step * f(t, V)

                t += dt_step

                # If another save_every bunch
                if step % save_every == 0:
                    iteration = step // save_every
                    print(
                        f"[f {iteration}/{num_save}]: Reached step {step} from {num_steps} ({100 * step / num_steps:.2f}%)")
                    times[iteration] = t
                    sol[iteration] = V

            if verbose:
                assert is_dimensionless(sol[0])
                assert is_dimensionless(sol[0][0])
                assert is_dimensionless(V0[0])
                assert is_dimensionless(V)

            return times, sol

        # 3. Solve the ODE system using manual runge kutta
        times, V_s = forward_euler(cylindrical_cable_equation, t_span=(0, t_max), V0=u0, dt=dt,
                                   saved_frames=400)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if plot:
            plot_difussion_unitless(times=times, V_s=V_s, simulation_params=si_units, dt=dt)

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:

    x0_values = [0 * um, 6.5 * um, 250 * um, 490 * um, 500 * um]
    x0_values = [to_SI(250*um, meter)]
    calls = list(itertools.product(x0_values, dts))

    results = Parallel(
        n_jobs=-3 if len(calls) > 2 else 1,  # use all CPU cores
        backend="loky",  # process-based (default)
        verbose=10)(delayed(solve)(x2, dt=dt, t_max=t_max, plot=True) for x2, dt in calls)

def simulate_crank_nicolson_split(x_N=301, dt=to_SI(0.001 * ms), x0=to_SI(250*um), t_max=to_SI(30 * ms), verbose=True):
    simulation_params = default_params.with_property(t=t_max, N=x_N)
    N = simulation_params.N
    si_units = simulation_params.to_numerical()

    # 1. Parameters
    dx = si_units.dx
    tau = si_units.tau

    # Spatial domain and initial condition
    x = si_units.x
    r_0 = si_units.r0

    b = si_units.b

    difussion = np.ones(N - 1) * b / dx ** 2
    difussion_decay = np.ones(N) * (- 2 * b / dx ** 2)
    leak = np.ones(N) * (- 1 / tau)

    # Sparse tridiagonal matrix
    D = diags(
        diagonals=[difussion, difussion_decay, difussion],
        offsets=[-1, 0, 1],
        format="lil"
    )

    L_matrix = diags(
        diagonals=[leak],
        offsets=[0],
        format="lil"
    )

    # ensure boundary conditions automatically in A matrix
    D[0, 0] = - 2 * b / dx ** 2
    D[0, 1] = 2 * b / dx ** 2
    D[-1, -2] = 2 * b / dx ** 2
    D[-1, -1] = - 2 * b / dx ** 2

    def synaptic_input_profile(t, x0, dt):
        return dirac_delta_unitless(x0=x0, t0=to_SI(1 * ms), x=x, t=t, dx=dx, dt=dt, I_e=to_SI(1.5 * pampere),
                                    tau_m=tau, r_of_x=r_0)

    def crank_nicolson_unitless_split(x0, t_span, V0, dt=to_SI(0.01 * ms), saved_frames=1, verbose=False):

        """
        Solve dV/dt = A V using Crank-Nicolson.

        (I - dt/2 A) V[n+1] = (I + dt/2 A) V[n]
        """

        t0, tf = t_span

        # Number of time steps
        num_steps = int(np.ceil((tf - t0) / dt))

        # Saving
        save_every = int(np.ceil(num_steps / saved_frames))
        num_save = num_steps // save_every + 1

        times = np.zeros(num_save)
        sol = np.zeros((3, num_save, len(V0)))

        # Convert dt if using quantities
        dt = float(dt)

        # Identity matrix
        I = eye(D.shape[0], format="csc")

        A = D + L_matrix
        # Crank-Nicolson matrices
        L = (I - 0.5 * dt * A).tocsc()
        R = (I + 0.5 * dt * A).tocsc()

        # Factorize once
        solve = factorized(L)

        # Initial condition
        t = t0
        V = V0.copy()

        times[0] = t
        sol[:, 0] = V

        for step in range(1, num_steps + 1):

            dt_step = min(dt, tf - t)

            # RHS
            input_t_and_t_half = 1 / 2 * dt * (
                        synaptic_input_profile(t=t, x0=x0, dt=dt / 2) + synaptic_input_profile(t=t + dt / 2, x0=x0,
                                                                                               dt=dt / 2))
            rhs = R @ V + input_t_and_t_half

            # Solve:
            # (I - dt/2 A) V_new = rhs
            V_n_plus_1 = solve(rhs)

            t += dt_step

            if step % save_every == 0:
                iteration = step // save_every

                if verbose:
                    print(
                        f"[CN {iteration}/{num_save}] "
                        f"step {step}/{num_steps}"
                    )
                V_mid = V + V_n_plus_1
                leak = 0.5 * L @ V_mid
                difussion = 0.5 * D @ V_mid
                times[iteration] = t
                sol[0, iteration] = V_n_plus_1
                sol[1, iteration] = leak
                sol[2, iteration] = difussion

            V = V_n_plus_1

        return times, sol

    def solve(x0, plot=True, t_max=to_SI(300 * ms), dt=to_SI(0.01 * ms)):

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        u0 = np.zeros(len(x))

        print(f"V0: {u0[0]}. Dimension {get_dimensions(u0[0])}")

        if verbose:
            assert is_dimensionless(u0)

        # 3. Solve the ODE via crank nicolson (x0, t_span, V0, dt =0.01 * ms, saved_frames=1, plot=True, verbose=False):
        times, V_s = crank_nicolson_unitless_split(t_span=(0, t_max), x0=x0, V0=u0, dt=dt, saved_frames=400)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if plot:
            plot_difussion_unitless_split(times=times, V_s=V_s, simulation_params=si_units, dt=dt,
                                          sim_type="Crank-Nicolson split")

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:

    return solve(x0=x0, dt=dt, t_max=t_max, plot=True)

def simulate_crank_nicolson_unitless(x_N=301, dt=to_SI(0.001 * ms), x0=to_SI(250*um), t_max=to_SI(30 * ms), verbose=True):
    simulation_params = default_params.with_property(t=t_max, N=x_N)
    si_units = simulation_params.to_numerical()

    # 1. Parameters
    dx = si_units.dx
    tau = si_units.tau

    # Spatial domain and initial condition
    x = si_units.x
    r_0 = si_units.r0

    r_of_x = np.ones(len(simulation_params.x)) * r_0

    b = si_units.b

    if verbose:
        mu = b * dt / dx ** 2
        if mu < 0.5:
            print(f"mu = {mu:.5f}: very safe")
        elif mu < 2:
            print(f"mu = {mu:.5f}: usually OK")
        else:
            print(f"mu = {mu:.5f}: CN may oscillate")

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

    def synaptic_input_profile(t, x0, dt):
        return dirac_delta_unitless(x0=x0, t0=to_SI(1 * ms), x=x, t=t, dx=dx, dt=dt, I_e=to_SI(1.5 * pampere),
                                    tau_m=tau, r_of_x=r_0)

    def crank_nicolson(x0, t_span, V0, dt=to_SI(0.01 * ms), saved_frames=1, verbose=False):

        """
        Solve dV/dt = A V using Crank-Nicolson.

        (I - dt/2 A) V[n+1] = (I + dt/2 A) V[n]
        """

        t0, tf = t_span

        # Number of time steps
        num_steps = int(np.ceil((tf - t0) / dt))

        # Saving
        save_every = int(np.ceil(num_steps / saved_frames))
        num_save = num_steps // save_every + 1

        times = np.zeros(num_save) * t0
        sol = np.zeros((num_save, len(V0)))

        # Convert dt if using quantities
        dt = float(dt)

        # Identity matrix
        I = eye(A.shape[0], format="csc")

        # Crank-Nicolson matrices
        L = (I - 0.5 * dt * A).tocsc()
        R = (I + 0.5 * dt * A).tocsc()

        # Factorize once
        solve = factorized(L)

        # Initial condition
        t = t0
        V = V0.copy()

        times[0] = t
        sol[0] = V

        for step in range(1, num_steps + 1):

            dt_step = min(dt, tf - t)

            # RHS
            input_t_and_t_half = 1 / 2 * dt * (
                    synaptic_input_profile(t=t, x0=x0, dt=dt / 2) + synaptic_input_profile(t=t + dt / 2, x0=x0,
                                                                                           dt=dt / 2))

            rhs = R @ V + input_t_and_t_half
            # Solve:
            # (I - dt/2 A) V_new = rhs
            V = solve(rhs)

            t += dt_step

            if step % save_every == 0:
                iteration = step // save_every

                if verbose:
                    print(
                        f"[CN {iteration}/{num_save}] "
                        f"step {step}/{num_steps}"
                    )

                times[iteration] = t
                sol[iteration] = V

        return times, sol

    def solve(x0, plot=True, t_max=to_SI(300 * ms), dt=to_SI(0.01 * ms)):

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        u0 = np.zeros(len(x))

        print(f"{u0[0]}. Dimension {get_dimensions(u0[0])}")

        if verbose:
            assert is_dimensionless(u0)

        # 3. Solve the ODE via crank nicolson (x0, t_span, V0, dt =0.01 * ms, saved_frames=1, plot=True, verbose=False):
        times, V_s = crank_nicolson(t_span=(0, t_max), x0=x0, V0=u0, dt=dt, saved_frames=400)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if plot:
            # def plot_difussion_unitless(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler")
            plot_difussion_unitless(times=times, V_s=V_s, dt=dt, simulation_params= si_units, sim_type="Crank-Nicolson unitless")

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:

    return solve(x0=x0, dt=dt, t_max=t_max, plot=True)


# TODO: Normalize the way Dayan has it. ie = Ie τm δ(x)δ(t)/ 2πa. A is the radius! We have a slightly different formulation. But Still
def dirac_delta(x0: Quantity, t0: Quantity, tau_m: Quantity, I_e: Quantity, x: np.ndarray[Quantity], r_of_x: Quantity,
                t: Quantity, dx: Quantity, dt: Quantity) -> np.ndarray[Quantity]:
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

    result = dirac_delta_unitless(
        x0 = to_SI(x0, meter),
        t0 = to_SI(t0, second),
        tau_m = to_SI(tau_m, second),
        I_e = to_SI(I_e, ampere),
        x = to_SI(x, meter),
        r_of_x=to_SI(r_of_x, meter),
        t = to_SI(t, second),
        dx = to_SI(dx, meter),
        dt = to_SI(dt, second)
    ) * ampere / meter**2

    assert have_same_dimensions(result[0], 1 * mampere / cm ** 2)
    return result

def dirac_delta_unitless(
        x0: float,
        t0: float,
        tau_m: float,
        I_e: float,
        x: np.ndarray[np.float32],
        r_of_x: float,
        t: float,
        dx: float,
        dt: float) -> np.ndarray[np.float32]:
    if t0 - t < 0 or t0 - t >= dt:
        return np.zeros(len(x))



    result = np.zeros(len(x))

    delta_xt = 1.0 / (dx * dt)

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

    print(f"Inserted delta at t={t}. t0={t0}, I_e={I_e}. dt = {dt: .3e} s. Total = {np.sum(result):.6e}")
    return result

def plot_difussion_unitless_split(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler"):

    plot_difussion_unitless(times = times, V_s = V_s.sum(axis=0), simulation_params = simulation_params, dt=dt, verbose=verbose, sim_type=sim_type)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(12, 13),
        gridspec_kw={'height_ratios': [3, 3, 4]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = simulation_params.x * meter / um
    r_of_x = np.ones(len(simulation_params.x)) * meter / um
    times = times * second / ms
    V_s = V_s * volt / mV

    im_1 = ax1.imshow(
        V_s[0],
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        # vmax=0.05
    )

    fig.colorbar(im_1, ax=ax1, label="Voltage (mV)")

    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("t [ms]")

    im_2 = ax2.imshow(
        V_s[1],
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        # vmax=0.05
    )

    fig.colorbar(im_2, ax=ax2, label="Voltage (mV)")

    ax2.set_xlabel("x [μm]")
    ax2.set_ylabel("t [ms]")


    desired_distances = [0, 250, 500]
    desired_distances = [0, 250, 500]
    for desired_distance in desired_distances:
        i =  np.searchsorted(x, desired_distance)

        ax3.plot(
            times,
            V_s[1, :, i],
            label=f"Leak x = {x[i]:.0f} μm",
            alpha=0.6
        )

        ax3.plot(
            times,
            V_s[2, :, i],
            label=f"Difussion x = {x[i]:.0f} μm",
            alpha = 0.6
        )

        ax3.plot(
            times,
            V_s[0, :, i],
            label=f"Sum x = {x[i]:.0f} μm"
        )

    ax3.set_xlabel("t [ms]")
    ax3.set_ylabel("V [mV]")
    ax3.set_title("Voltage at selected positions")
    ax3.legend()

    fig.suptitle(
        f"{sim_type.capitalize()} simulation for cable equation in cylinder model \n"
        f"x = [{x[0]} - {x[-1]:.2f}] μm, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={simulation_params.dx * meter / um: .5f} μm, dt={dt: .3e} s \n"
        f""
    )

    plt.tight_layout()
    plt.show()


def plot_difussion_unitless(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler"):
    if verbose:
        assert is_dimensionless(V_s[0][0])
        assert is_dimensionless(times[0])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(10, 11),
        gridspec_kw={'height_ratios': [3, 2, 1]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = simulation_params.x * meter / um
    r_of_x = np.ones(len(simulation_params.x)) * meter / um
    times = times * second / ms
    V_s = V_s * volt / mV

    print("x:", x[0], x[-1], len(x))
    print("times:", times[0], times[-1], len(times))
    print("V_s:", V_s.shape)

    im = ax1.imshow(
        V_s,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        # vmax=0.05
    )

    fig.colorbar(im, ax=ax1, label="Voltage (mV)")

    ax1.set_title(
        f"{sim_type.capitalize()} simulation for cable equation in cylinder model \n"
        f"x = [{x[0]} - {x[-1]:.2f}] μm, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={simulation_params.dx * meter /um: .5f} μm, dt={dt: .3e} s \n"
        f""
    )
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("t [ms]")


    print("x: ", x)
    indices = [0, 3, 20, 50, 80, 99]
    desired_distances = [0, 250, 500]
    for desired_distance in desired_distances:
        i =  np.searchsorted(x, desired_distance)
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} μm",
            alpha=0.6,
            lw=2
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

    # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def plot_difussion_solution(times, x, r_of_x, V_s, verbose=True, dt: Quantity = 0 * ms):
    if verbose:
        assert have_same_dimensions(x[0], 1 * um)
        assert have_same_dimensions(r_of_x[0], 1 * um)
        assert have_same_dimensions(V_s[0][0], 1 * mV)
        assert have_same_dimensions(times[0], 1 * ms)

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
        V_s,
        aspect='auto',
        origin='lower',
        extent=[x[0], x[-1], times[0], times[-1]],
        cmap='cividis_r'
        # vmax=0.05
    )

    fig.colorbar(im, ax=ax1, label="Voltage (mV)")

    ax1.set_title(
        f"Cable equation. Max V = {np.max(V_s):.4f} mV. dt={dt / second : .3e} second"
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

    # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


class CylindricalDendriticTreePDECase(unittest.TestCase):

    def test_difussion_pde_triagonal_matrix_in_SI_units(self):
        run_simulation_unitless(x_N = 301, verbose=False, t_max=to_SI(1 * ms))

    def test_diffusion_pdes(self, debugging=False):

        #Ns = [101, 201, 301, 401, 501]
        #Ns = [11]
        Ns = [301, 501, 1001]

        Parallel(
            n_jobs=1 if debugging else -1,
            backend="loky",
            verbose=10
        )(
            delayed(run_simulation_unitless)(x_N=N, t_max=to_SI(10 * ms))
            for N in Ns
        )

    def test_crank_nicolson_split(self):

        Ns = [501, 1001]
        x0_values = [to_SI(250 * um)]
        #dts = [to_SI(1E-6 * second), to_SI(1E-7 * second), to_SI(5E-8 * second), to_SI(1E-8 * second), to_SI(5E-9 * second), to_SI(1E-9 * second)]
        dts = [to_SI(1E-6 * second)]
        t_maxs = [to_SI(25 * ms)]
        calls = list(itertools.product(Ns, x0_values, dts, t_maxs))

        Parallel(
            n_jobs=1,
            #n_jobs=1 if is_debugging() else -1,
            backend="loky",
            verbose=10)(
            delayed(simulate_crank_nicolson_split)(x_N=N, t_max=t_max, dt=dt, x0=x0, verbose=True)
            for N, x0, dt, t_max in calls
        )

    def test_crank_nicolson_unitless(self):

        import sys

        debugging = (
                sys.gettrace() is not None
                or "pydevd" in sys.modules
        )

        print("debugging =", debugging)
        print("trace =", sys.gettrace())
        print("pydevd =", "pydevd" in sys.modules)

        Ns = [301, 501, 1001]
        dts = [to_SI(1E-7 * second), to_SI(1E-8 * second), to_SI(5E-9 * second)]

        Ns = [101, 301, 501, 701, 1001]
        x0_values = [to_SI(250 * um)]
        dts = [to_SI(1E-7 * second)]
        t_maxs = [to_SI(25 * ms)]

        calls = list(itertools.product(Ns, x0_values, dts, t_maxs))

        Parallel(
            n_jobs=1 if debugging else -1,
            backend="loky",
            verbose=10)(
            delayed(simulate_crank_nicolson_unitless)(x_N=N, t_max=t_max, dt=dt, x0=x0, verbose=True)
            for N, x0, dt, t_max in calls
        )


    def test_difussion_pde_crank_nicolson(self, verbose=False, x_N=101, t_max=1 * ms):

        c_m = 1 * uF / cm ** 2
        Rm = 2 * 1E4 * ohm * cm ** 2
        gL = 1 / Rm
        ra = 100 * ohm * cm

        # 1. Parameters
        L = 500.0 * um
        N = x_N
        dx = L / (N - 1)  # um

        tau = c_m / gL  # ms

        assert have_same_dimensions(tau, 1 * second)

        print("tau=", tau)

        # Spatial domain and initial condition
        x = np.linspace(0, L, N)

        r_0 = 2 * um

        r_of_x = np.ones(len(x)) * r_0
        b = r_0 / (2 * c_m * ra)
        assert have_same_dimensions(b, meter ** 2 / second)

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

        # analyse_eigenvalues_generalized_locally_toeplitz_matrix(A)

        A = A.tocsr().toarray() * (1 / second)

        dts = [0.4 / (2 * b / dx ** 2 + 1 / tau)]

        def synaptic_input_profile(t, x0, dt):
            return dirac_delta(x0=x0, t0=0.1 * ms, x=x, t=t, dx=dx, dt=dt, I_e=1.5 * pampere, tau_m=tau, r_of_x=r_0)

        def crank_nicolson(x0, t_span, V0, dt=0.01 * ms, saved_frames=1, verbose=False):

            """
            Solve dV/dt = A V using Crank-Nicolson.

            (I - dt/2 A) V[n+1] = (I + dt/2 A) V[n]
            """

            t0, tf = t_span

            # Number of time steps
            num_steps = int(np.ceil((tf - t0) / dt))

            # Saving
            save_every = int(np.ceil(num_steps / saved_frames))
            num_save = num_steps // save_every + 1

            times = np.zeros(num_save) * t0
            sol = np.zeros((num_save, len(V0)))

            # Identity matrix
            I = eye(A.shape[0], format="csc")

            # Crank-Nicolson matrices
            L = I - 0.5 * dt * A
            R = I + 0.5 * dt * A

            # Factorize once
            solve = factorized(L)

            # Initial condition
            t = t0
            V = V0.copy()

            times[0] = t
            sol[0] = V

            for step in range(1, num_steps + 1):

                dt_step = min(dt, tf - t)

                # RHS
                rhs = R @ V + 1 / 2 * dt * (
                            synaptic_input_profile(t=t, x0=x0, dt=dt) + synaptic_input_profile(t=t + dt, x0=x0, dt=dt))

                # Solve:
                # (I - dt/2 A) V_new = rhs
                V = solve(rhs)

                t += dt_step

                if step % save_every == 0:
                    iteration = step // save_every

                    if verbose:
                        print(
                            f"[CN {iteration}/{num_save}] "
                            f"step {step}/{num_steps}"
                        )

                    times[iteration] = t
                    sol[iteration] = V

            return times, sol

        def solve(x0, plot=True, t_max=300 * ms, dt=0.01 * ms):

            print(f"tau = {tau}")
            print(f"dt = {dt}")

            if is_dimensionless(t_max):
                t_max = t_max * ms

            u0 = np.zeros(len(x)) * mV

            print(f"{u0[0] / volt}. Dimension {get_dimensions(u0[0] / volt)}")

            if verbose:
                assert have_same_dimensions(1 * volt / second, 1 * ampere / uF)
                assert have_same_dimensions(u0, 1 * volt)
                assert have_same_dimensions(u0, 1 * volt)

            # 3. Solve the ODE via crank nicolson (x0, t_span, V0, dt =0.01 * ms, saved_frames=1, plot=True, verbose=False):
            times, V_s = crank_nicolson(t_span=(0 * ms, t_max), x0=x0, V0=u0, dt=dt, saved_frames=400)

            if verbose:
                assert have_same_dimensions(times[0], 1 * ms)
                assert have_same_dimensions(V_s[0][0], 1 * mV)
                assert have_same_dimensions(V_s[0], 1 * mV)

            if plot:
                plot_difussion_solution(times=times, x=x, r_of_x=r_of_x, V_s=V_s, dt=dt)

            return np.max(V_s), np.argmax(V_s[1]), dt, x0


        x0_values = [0 * um, 6.5 * um, 250 * um, 490 * um, 500 * um]
        x0_values = [250 * um]
        calls = list(itertools.product(x0_values, dts))

        results = Parallel(
            n_jobs=-3 if len(calls) > 2 else 1,  # use all CPU cores
            backend="loky",  # process-based (default)
            verbose=10)(delayed(solve)(x2, dt=dt, t_max=t_max, plot=True) for x2, dt in calls)


    def test_crank_nicolson_split_single(self):
        simulate_crank_nicolson_split(x_N=2001, t_max = to_SI(1 * ms), verbose=True, dt=to_SI(0.5 * 1E-7 * second), x0=to_SI(250 * um))

    def test_difussion_pde_crank_nicolson_unitless(self, verbose=False, x_N=2001, t_max=to_SI(30 * ms)):

        x0_values = []
        dts = [to_SI(1E-7 * second),  to_SI(0.5E-6 * second), to_SI(1E-6 * second), to_SI(0.5 * 1E-7 * second)]
        calls = list(itertools.product(x0_values, dts))

        simulate_crank_nicolson_split(x_N = x_N, x0=to_SI(250 * um), dt=dts[-1], t_max=to_SI(t_max), verbose=verbose)
        simulate_crank_nicolson_unitless(x_N = x_N, x0=to_SI(250 * um), dt=dts[-1], t_max=to_SI(t_max), verbose=verbose)

        '''
        results = Parallel(
            n_jobs=-3 if len(calls) > 2 else 1,  # use all CPU cores
            backend="loky",  # process-based (default)
            verbose=10)(delayed(solve)(x2, dt=dt, t_max=t_max, plot=True) for x2, dt in calls)
        '''


if __name__ == '__main__':
    unittest.main()
