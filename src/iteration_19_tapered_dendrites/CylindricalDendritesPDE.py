import itertools
import unittest
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from brian2 import cm, uF, ohm, um, Quantity, is_dimensionless, get_dimensions, volt, have_same_dimensions, mV, meter, \
    umeter, msecond, mvolt
from brian2.units import second, ms
from brian2.units.allunits import ampere, mampere, pampere
from joblib import Parallel, delayed
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

from Plotting import show_plots_non_blocking
from iteration_19_tapered_dendrites.data import CableParameters, to_SI, NumericalCableParameters

rm = 2 * 1E4 * ohm * cm ** 2

default_params = CableParameters(c_m=1 * uF / cm ** 2,
                                 rm=rm,
                                 gL=1 / rm,
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

def run_simulation_unitless(x_N, t_max,verbose=False, saved_frames = 1200):
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
            plot_difussion_unitless(times=times, V_s=V_s, p=si_units, dt=dt)

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

        # for splits in [50, 100, 200, 250, 500, 750, 1000, 1250, 1500]:

    x0_values = [0 * um, 6.5 * um, 250 * um, 490 * um, 500 * um]
    x0_values = [to_SI(250*um, meter)]
    calls = list(itertools.product(x0_values, dts))

    results = Parallel(
        n_jobs=-3 if len(calls) > 2 else 1,  # use all CPU cores
        backend="loky",  # process-based (default)
        verbose=10)(delayed(solve)(x2, dt=dt, t_max=t_max, plot=True) for x2, dt in calls)

def simulate_crank_nicolson_split(x_N=301, dt=to_SI(0.001 * ms), x0=to_SI(250*um), t0=to_SI(0.1 * ms), t_max=to_SI(30 * ms), saved_frames = 1200, verbose=True):
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
        return dirac_delta_unitless(x0=x0, t0=t0, x=x, t=t, dx=dx, dt=dt, I_e=to_SI(1.5 * pampere),
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
            input_t_and_t_half = 1 / 2 * dt_step * (
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
                V_mid = 0.5 * (V + V_n_plus_1)
                leak = dt_step * L @ V_mid
                difussion = dt_step * D @ V_mid
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
        times, V_s = crank_nicolson_unitless_split(t_span=(0, t_max), x0=x0, V0=u0, dt=dt, saved_frames=saved_frames)

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

def constant_synaptic_input_profile(t, x0, dt, p: NumericalCableParameters):
    return dirac_delta_unitless(x0=x0, t0=t+0.1 * dt, x=p.x, t=t, dx=p.dx, dt=dt, I_e=p.I_e, tau_m=p.tau, r_of_x=p.r0)

def synaptic_input_profile(t, x0, dt, p: NumericalCableParameters):
    return dirac_delta_unitless(x0=x0, t0=to_SI(0.1 * ms), I_e=p.I_e, x=p.x, t=t, dx=p.dx, dt=dt, tau_m=p.tau, r_of_x=p.r0)

def simulate_crank_nicolson_constant_input_unitless(x_N=301, dt=to_SI(0.001 * ms), x0=to_SI(250*um), t0=to_SI(0.1 * ms), t_max=to_SI(30 * ms), saved_frames = 1200, verbose=True):
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
                    constant_synaptic_input_profile(t=t, x0=x0, dt=dt / 2, p=si_units) + synaptic_input_profile(t=t + dt / 2, x0=x0,
                                                                                           dt=dt / 2, p=si_units))

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
        times, V_s = crank_nicolson(t_span=(0, t_max), x0=x0, V0=u0, dt=dt, saved_frames=saved_frames)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if plot:
            # def plot_difussion_unitless(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler")
            plot_difussion_unitless(times=times, V_s=V_s, dt=dt, p= si_units, sim_type="Crank-Nicolson unitless")

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

    return solve(x0=x0, dt=dt, t_max=t_max, plot=True)

def crank_nicolson(x0, t_span, V0, A, p: NumericalCableParameters, dt=to_SI(0.01 * ms), saved_frames=1, verbose=False):

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
                    synaptic_input_profile(t=t, x0=x0, dt=dt, p=p) + synaptic_input_profile(t=t + dt, x0=x0,
                                                                                           dt=dt, p=p))

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

def simulate_crank_nicolson_unitless_closed_cylinder(x_N=301, dt=to_SI(0.001 * ms), x0=to_SI(250 * um), t0=to_SI(0.1 * ms), t_max=to_SI(30 * ms), saved_frames = 1200, verbose=True):
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

    def solve(x0, plot=True, t_max=to_SI(300 * ms), dt=to_SI(0.01 * ms)):

        print(f"tau = {tau}")
        print(f"dt = {dt}")

        u0 = np.zeros(len(x))

        print(f"{u0[0]}. Dimension {get_dimensions(u0[0])}")

        if verbose:
            assert is_dimensionless(u0)

        # 3. Solve the ODE via crank nicolson (x0, t_span, V0, dt =0.01 * ms, saved_frames=1, plot=True, verbose=False):
        times, V_s = crank_nicolson(t_span=(0, t_max), x0=x0, V0=u0, dt=dt, A=A, saved_frames=saved_frames, p=si_units)

        if verbose:
            assert is_dimensionless(times[0])
            assert is_dimensionless(V_s[0][0])
            assert is_dimensionless(V_s[0])

        if plot:
            # def plot_difussion_unitless(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler")
            plot_difussion_unitless(times=times, V_s=V_s, dt=dt, p= si_units, sim_type="Crank-Nicolson unitless")

        return np.max(V_s), np.argmax(V_s[1]), dt, x0

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
        dt: float,
        verbose=False,
) -> np.ndarray[np.float32]:
    if t0 - t < 0 or t0 - t >= dt:
        return np.zeros(len(x))

    result = np.zeros(len(x))

    delta_xt = 1.0 / (dx * dt)

    i_e = I_e * tau_m / (np.pi * r_of_x**2) * delta_xt

    if x0 <= x[0]:
        result[0] = i_e

    elif x0 >= x[-1]:
        result[-1] = i_e

    else:
        idx = np.searchsorted(x, x0)

        # Exact match -> inject into a single node
        if np.isclose(x[idx], x0, rtol=0.0, atol=1e-12 * dx):
            result[idx] = i_e

        else:
            i_e_index = idx - 1

            alpha = (x0 - x[i_e_index]) / dx

            result[i_e_index] = i_e * (1.0 - alpha)
            result[i_e_index + 1] = i_e * alpha

    if verbose:
        print(f"Inserted delta at t={t}. t0={t0}, I_e={I_e}. dt = {dt: .3e} s. Total = {np.sum(result):.6e}")
    return result

def plot_difussion_unitless_split(times, V_s, simulation_params: NumericalCableParameters, dt, verbose=True, sim_type="forward Euler"):

    plot_difussion_unitless(times = times, V_s = V_s[0], p= simulation_params, dt=dt, verbose=verbose, sim_type=sim_type)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1,
        figsize=(12, 13),
        gridspec_kw={'height_ratios': [3, 3, 3, 3, 3]},
        sharex=True
    )

    x = simulation_params.x * meter / umeter
    times = times * second / msecond
    V_s = V_s * volt / mvolt
    desired_distances = [0, 250]
    indexes = [np.searchsorted(x, desired_distance) for desired_distance in desired_distances]

    for index in indexes:
        ax1.plot(
            times,
            V_s[1, :, index],
            label=f"Leak x = {x[index]:.0f} "r"$\mu$ m",
            alpha=0.6
        )

        ax2.plot(
            times,
            V_s[2, :, index],
            label=f"Difussion x = {x[index]:.0f} "r"$\mu$ m",
            alpha=0.6
        )

        summ = V_s[1, :, index] + V_s[2, :, index]
        ax3.plot(
            times,
            summ,
            label=f"Sum x = {x[index]:.0f} "r"$\mu$ m"
        )

        ax4.plot(
            times,
            V_s[0, :, index],
            label=f"CN x = {x[index]:.0f} "r"$\mu$ m"
        )

        ax5.plot(
            times,
            V_s[0, :, index] - summ,
            label=f"Diff CN - summ x = {x[index]:.0f} "r"$\mu$ m"
        )


    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.legend()
        ax.set_ylabel("V [mV]")


    ax1.set_title("Leak")
    ax2.set_title("Difussion")
    ax3.set_title("Leak + Difussion")
    ax4.set_title("Actual Crank-Nicolson")
    ax5.set_title("Difference Crank-Nicolson - (Leak + Difussion)")

    ax5.set_xlabel("t [ms]")

    fig.suptitle(
        f"Split {sim_type.capitalize()} simulation for cable equation in cylinder model \n"
        f"x = [{x[0]} - {x[-1]:.2f}] " r"$\mu$"f"m, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={simulation_params.dx * meter / um: .5f} " r"$\mu$"f"m, dt={dt: .3e} s \n"
        f""
    )

    plt.tight_layout()
    plt.show()

def show_difussion_simulation_as_image(times, V_s, p: NumericalCableParameters, dt, x0 =250 * um, t0=0.1 * ms, verbose=True, sim_type="forward Euler"):
    if verbose:
        assert is_dimensionless(V_s[0][0])
        assert is_dimensionless(times[0])

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=(11, 12),
        gridspec_kw={'height_ratios': [2, 2, 2, 1]}
    )

    # ============================================
    # Top: space-time voltage map
    # ============================================
    x = p.x * meter / um
    x0 = x0 / um
    t0 = t0 / ms
    r_of_x = np.ones(len(p.x)) * meter / um
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
        f"x = [{x[0]} - {x[-1]:.2f}] "r"$\mu$"f"m, split in {len(x)} nodes. \n"
        f"Max V = {np.max(V_s):.4f} mV. dx={p.dx * meter / um: .5f} "r"$\mu$"f"m, dt={dt: .3e} s \n"
        f""
    )
    ax1.set_xlabel(r"x [$\mu$m]")
    ax1.set_ylabel("t [ms]")


    print("x: ", x)

    desired_positions = [100, x0, 500]
    for desired_distance in desired_positions:
        i =  np.searchsorted(x, desired_distance)
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} "r"$\mu$ m",
            alpha=0.6,
            lw=2
        )

    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("V [mV]")
    ax2.set_title("Voltage at selected positions")
    ax2.legend()

    # ============================================
    # Second to last: Gaussian around injection point!?
    # ============================================
    index_x0 = np.searchsorted(x, x0)


    # which is time of injection?
    v_tot = V_s.sum(axis=1)
    v_t_diff = np.diff(v_tot)
    t_index = np.argwhere(v_t_diff > 0)
    t_index = t_index[0][0] if len(t_index) > 0 else 0

    taum_ms = p.tau * second / ms

    for t_i in range(t_index, t_index+6):

        if len(x) < 200:
            ax3.plot(
                x,
                V_s[t_i, :],
                label=r"t/$\tau$" f"= {times[t_i] / taum_ms :.4f}",
                alpha=0.6,
                lw=2
            )
        else:
            offs = 250 if len(x) > 2000 else 100
            ax3.plot(
                x[index_x0-offs: index_x0+offs],
                V_s[t_i, index_x0-offs: index_x0+offs],
                label=r"t/$\tau$" f"= {times[t_i] / taum_ms :.4f}",
                alpha=0.6,
                lw=2
            )


    ax3.set_xlabel(r"x [$\mu$m]")
    ax3.set_ylabel("V [mV]")
    ax3.legend()
    ax3.set_title(r"Voltage at time steps around $\delta$-pulse")


    # ============================================
    # Bottom: cone geometry
    # ============================================

    r = r_of_x

    # Cone walls
    ax4.plot(x, r, 'k', linewidth=2)
    ax4.plot(x, -r, 'k', linewidth=2)

    # Fill cone
    ax4.fill_between(
        x,
        -r,
        r,
        color='gray',
        alpha=0.3
    )

    # Center axis y=0
    ax4.axhline(
        0,
        color='gray',
        linestyle=':',
        linewidth=1.5
    )

    # Vertical line at x=0
    ax4.axvline(
        0,
        color='gray',
        linestyle='--',
        linewidth=1.5
    )

    ax4.set_xlabel("x")
    ax4.set_ylabel("radius")
    ax4.set_title("Cable geometry")

    # ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    show_plots_non_blocking()


def plot_difussion_unitless(times, V_s, p: NumericalCableParameters, dt, x0 =250 * um, t0=0.1 * ms, verbose=True, sim_type="forward Euler"):
    desired_positions = [100, x0 / um, 500]

    show_difussion_simulation_as_image(times=times, V_s = V_s, p=p, dt=dt, x0=x0, t0=t0, verbose=verbose, sim_type=sim_type)
    plot_tuckwell_solution_infinite_cable(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0, times=times, x0=x0, sim_type=sim_type)
    plot_tuckwell_solution_closed_cable(V_s=V_s, desired_positions=desired_positions, p=p, t0=t0, times=times, x0=x0, sim_type=sim_type)


def plot_tuckwell_solution_infinite_cable(V_s, desired_positions, p, t0, times, x0, sim_type="forward Euler"):
    V_s = V_s * volt / mV
    x = p.x * meter / um
    x0 = x0 / um
    t0 = t0 / ms

    times = times * second / ms

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(11, 12),
        gridspec_kw={'height_ratios': [2, 2, 2]}
    )
    lam = p.lambd() * meter / um  # lambda [um]
    tau_m = p.tau * second / msecond  # tau_m [ms]
    D = lam ** 2 / tau_m  # um^2/ms
    t_rel = times - t0
    # avoid t=0 singularity
    heavyside = np.ones_like(t_rel)
    heavyside[t_rel <= 0] = 0
    t_rel[t_rel <= 0] = 1
    for distance, ax in zip(desired_positions, [ax1, ax2, ax3]):
        ax.set_title(f"x={distance: .2f} "r"$\mu$"f"m, x0={x0:.0f} "r"$\mu$"f"m, t0 = {t0} ms")
        i = np.searchsorted(x, distance)
        ax.plot(
            times,
            V_s[:, i],
            label=f"Simulation",
            alpha=0.6,
            lw=2
        )

        dx = distance - x0

        pref_of_t = heavyside * p.I_e * p.R_lambda() / np.sqrt(4 * np.pi * t_rel / tau_m)

        G = pref_of_t * np.exp(-(dx ** 2) / (4 * D * t_rel)) * np.exp(-t_rel / tau_m)

        ax.plot(
            times,
            10 * G,
            lw=2,
            label=f"Tuckwell"
        )

        ax.axvline(x=t0, color='gray', linestyle=':', linewidth=1.5)
        xticks = list(ax.get_xticks())
        xticks.append(t0)
        xticks = sorted(set(xticks))
        ax.set_xticks(xticks)

        # Replace only the t0 tick label
        labels = []
        for tick in xticks:
            if abs(tick - t0) < 1e-12:
                labels.append(r'$t_0$')
            else:
                labels.append(f'{tick:g}')
        ax.set_xticklabels(labels)

        ax.set_xlabel("t [ms]")
        ax.set_ylabel("V(x) [mV]")
        ax.legend()
    fig.suptitle(f"{sim_type.capitalize()} simulation vs Tuckwell theory infinite cable")
    fig.tight_layout()

    show_plots_non_blocking()

def plot_tuckwell_solution_closed_cable(times, V_s, desired_positions, t0, x0, p, sim_type="forward Euler"):

    times = times * second / ms
    V_s = V_s * volt / mV
    x = p.x * meter / um
    x0 = x0 / um
    t0 = t0 / ms
    lam = p.lambd() * meter / um  # lambda [um]
    tau_m = p.tau * second / msecond  # tau_m [ms]
    D = lam ** 2 / tau_m  # um^2/ms

    t_rel = times - t0
    # avoid t=0 singularity
    heavyside = np.ones_like(t_rel)
    heavyside[t_rel <= 0] = 0
    t_rel[t_rel <= 0] = 1

    pref_of_t = heavyside * p.I_e * p.R_lambda() * volt / mvolt * np.exp(- t_rel / tau_m)

    n_max = 11

    '''
V(x,t) = H(t-t_0) * I_e R_\lambda * e^{-(t-t_0)/\tau_m} * \left[ \frac{1}{L} + \frac{2}{L} * 
        \sum_{n=1}^{\infty}  \cos \left(\frac{n\pi x}{L}\right) \cos \left(\frac{n\pi x_0}{L}\right) \exp \left( -D\left(\frac{n\pi}{L}\right)^2 (t-t_0) \right) \right]
    '''
    L = p.L * meter / um
    x_pos, n_s = np.meshgrid(desired_positions, np.arange(1, n_max))
    cos_s = 2 / L * np.cos(n_s * np.pi * x_pos / L) * np.cos(n_s * np.pi * x0 / L)

    t_s, n_ts = np.meshgrid(t_rel, np.arange(1, n_max))
    tu = np.exp(-D * (n_ts * np.pi / L) ** 2 * t_s)

    #V_s_theory = 14.5 * pref_of_t * (1 / L + cos_s.T @ tu)
    V_s_theory = pref_of_t * (1 / L + cos_s.T @ tu)


    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(11, 12),
        gridspec_kw={'height_ratios': [2, 2, 2]}
    )

    for distance, ax, V_th in zip(desired_positions, [ax1, ax2, ax3], V_s_theory):
        ax.set_title(f"x={distance: .2f} "r"$\mu$"f"m, x0={x0:.0f} "r"$\mu$"f"m, t0 = {t0} ms")
        i = np.searchsorted(x, distance)

        mean_ratio = np.mean(V_s[100:, i] / V_th[100:])
        print(f"XXXXXXXXXXXXXXXX mean ratio for {i}-{distance}: {mean_ratio:.5f}")
        ax.plot(
            times,
            1/mean_ratio * V_s[:, i],
            label=f"Simulation",
            alpha=0.6,
            lw=2
        )

        ax.plot(
            times,
            V_th,
            lw=2,
            label=f"Tuckwell"
        )



        ax.axvline(x=t0, color='gray', linestyle=':', linewidth=1.5)
        xticks = list(ax.get_xticks())
        xticks.append(t0)
        xticks = sorted(set(xticks))
        ax.set_xticks(xticks)

        # Replace only the t0 tick label
        labels = []
        for tick in xticks:
            if abs(tick - t0) < 1e-12:
                labels.append(r'$t_0$')
            else:
                labels.append(f'{tick:g}')
        ax.set_xticklabels(labels)

        ax.set_xlabel("t [ms]")
        ax.set_ylabel("V(x) [mV]")
        ax.legend()
    fig.suptitle(f"{sim_type.capitalize()} Simulation vs Tuckwell theory closed rod")
    fig.tight_layout()

    show_plots_non_blocking()


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
    ax1.set_xlabel(r"x [$\mu$m]")
    ax1.set_ylabel("t [ms]")

    indices = [0, 3, 20, 50, 80, 99]
    for i in indices:
        ax2.plot(
            times,
            V_s[:, i],
            label=f"x = {x[i]:.0f} "r"$\mu$m"
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
        t_maxs = [to_SI(1 * ms)]

        calls = list(itertools.product(Ns, x0_values, dts, t_maxs))

        Parallel(
            n_jobs=1 if debugging else -1,
            backend="loky",
            verbose=10)(
            delayed(simulate_crank_nicolson_unitless_closed_cylinder)(x_N=N, t_max=t_max, dt=dt, x0=x0, verbose=True)
            for N, x0, dt, t_max in calls
        )

    def test_crank_nicolson_split_single(self):
        #simulate_crank_nicolson_split(x_N=1001, t_max = to_SI(5 * ms), verbose=True, dt=to_SI(0.5 * 1E-7 * second), x0=to_SI(250 * um))
        dt = to_SI(1E-9 * second)
        x_N = 1001
        t_max = to_SI(3 * ms)
        t0 = to_SI(0.1 * ms)


        l  = lambda _: simulate_crank_nicolson_unitless_closed_cylinder(x_N=x_N, t_max = t_max, verbose=True, t0=t0, dt=dt, saved_frames = 600, x0=to_SI(250 * um))
        l_split = lambda _: simulate_crank_nicolson_split(x_N=x_N, t_max = t_max, verbose=True, t0=t0, dt=dt, saved_frames = 600, x0=to_SI(250 * um))


        results = Parallel(n_jobs=1)(
            delayed(func)(None) for func in [l, l_split]
        )

        # A clue might be here. There is a scale difference at play!!
        '''
        x: 0.0 500.00000000000006 1001
times: 0.0 2.999999999822809 601
V_s: (601, 1001)
        XXXXXXXXXXXXXXXX mean ratio for 200-100: 28.28427
XXXXXXXXXXXXXXXX mean ratio for 500-250.00000000000003: 28.28427
XXXXXXXXXXXXXXXX mean ratio for 1000-500: 28.28427
tau = 0.019999999999999997
dt = 1e-09
V0: 0.0. Dimension 1
x: 0.0 500.00000000000006 1001
times: 0.0 2.999999999822809 601
V_s: (601, 1001)
x:  [0.000e+00 5.000e-01 1.000e+00 ... 4.990e+02 4.995e+02 5.000e+02]
XXXXXXXXXXXXXXXX mean ratio for 200-100: 0.14142
XXXXXXXXXXXXXXXX mean ratio for 500-250.00000000000003: 0.14142
XXXXXXXXXXXXXXXX mean ratio for 1000-500: 0.14142
        '''

        '''
        dts = [
            dt,
            #to_SI(0.5E-7 * second),
            #to_SI(1E-8 * second),
            #to_SI(0.5E-8 * second),
            #to_SI(1E-6 * second),
            #to_SI(0.5E-6 * second),
        ]

        Parallel(n_jobs=len(dts))(
            delayed(simulate_crank_nicolson_split)(
                x_N=x_N,
                t_max=t_max,
                dt=dt,
                x0=to_SI(250 * um),
                verbose=True,
            )
            for dt in dts
        )
        '''

    def test_crank_nicolson_constant_input(self):
        #simulate_crank_nicolson_split(x_N=1001, t_max = to_SI(5 * ms), verbose=True, dt=to_SI(0.5 * 1E-7 * second), x0=to_SI(250 * um))
        dt = to_SI(0.5 * 1E-8 * second)
        x_N = 2001
        t_max = to_SI(3 * ms)
        t0 = to_SI(0.1 * ms)


        l  = lambda _: simulate_crank_nicolson_constant_input_unitless(x_N=x_N, t_max = t_max, verbose=True, t0=t0, dt=dt, saved_frames = 600, x0=to_SI(250 * um))

        results = Parallel(n_jobs=1)(
            delayed(func)(None) for func in [l]
        )

    def test_difussion_pde_crank_nicolson_unitless(self, verbose=False, x_N=2001, t_max=to_SI(30 * ms)):

        x0_values = []
        dts = [to_SI(1E-7 * second),  to_SI(0.5E-6 * second), to_SI(1E-6 * second), to_SI(0.5 * 1E-7 * second)]
        calls = list(itertools.product(x0_values, dts))

        simulate_crank_nicolson_split(x_N = x_N, x0=to_SI(250 * um), dt=dts[-1], t_max=to_SI(t_max), verbose=verbose)
        simulate_crank_nicolson_unitless_closed_cylinder(x_N = x_N, x0=to_SI(250 * um), dt=dts[-1], t_max=to_SI(t_max), verbose=verbose)

        '''
        results = Parallel(
            n_jobs=-3 if len(calls) > 2 else 1,  # use all CPU cores
            backend="loky",  # process-based (default)
            verbose=10)(delayed(solve)(x2, dt=dt, t_max=t_max, plot=True) for x2, dt in calls)
        '''


if __name__ == '__main__':
    unittest.main()
