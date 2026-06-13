import unittest

import numpy as np
from brian2 import mV, have_same_dimensions, siemens, nS, Hz, ms
from matplotlib import pyplot as plt

from Plotting import show_plots_non_blocking
from iteration_16.model import calibrated_configuration, ConductanceDiffusionSimulationConfig
from iteration_16.second_order_equation import solve_ge0_from_moments, compute_A_B_C


def plot_ge0_quadratic_parabola(A, B, C, num_points=1000, show_roots=True):
    """
    Plots the quadratic:
        f(x) = A x^2 + B x + C

    and optionally highlights roots and vertex.
    """

    # --- domain ---
    # choose range around vertex if possible
    if A != 0:
        x_vertex = - B / (2 * A)
        span = 2.0 * (abs(2 * x_vertex))
        x = np.linspace(x_vertex - span, x_vertex + span, num_points)
    else:
        x = np.linspace(-10, 10, num_points)

    # --- function ---
    y = A * x ** 2 + B * x + C

    # --- vertex ---
    vertex_x = -B / (2 * A) if A != 0 else None
    vertex_y = A * vertex_x ** 2 + B * vertex_x + C if A != 0 else None

    # --- roots ---
    roots = None
    if show_roots:
        D = B ** 2 - 4 * A * C
        if D >= 0 and A != 0:
            sqrtD = np.sqrt(D)
            r1 = (-B + sqrtD) / (2 * A)
            r2 = (-B - sqrtD) / (2 * A)
            roots = (r1, r2)

    # --- plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=r"$A g_{e,0}^2 + B g_{e,0} + C$")

    # zero line
    plt.axhline(0, color="black", linewidth=1)

    # vertex
    if vertex_x is not None:
        plt.scatter(vertex_x, vertex_y, color="red", label="vertex")

    # roots
    if roots is not None:
        plt.scatter(roots, [0, 0], color="green", label="roots")

    plt.title("Quadratic constraint for $g_{e,0}$")
    plt.xlabel("$g_{e,0}$")
    plt.ylabel("f($g_{e,0}$)")
    plt.legend()

    show_plots_non_blocking()

mu_v = -47.61595645 * mV
sigma_v = 1.90531046 * mV

def comp_E_0(config: ConductanceDiffusionSimulationConfig, rho=1, r = np.linspace(0.0, 5000.0, 1000) * Hz):
    g_L = config.g_L
    C = config.membrane_capacitance

    E_L = config.e_L
    E_e = config.e_ampa
    E_i = config.e_gaba

    tau_e = config.tau_ampa
    tau_i = config.tau_gaba

    w_e = config.w_ampa
    w_i = config.w_gaba

    g_e0 = tau_e * w_e * r
    g_i0 = tau_i * w_i * r * rho

    g_tot = g_L + g_e0 + g_i0

    assert have_same_dimensions(g_e0, 1 * nS)
    assert have_same_dimensions(g_i0, 1 * nS)

    assert have_same_dimensions(g_tot, 1 * nS)

    return r, (E_L * g_L + E_e * g_e0 + E_i * g_i0)/g_tot

def comp_sigma_v_squared(config: ConductanceDiffusionSimulationConfig, rho=1, r = np.linspace(0.0, 5000.0, 1000) * Hz):
    g_L = config.g_L
    C = config.membrane_capacitance

    E_L = config.e_L
    E_e = config.e_ampa
    E_i = config.e_gaba

    tau_e = config.tau_ampa
    tau_i = config.tau_gaba

    w_e = config.w_ampa
    w_i = config.w_gaba

    g_e0 = tau_e * w_e * r
    g_i0 = tau_i * w_i * r * rho

    sigma_e = w_e * np.sqrt(tau_e * r / 2)
    sigma_i = w_i * np.sqrt(rho * tau_i * r / 2)

    g_tot = g_L + g_e0 + g_i0

    E_0 = (g_L * E_L + g_e0 * E_e + g_i0 * E_i) / g_tot
    tau_0 = C / g_tot

    sigma_e_sq = (sigma_e / g_tot) ** 2 * (E_e - E_0) ** 2 * tau_e / (tau_e + tau_0)
    sigma_i_sq = (sigma_i / g_tot) ** 2 * (E_i - E_0) ** 2 * tau_i / (tau_i + tau_0)
    sigma_v_sq = sigma_e_sq + sigma_i_sq

    assert  have_same_dimensions(tau_0, 1 * ms)
    assert have_same_dimensions(sigma_v_sq, 1 * mV ** 2)

    print(f"rho{rho}, tau 0: ", tau_0)

    return r, sigma_v_sq, sigma_e, sigma_i



class MyTestCase(unittest.TestCase):
    def test_finding_roots(self):
        print(solve_ge0_from_moments(conductance_diffusion_config=calibrated_configuration, mu_v=mu_v, sigma_v=sigma_v))

    def test_investigate_equation_terms(self):
        mu_v_unitless = mu_v / mV
        sigma_v_unitless = sigma_v / mV

        object_under_test = calibrated_configuration
        gL = object_under_test.g_L / nS

        Ee = object_under_test.e_ampa / mV
        Ei = object_under_test.e_gaba / mV
        EL = object_under_test.e_L / mV

        # --- shifted potentials ---
        bar_EL = EL - mu_v_unitless
        bar_Ee = Ee - mu_v_unitless
        bar_Ei = Ei - mu_v_unitless

        print("\n=== SHIFTED POTENTIALS ===")
        print("bar_E_L =", bar_EL)
        print("bar_E_e =", bar_Ee)
        print("bar_E_i =", bar_Ei)

        # --- useful factors ---
        a = (1 - bar_EL / bar_Ei)
        b = (1 - bar_Ee / bar_Ei)

        print("(1 - bar_EL / bar_Ei)", a)
        print("(1 - bar_EL / bar_Ees) =", b)

        A = (
                4 * sigma_v_unitless ** 2 * b ** 2
                - 2 * bar_Ee ** 2
        )

        B = (
                8 * sigma_v_unitless ** 2 * gL * a * b
                - 2 * gL * bar_EL * bar_Ee
        )

        C = (
                4 * sigma_v_unitless ** 2 * gL ** 2 * a ** 2
                - gL ** 2 * bar_EL ** 2
        )

        print("A: ", A)
        print("B: ", B)
        print("C: ", C)

    def test_understand_negative_discrimintant(self):
        A, B, C = compute_A_B_C(conductance_diffusion_config=calibrated_configuration, mu_v=mu_v, sigma_v=sigma_v)

        print("A: ", A)
        print("B: ", B)
        print("C: ", C)

        # check units
        self.assertTrue(have_same_dimensions(A, 1 * mV ** 2))
        self.assertTrue(have_same_dimensions(B, 1 * mV ** 2 * siemens))
        self.assertTrue(have_same_dimensions(C, 1 * mV ** 2 * siemens ** 2))

        A = A / mV ** 2
        B = B / (mV ** 2 * siemens)
        C = C / (mV * siemens) ** 2

        plot_ge0_quadratic_parabola(A, B, C)

    def test_look_for_parameters_reaching_desired_values(self):
        configuration = ConductanceDiffusionSimulationConfig(
            w_ampa=10000 * nS,
            w_gaba=1 * nS
        )
        A, B, C = compute_A_B_C(conductance_diffusion_config=configuration, mu_v=-50 * mV, sigma_v=0.3 * mV)

        plot_ge0_quadratic_parabola(A, B, C)

    def test_plot_e_0_and_sigma_v_various_amps(self):
        config = calibrated_configuration

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            sharex=True,
            constrained_layout=True,
        )

        for rho in [0.1, 0.3, 1, 2, 4]:

            r, E_0 = comp_E_0(config, rho)
            r, sigma_v_sq, _, _ = comp_sigma_v_squared(config, rho)

            axes[0].plot(r / Hz, E_0 / mV, lw=2)
            axes[1].plot(r / Hz, sigma_v_sq / (mV**2), lw=2, label=f"Rho {rho}")

        axes[0].axhline(
            y=mu_v / mV,
            color="black",
            linestyle="--",
            linewidth=1,
            label=fr"$\mu_V={mu_v / mV:.3f}$ mV"
        )

        var_v = (sigma_v / mV) ** 2
        axes[1].axhline(
            y=var_v,
            color="black",
            linestyle="--",
            linewidth=1,
            label=fr"$\sigma_v^2={var_v:.3f}$ mV $^2$"
        )

        axes[0].set_ylabel(r"$E_0$ [mV]")
        axes[1].set_ylabel(r"$\sigma_v^2$ [mV]")
        axes[1].set_xlabel(r"$r_i$ (Hz)")

        fig.legend()
        show_plots_non_blocking(caller_test_case=self)


    '''
    We are trying to solve the equations: E_0 = given, sigma_v = given.
    We chose initially "weak" synapses. Meaning... AMPA & GABA produce a 0.5 mV of deflection at the soma.
    However, with these values, the maximal 
    
    
    However, for our range of parameter, we find something extremely surprising.
    '''
    def test_plot_e_0_and_sigma_v_various_ks(self):
        config = calibrated_configuration

        g_L = config.g_L
        C = config.membrane_capacitance

        E_L = config.e_L
        E_e = config.e_ampa
        E_i = config.e_gaba

        tau_e = config.tau_ampa
        tau_i = config.tau_gaba

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 7),
            sharex=True,
            constrained_layout=True,
        )

        #for k in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 6, 8]:
        for k in [1]:
            w_e = config.w_ampa
            w_i = k * w_e

            r = np.linspace(0.0, 5000.0, 1000) * Hz

            g_e0 = tau_e * w_e * r
            g_i0 = tau_i * w_i * r

            sigma_e = w_e * np.sqrt(tau_e * r / 2)
            sigma_i = w_i * np.sqrt(tau_i * r / 2)

            g_tot = g_L + g_e0 + g_i0

            E_0 = (g_L * E_L + g_e0 * E_e + g_i0 * E_i) / g_tot
            tau_0 = C / g_tot

            sigma_e_sq = (sigma_e / g_tot) ** 2 * (E_e - E_0) ** 2 * tau_e / (tau_e + tau_0)
            sigma_i_sq = (sigma_i / g_tot) ** 2 * (E_i - E_0) ** 2 * tau_i / (tau_i + tau_0)
            sigma_v_sq = sigma_e_sq + sigma_i_sq

            self.assertTrue(have_same_dimensions(tau_0, 1 * ms))
            self.assertTrue(have_same_dimensions(g_e0, 1 * nS))
            self.assertTrue(have_same_dimensions(g_i0, 1 * nS))

            self.assertTrue(have_same_dimensions(g_tot, 1 * nS))

            self.assertTrue(have_same_dimensions(sigma_v_sq, 1 * mV ** 2))

            print("tau 0: ", tau_0)

            axes[0].plot(r / Hz, E_0 / mV, lw=2)
            axes[1].plot(r / Hz, sigma_v_sq / (mV ** 2), lw=2, label=f"k {k}")

            axes[2].plot(r / Hz, g_e0 / nS, lw=2, label=r"$g_{e,0} $"f"k {k} [nS]")
            axes[2].plot(r / Hz, g_i0 / nS, lw=2, label=r"$g_{e,0} $"f"k {k} [nS]")

        axes[0].axhline(
            y=mu_v / mV,
            color="black",
            linestyle="--",
            linewidth=1,
            label=fr"$\mu_V={mu_v / mV:.3f}$ mV"
        )

        var_v = (sigma_v / mV)**2
        axes[1].axhline(
            y= var_v,
            color="black",
            linestyle="--",
            linewidth=1,
            label=fr"$\sigma_v^2={var_v:.3f}$ mV $^2$"
        )

        axes[0].set_title(
            r"$E_0=\frac{g_L E_L + g_{e,0} E_e + g_{i,0} E_i}"
            r"{g_L + g_{e,0} + g_{i,0}}$",
            fontsize=12,
        )
        axes[0].set_ylabel(r"$E_0$ [mV]")

        axes[1].set_title(
            r"$\sigma_v^2="
            r"\left(\frac{\sigma_e^2}{g_L+g_{e,0}+g_{i,0}}\right)^2"
            r"(E_e-E_0)^2"
            r"\frac{\tau_e}{\tau_e+\frac{C}{g_L+g_{e,0}+g_{i,0}}}$"
            r"$+\left(\frac{\sigma_i^2}{g_L+g_{e,0}+g_{i,0}}\right)^2"
            r"(E_i-E_0)^2"
            r"\frac{\tau_i}{\tau_i+\frac{C}{g_L+g_{e,0}+g_{i,0}}}$",
            fontsize=11,
        )
        axes[1].set_ylabel(r"$\sigma_v^2$ [mV]")
        axes[1].set_xlabel(r"$r_i$ (Hz)")

        for ax in axes:
            ax.legend()
        show_plots_non_blocking(caller_test_case=self)

    def test_plot_contours(self):

        for w_amp in [1, 3, 5, 10]:

            cfg = calibrated_configuration

            # Parameters
            g_L = cfg.g_L
            C = cfg.membrane_capacitance

            E_L = cfg.e_L
            E_e = cfg.e_ampa
            E_i = cfg.e_gaba

            tau_e = cfg.tau_ampa
            tau_i = cfg.tau_gaba

            w_e = cfg.w_ampa * w_amp
            w_i = cfg.w_gaba * w_amp

            # grid
            r_i_vals = np.linspace(0, 15_000, 300) * Hz
            r_e_vals = np.linspace(0, 200_000, 300) * Hz

            Ri, Re = np.meshgrid(r_i_vals, r_e_vals)

            g_e0 = tau_e * w_e * Re
            g_i0 = tau_i * w_i * Ri

            sigma_ge = w_e * np.sqrt(tau_e * Re / 2)
            sigma_gi = w_i * np.sqrt(tau_i * Ri / 2)

            g_tot = g_L + g_e0 + g_i0

            E0 = (
                         g_L * E_L
                         + g_e0 * E_e
                         + g_i0 * E_i
                 ) / g_tot

            tau0 = C / g_tot

            sigma_v2 = (
                    (sigma_ge / g_tot) ** 2
                    * (E_e - E0) ** 2
                    * tau_e / (tau_e + tau0)
                    +
                    (sigma_gi / g_tot) ** 2
                    * (E_i - E0) ** 2
                    * tau_i / (tau_i + tau0)
            )

            plt.figure(figsize=(8, 6))

            c1 = plt.contour(
                Ri / Hz,
                Re / Hz,
                E0 / mV,
                levels=[mu_v / mV],
                colors="blue",
                linewidths=2,
            )

            c2 = plt.contour(
                Ri / Hz,
                Re / Hz,
                sigma_v2 / (mV**2),
                levels=[(sigma_v / mV)**2],
                colors="red",
                linewidths=2,
            )
            import matplotlib.lines as mlines
            legend_handles = [
                mlines.Line2D([], [], color="blue", linewidth=2,
                              label=r"$E_0 = $" f"{mu_v / mV:.3f} mV"),
                mlines.Line2D([], [], color="red", linewidth=2,
                              label=r"$\sigma_v^2 = $ "f"{(sigma_v/mV)**2 :.3f} mV""$^2$"),
            ]

            plt.xlabel(r"$N_I \cdot r_i$ (Hz)")
            plt.ylabel(r"$N_E \cdot r_e$ (Hz)")

            plt.title(
                rf"$E_0={mu_v / mV}$ mV (blue), "
                rf"$\sigma_V={sigma_v / mV}$ mV (red), w={w_amp}"
            )

            plt.legend(handles=legend_handles)

            plt.show()

            plt.contourf(
                Ri / Hz,
                Re / Hz,
                sigma_v2 / (mV**2),
                levels=10,
                cmap="viridis"
            )
            plt.colorbar(label=r"$\sigma_V$ (mV)")

            plt.contour(
                Ri / Hz,
                Re / Hz,
                E0 / mV,
                levels=[mu_v / mV],
                colors="white",
                linewidths=3,
            )
            plt.show()

    def test_understand_components_of_e_0(self):
        pass
        


if __name__ == '__main__':
    unittest.main()
