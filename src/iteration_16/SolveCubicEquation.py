import sys
import unittest

import matplotlib
import numpy as np
from brian2 import mV, have_same_dimensions, nS, Hz, ms, farad, nF, second, is_dimensionless, Mohm
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from iteration_16.model import calibrated_configuration, ConductanceDiffusionSimulationConfig

logger.remove()  # remove default handler

logger.add(
    sys.stderr,
    level="DEBUG",
    format="{time:HH:mm:ss.SSS} | {level} | {message}",
    backtrace=True,
    diagnose=True
)

logger.add(
    "ui_debug.log",
    level="DEBUG",
    rotation="5 MB",
    retention="3 days",
    backtrace=True,
    diagnose=True
)

def polynomial_x(r, cfg: ConductanceDiffusionSimulationConfig, gamma, sigma_v):
    """
    Returns P(x) where x = g r
    """

    g = cfg.k * cfg.w_gaba * cfg.tau_gaba * cfg.N_I
    x = gamma * g * r

    gL = cfg.g_L
    C = cfg.membrane_capacitance

    # synaptic weights
    K_e = cfg.w_ampa * cfg.tau_ampa * (cfg.e_ampa - cfg.e_L)**2
    K_i = cfg.w_gaba * cfg.tau_gaba * (cfg.e_gaba - cfg.e_L)**2

    assert have_same_dimensions(K_e , 1 * farad * mV**2)
    assert have_same_dimensions(K_i, 1 * farad * mV ** 2)

    # shorthand
    te = cfg.tau_ampa
    ti = cfg.tau_gaba

    # coefficients
    a3 = 2 * sigma_v**2 * te * ti * (1 + gamma)**3

    assert have_same_dimensions(a3, 1*ms**2 * mV**2)

    a2 = (
        2 * sigma_v**2 * C * (te + ti) * (1 + gamma)**2
        - (1 + gamma) * (K_e * ti + gamma * K_i * te)
    )
    assert have_same_dimensions(a2, 1 * nS * ms ** 2 * mV ** 2)

    a1 = (
        2 * sigma_v**2 * C**2 * (1 + gamma)
        - (
            K_e * (ti * gL + C)
            + gamma * K_i * (te * gL + C)
        )
    )
    assert have_same_dimensions(a1, 1 * nS**2 * ms ** 2 * mV ** 2)

    a0 = (
        2 * sigma_v**2
        * (
            te * ti * gL**3
            + C * (te + ti) * gL**2
            + C**2 * gL
        )
    )
    assert have_same_dimensions(a0, 1 * nS **3 * ms ** 2 * mV**2)

    result = a3*x**3 + a2*x**2 + a1*x + a0
    assert have_same_dimensions(result[0], 1 * nS **3 * ms ** 2 * mV**2)

    return result / (nS **3 * ms ** 2 * mV**2)

def E_0(r, cfg: ConductanceDiffusionSimulationConfig, gamma):

    if is_dimensionless(r):
        r = r * Hz

    gL = cfg.g_L
    EL = cfg.e_L
    Ee = cfg.e_ampa
    Ei = cfg.e_gaba

    g = cfg.g()
    ge = g * r
    gi = gamma * g * r
    g0 = gL + ge + gi

    num = gL * EL + ge * Ee + gi * Ei

    return num / g0

def sigma_sq(r, cfg: ConductanceDiffusionSimulationConfig, gamma):
    gL = cfg.g_L
    Ee = cfg.e_ampa
    Ei = cfg.e_gaba
    C = cfg.membrane_capacitance

    ge = cfg.g() * r
    gi = gamma * cfg.g() * r

    assert have_same_dimensions(ge[1], 1 * nS)
    assert have_same_dimensions(gi[1], 1 * nS)

    g0 = gL + ge + gi
    assert have_same_dimensions(g0, 1 * nS)

    sigma_e_sq = 0.5 * cfg.w_ampa * ge
    sigma_i_sq = 0.5 * cfg.w_gaba * gi
    assert  have_same_dimensions(sigma_e_sq, 1 * nS**2)
    assert  have_same_dimensions(sigma_i_sq, 1 * nS**2)

    E_0_comp = E_0(r, cfg, gamma)
    assert have_same_dimensions(E_0_comp[1], 1 * mV)


    sigma_term_e = sigma_e_sq / (g0 ** 2) * (Ee - E_0_comp) ** 2 * cfg.tau_gaba /  (cfg.tau_ampa + C / g0)
    sigma_term_i = sigma_i_sq / (g0 ** 2) * (Ei - E_0_comp) ** 2 * cfg.tau_gaba / (cfg.tau_gaba + C / g0)

    assert is_dimensionless(cfg.tau_gaba /  (cfg.tau_ampa + C / g0))
    assert have_same_dimensions(sigma_term_e[1], 1 * mV**2)
    assert have_same_dimensions(sigma_term_i[1], 1 * mV**2)

    sigma_v_squared = sigma_term_e + sigma_term_i
    return sigma_v_squared


def plot_poly(gamma=0.5, sigma_v=2 * mV, cfg: ConductanceDiffusionSimulationConfig = calibrated_configuration):

    r = np.linspace(0, 50000, 1000) * Hz
    g = cfg.k * cfg.w_gaba * cfg.tau_gaba * cfg.N_I
    x = gamma * g * r

    y = polynomial_x(x, cfg, gamma, sigma_v)

    plt.figure(figsize=(6,4))
    plt.plot(r, y)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("r (Hz)")
    plt.ylabel("P(r)")
    plt.title("Cubic intersection landscape")
    plt.show()


class TripleExplorer:

    def __init__(self, cfg):

        self.cfg = cfg

        # parameters
        self.gamma = 0.5

        self.r = np.linspace(0, 2000, 2000) * Hz

        self.mu_v = - 47.61595645 * mV
        self.sigma_v = 1.9 * mV

        self.fig = plt.figure(figsize=(12, 12))

        graphs = 3
        sliders = 7

        gs = GridSpec(
            graphs + sliders, 1,
            height_ratios=[3] * graphs +  [0.3] * sliders
        )

        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])

        ax_gamma = self.fig.add_subplot(gs[graphs + 1])
        ax_C = self.fig.add_subplot(gs[graphs + 2])
        ax_g_L = self.fig.add_subplot(gs[graphs + 3])

        ax_we = self.fig.add_subplot(gs[graphs + 4])
        ax_wi = self.fig.add_subplot(gs[graphs + 5])

        ax_r_max = self.fig.add_subplot(gs[graphs + 6])

        # lines
        self.line_E, = self.ax1.plot([], [], lw=2)
        self.line_S, = self.ax2.plot([], [], lw=2)
        self.line_P, = self.ax3.plot([], [], lw=2)

        self.root_scatter = self.ax3.scatter([], [], color="red", zorder=10)

        self.ax1.set_title(r"$E_0(r, \gamma)$")
        self.ax2.set_title(r"$\sigma_v(r, \gamma)$")
        self.ax3.set_title("P(gr) with zero crossings")
        self.ax3.axhline(0, color="black", lw=1)

        self.ax1.set_ylabel("mV")
        self.ax2.set_ylabel(r"$mV^2$")
        self.ax3.set_ylabel("P(x)")
        self.ax3.set_xlabel("r (Hz)")

        self.ax1.axhline(self.mu_v / mV, color="black", lw=1, linestyle="--")
        self.ax2.axhline(self.sigma_v / mV, color="black", lw=1, linestyle="--")
        self.ax3.axhline(0, color="black", lw=1, linestyle="--")

        self.s_gamma = Slider(ax_gamma, "γ", 0.1, 2.0, valinit=0.5)
        self.s_C     = Slider(ax_C, "C", 0.1, valinit=0.5, valmax=10, valstep=0.1)
        self.s_g_L     = Slider(ax_g_L, "$g_L$ (nS)", 5, valinit=20, valmax=100, valstep=1)
        self.s_we    = Slider(ax_we, r"$w_e$ (nS)", 0.01, valinit=0.5, valmax=10)
        self.s_wi    = Slider(ax_wi, r"$w_i$ (nS)", 0.01, valinit=0.5, valmax=10)
        self.s_r_max    = Slider(ax_r_max, r"$r_\mathrm{max}$", valmin=0, valinit=1000, valstep=50, valmax=2000)

        self.fig_title = self.fig.suptitle(self.get_title())


        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)

        for s in [self.s_gamma, self.s_C, self.s_g_L, self.s_we, self.s_wi, self.s_r_max]:
            s.on_changed(self.update)

        self.update(None)

    def E0(self, r):
        return E_0(r, self.cfg, self.gamma)

    def sigma2(self, r):
        return sigma_sq(r, self.cfg, self.gamma)

    def P(self, r):
        return polynomial_x(r, self.cfg, self.gamma, self.sigma_v)

    def update(self, val):
        try:
            logger.debug("update triggered")
            self.gamma = self.s_gamma.val

            new_C = self.s_C.val * nF
            new_g_L = self.s_g_L.val * nS
            new_w_e = self.s_we.val * nS
            new_w_i = self.s_wi.val * nS

            r_max_new = self.s_r_max.val
            self.r = np.linspace(0, r_max_new, 2000) * Hz

            self.cfg = self.cfg.with_property(membrane_capacitance = new_C, w_ampa = new_w_e, w_gaba = new_w_i, g_L = new_g_L)

            E_vals = self.E0(self.r) / mV
            S_vals = self.sigma2(self.r) / (mV ** 2)
            P_vals = self.P(self.r)

            # plot 1: E0
            self.line_E.set_data(self.r, E_vals)

            self.ax1.relim()
            self.ax1.autoscale_view()

            # plot 2: sigma
            self.line_S.set_data(self.r, S_vals)
            self.ax2.relim()
            self.ax2.autoscale_view()

            # plot 3: polynomial
            self.line_P.set_data(self.r, P_vals)

            # zero crossings
            sign_changes = np.where(np.diff(np.sign(P_vals)))[0]
            roots_r = self.r[sign_changes]

            [coll.remove() for coll in list(self.ax3.collections)]
            self.ax3.scatter(roots_r, np.zeros_like(roots_r), color="red")

            self.ax3.relim()
            self.ax3.autoscale_view()

            self.fig_title.set_text(self.get_title())
            self.fig.canvas.draw_idle()
        except Exception as e:
            logger.exception("Exception inside update()", e)

    def get_title(self):
            return ("Search for parameters such that"r"$\mu_v=$"f"{self.mu_v / mV :.2f}, "r"$\sigma_v=$"f"{self.sigma_v / mV :.2f} \n" 
                "Our model:" r"$R_{\mathrm{in}}=$" f"{(1 / self.cfg.g_L) / Mohm : .2f} MΩ, " r"$N_E$="f"{self.cfg.N_E} "r"$N_I$="f"{self.cfg.N_I}\n")



class MyTestCase(unittest.TestCase):

    def test_plot_e_0_sigma_0_p_of_x(self):
        cfg = calibrated_configuration

        gamma = 0.5
        sigma_v = 1.9 * mV

        r = np.linspace(0, 2000, 1000)

        E_vals = E_0(r, cfg, gamma)
        #sigma_vals = sigma_sq(r, cfg, gamma)
        #P_vals = polynomial_x(r, cfg, gamma, sigma_v)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(10, 10),
            sharex=True,
        )

        ax1.plot(r / Hz, E_vals)

        ax1.set_title(r"$E_0(r)$")
        ax1.set_ylabel("mV")

        #ax2.plot(r / Hz, sigma_vals)

        ax2.set_title(r"$\sigma_v^2(r)$")
        ax2.set_ylabel(r"$mV^2$")

        #ax3.plot(r / Hz, P_vals)

        ax3.axhline(
            0,
            color="black",
            lw=1,
        )



        ax3.set_title(r"$P(x)$")
        ax3.set_xlabel("r (Hz)")

        plt.tight_layout()
        plt.show()



    def test_ui(self):
        matplotlib.use("QtAgg")
        cfg = calibrated_configuration.with_property(N_E = 800, N_I = 200)
        ui = TripleExplorer(cfg)
        plt.tight_layout()
        plt.show()

    def test_check_E_0_units_match_brian2(self):
        cfg = calibrated_configuration
        gamma = 1
        r = np.linspace(0, 1000, 101) * Hz

        e_0_np = E_0(r, cfg, gamma)
        test_compare = [None] * len(r)
        g = calibrated_configuration.g()
        one_rate = r[3]
        one_rate_no_unit = one_rate / Hz

        one_rate_computed_with_brian2_units = (g * one_rate) / nS
        one_rate_computed_unitless = (g / (nS * second)) * one_rate_no_unit
        self.assertAlmostEqual(one_rate_computed_with_brian2_units, one_rate_computed_unitless)

        for index, rate in enumerate(r):

            g = calibrated_configuration.g()
            assert have_same_dimensions(g, 1 * nS * ms)

            g_e0 = g * rate
            g_i0 = gamma * g * rate
            assert have_same_dimensions(g_e0, 1 * nS)
            assert have_same_dimensions(g_i0, 1 * nS)

            g_0 = cfg.g_L + g_e0 + g_i0
            current_e_0 = (cfg.g_L * cfg.e_L + g_e0 * cfg.e_ampa + g_i0 * cfg.e_gaba) / g_0
            assert have_same_dimensions(current_e_0, 1 * mV)
            test_compare[index] = current_e_0 / mV


        plt.plot(r / Hz, test_compare, label="Brian 2 units")
        plt.plot(r / Hz, e_0_np / mV, label="no units")
        plt.legend()
        plt.tight_layout()
        plt.show()

        np.testing.assert_array_almost_equal(e_0_np / mV, test_compare)

    def test_check_sigma_v_sq_units_match_brian2(self):
        cfg = calibrated_configuration
        gamma = 1
        r = np.linspace(0, 1000, 101) * Hz


        sigma_np_result = sigma_sq(r, cfg, gamma)


        #plt.plot(r / Hz, test_compare, label="Brian 2 units")
        plt.plot(r / Hz, sigma_np_result / (mV**2), label="no units")
        plt.legend()
        plt.tight_layout()
        plt.show()







if __name__ == '__main__':
    unittest.main()
