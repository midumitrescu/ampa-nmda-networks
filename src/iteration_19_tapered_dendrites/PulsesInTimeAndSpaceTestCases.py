import unittest

import numpy as np
from brian2 import ohm, cm, uF, um, ms, second, Hz, meter
from brian2.units.allunits import pampere

from scipy.stats import uniform, expon

from Plotting import show_plots_non_blocking
from iteration_19_tapered_dendrites.conical_data import create_delta_pulses, ConicalCableParameters
from iteration_19_tapered_dendrites.data import to_SI

import matplotlib.pyplot as plt

rm = 2 * 1E4 * ohm * cm ** 2

default_params = ConicalCableParameters(c_m=1 * uF / cm ** 2,
                                 rm=rm,
                                 gL=1 / rm,
                                 ra=100 * ohm * cm,
                                 L=500.0 * um,
                                 N=101,
                                 r_at_0=2 * um,
                                 r_at_L=0.5 * um,
                                 I_e=150 * pampere)

class MyTestCase(unittest.TestCase):
    def test_something(self):
        x_N = 101
        dt_ = to_SI(1E-8 * second)
        t_max = to_SI(1 * second)
        L = to_SI(500 * um)
        I_e = to_SI(150 * pampere)
        simulation_params = default_params.with_SI_properties(t=t_max, N=x_N, dt=dt_, L=L, I_e=I_e)

        r_i = 50 * Hz

        t_distribution = expon(scale=1.0 / r_i)
        x_distribution = uniform(loc=0, scale = L)

        res = create_delta_pulses(
            x=simulation_params.x,
            t_max=t_max,
            dt=dt_,
            x_distribution=x_distribution,
            t_distribution=t_distribution,
        )

        x = np.linspace(
            0,
            simulation_params.L,
            simulation_params.N
        )

        r = simulation_params.radius(x)

        fig, (ax_scatter, ax_dendrite) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        # Cable boundaries
        ax_dendrite.fill_between(
            x / um,
            -r / um,
            r / um,
            color="lightgray",
            alpha=0.6
        )

        ax_dendrite.plot(
            x / um,
            r / um,
            color="black"
        )

        ax_dendrite.plot(
            x / um,
            -r / um,
            color="black"
        )

        # Synaptic events
        ax_dendrite.scatter(
            res[1] / um,
            np.zeros_like(res[0]),
            color="red",
            s=20,
            marker="|"
        )

        ax_dendrite.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
        ax_dendrite.set_ylabel(r"radius $[\mu\mathrm{m}]$")


        ax_scatter.scatter(
            res[1] / um,
            res[0] / ms,
            s=15,
            marker="|",
            linewidths=1.5,
            color="black"
        )

        ax_scatter.set_xlim(0, simulation_params.L / um)
        ax_scatter.set_ylim(0, t_max / ms)

        ax_scatter.set_xlabel(r"$x\;[\mu\mathrm{m}]$")
        ax_scatter.set_ylabel(r"$t\;[\mathrm{ms}]$")
        ax_scatter.set_title("Space-time raster of synaptic events")


        fig.tight_layout()
        show_plots_non_blocking()


if __name__ == '__main__':
    unittest.main()
