import unittest

from sympy.physics.control.control_plots import plt
from sympy.physics.quantum.identitysearch import np

from Plotting import show_plots_non_blocking
from iteration_17_statistics_of_s_variable.SDiffusionConfig import SDiffusionConfig, SDiffusionSimulation


class MyTestCase(unittest.TestCase):


    def test_run_once(self):
        object_under_test = SDiffusionConfig()

        dt = object_under_test.dt

        result = SDiffusionSimulation.run(object_under_test)

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            constrained_layout=True
        )

        ax = axs[0, 0]

        t_ms = result.t

        window = slice(0, 20000)

        ax.plot(
            t_ms[window],
            result.x[window],
            lw=1,
            label="x"
        )

        ax.plot(
            t_ms[window],
            result.s[window],
            lw=1,
            label="s"
        )

        ax.set_title("Sample trajectory")
        ax.set_xlabel("t [ms]")
        ax.legend()

        ax = axs[0, 1]

        '''
        lags = np.arange(len(C_emp)) * dt

        ax.plot(
            lags / ms,
            C_emp,
            label="simulation"
        )

        ax.plot(
            lags / ms,
            C_theory,
            "--",
            lw=2,
            label="OU theory"
        )
        
        ax.set_xlim(0, 10 * tau_rise / ms)
        '''
        ax.set_title("Autocovariance")
        ax.set_xlabel(r"$\tau$ [ms]")
        ax.set_ylabel(r"$C_x(\tau)$")

        ax.legend()

        show_plots_non_blocking(caller_test_case=self)


if __name__ == '__main__':
    unittest.main()
