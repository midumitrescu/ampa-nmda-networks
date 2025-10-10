import unittest

import numpy as np
from brian2 import ms, Hz, mV, defaultclock, ufarad, cm, msiemens, NeuronGroup, PoissonInput, StateMonitor, run

from it_2_richardson import PlotParams, sim, Experiment
import matplotlib.pyplot as plt
from loguru import logger


'''
Tests that make sure it 2 is working. 

The firing rates are so high, that the syncronization between independent neurons is visible (sinusoidal wave)
'''
class It2TestCase(unittest.TestCase):

    def test_initialize_params(self):
        example =  {
            "g": 4.5,
            "nu_ext_over_nu_thr": 0.9,
            "t_range": [1800, 2000],
            "rate_range": [0, 250],
            "rate_tick_step": 50,
        }

        object_under_test = Experiment(example)

        self.assertEqual(10_000, object_under_test.network_params.N_E)
        self.assertEqual(2_500, object_under_test.network_params.N_I)
        self.assertEqual(12_500, object_under_test.network_params.N)

    def test_should_simulate_for_200_ms(self):
        two_hundred_ms_simulation = {
            "panel": self._testMethodName,
            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "rate_range": [0, 250],
            "rate_tick_step": 50,
        }
        object_under_test = Experiment(two_hundred_ms_simulation)
        rate_monitor, spike_monitor, _ = sim(experiment=object_under_test)

        self.assertEqual(200*ms, object_under_test.sim_time)
        self.assertAlmostEqual(199.95, float(rate_monitor.t[-1] * 1000))

    def test_should_simulate_for_300_ms(self):
        two_hundred_ms_simulation = {
            "panel": self._testMethodName,
            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "t_range": [0, 300],
            "rate_range": [0, 250],
            "rate_tick_step": 50,
        }
        object_under_test = Experiment(two_hundred_ms_simulation)
        rate_monitor, spike_monitor, _ = sim(experiment=object_under_test)

        self.assertEqual(300*ms, object_under_test.sim_time)
        self.assertAlmostEqual(299.95, float(rate_monitor.t[-1] * 1000))

    def test_should_simulate_for_100_ms_from_t_range_field(self):
        simulation_for_1000_ms = {
            "panel": self._testMethodName,
            "g": 0,
            "nu_ext_over_nu_thr": 10,
            "t_range": [0, 100],
            "rate_range": [0, 250],
            "rate_tick_step": 50,
        }

        self.assertEqual(100 * ms, Experiment(simulation_for_1000_ms).sim_time)

    def test_nu_threshold(self):

        should_fire =  {
            "panel": self._testMethodName,
            "sim_time": 100,
            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "t_range": [50, 100],
            "rate_range": [0, 250],
            "voltage_range": [-70, -35],
            "rate_tick_step": 50,
        }

        object_under_test = Experiment(should_fire)
        self.assertEqual(200 * Hz, object_under_test.nu_thr)
        self.assertEqual(25 * mV, object_under_test.mean_excitatory_input)

        rate_monitor, spike_monitor, v_monitor = sim(experiment=object_under_test)
        #make sure that the network fires, when no inhibition is present
        # this is achieved by taking gamma = 0
        plt.show(block=False)
        plt.close()

    def test_understand_why_nu_ext_is_so_large(self):
        np.random.seed(0)

        for ratio in np.linspace(0.9, 1.1, num=5):
            should_fire = {
                "panel": self._testMethodName,
                "g": 0,
                "sim_time": 120,
                "nu_ext_over_nu_thr": ratio,
                "t_range": [100, 110],
                "voltage_range": [-70, -35],
                "rate_tick_step": 50,
            }

            object_under_test = Experiment(should_fire)

            rate_monitor, spike_monitor, v_monitor = sim(experiment=object_under_test)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plt.show(block=False)
            plt.close()

    def test_simulation_without_self_synapses_should_show_only_poisson_rates(self):

        for mult in np.linspace(0.01, 1, num=10):
            np.random.seed(0)
            no_feedback_synapses = Experiment({
                "panel": self._testMethodName,
                "g": 0,
                "sim_time": 120,
                "sim_clock": 0.05 * ms,
                "nu_ext_over_nu_thr": 1,
                "t_range": [100, 120],
                "rate_range": [0, 250],
                "voltage_range": [-70, -30],
                "rate_tick_step": 50,
                "epsilon": 0,
                "C_ext": 1000,
            })

            delta_v_to_threshold = no_feedback_synapses.neuron_params.theta - no_feedback_synapses.neuron_params.E_leak
            #tau = 20 * ms
            denominator = no_feedback_synapses.neuron_params.J * no_feedback_synapses.network_params.C_ext * no_feedback_synapses.neuron_params.tau
            no_feedback_synapses.neuron_params.nu_thr = delta_v_to_threshold / denominator
            no_feedback_synapses.nu_thr = no_feedback_synapses.neuron_params.nu_thr
            no_feedback_synapses.nu_ext = no_feedback_synapses.nu_ext_over_nu_thr * no_feedback_synapses.nu_thr * mult

            logger.info("Computed nu threshold for testing = {}", no_feedback_synapses.neuron_params.nu_thr)

            rate_monitor, spike_monitor, v_monitor = sim(experiment=no_feedback_synapses)
            # make sure that the network fires, when no inhibition is present
            # this is achieved by taking gamma = 0
            plt.show(block=False)
            plt.close()


    def test_threshold_not_working(self):
        np.random.seed(0)
        defaultclock.dt = 0.05 * ms
        theta = -40 * mV
        V_r = -65 * mV
        E_leak = -65 * mV
        tau_rp = 2 * ms

        C = 1 * ufarad * (cm ** -2)
        g_L = 0.04 * msiemens * (cm ** -2)
        J = 0.5 * mV

        tau_m = C / g_L

        C_Ext = 1_000
        nu_ext = (theta - E_leak) / (J*C_Ext*tau_m)

        neurons = NeuronGroup(10,
                              """
                              dv/dt = 1/C * (-g_L * (v-E_leak)): volt (unless refractory)                          """,
                              threshold="v >= theta",
                              reset="v = V_r",
                              refractory=tau_rp,
                              method="exact")
        neurons.v[:] = -65 * mV
        external_poisson_input = PoissonInput(
            target=neurons, target_var="v", N=C_Ext, rate=nu_ext,
            weight=J
        )

        v_monitor = StateMonitor(source=neurons,
                                 variables="v", record=True)
        duration = 500 * ms
        run(duration, report='text')

        plt.axhline(y = theta/mV, linestyle="dotted", linewidth="0.3", color="k",
                            label="$\\theta$")

        plt.plot()
        for i in range(0, 2):
            plt.plot(v_monitor.t / ms, v_monitor[i].v / mV, label=f"Neuron {i}")

        plt.ylim([-70, -35])
        plt.xlim([210, 220])
        plt.show(block=False)
        plt.close()


if __name__ == '__main__':
    unittest.main()
