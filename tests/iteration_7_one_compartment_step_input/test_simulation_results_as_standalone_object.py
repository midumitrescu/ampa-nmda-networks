import unittest

import brian2
import numpy as np
from brian2 import *

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot

from numpy.testing import assert_array_equal, assert_allclose

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class SimulationResultsPickled(unittest.TestCase):

    def test_rate_is_correctly_extracted(self):
        config = {

            Experiment.KEY_IN_TESTING: True,
            Experiment.KEY_SIMULATION_METHOD: "euler",
            "panel": "Exemplifying up and down states without NMDA input",

            Experiment.KEY_SIMULATION_CLOCK: 0.5,

            "g": 1,
            "g_ampa": 2.4e-06,
            "g_gaba": 2.4e-06,
            "g_nmda": 2e-05,

            "up_state": {
                "N_E": 1000,
                "gamma": 1.2,
                "nu": 100,
            },
            "down_state": {
                "N_E": 100,
                "gamma": 3,
                "nu": 10,
            },

            Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],

            "t_range": [[0, 2000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }
        results = sim_and_plot(Experiment(config))

        print(len(results.rate_monitor_rates()))
        print(mean(results.rate_monitor_rates()))
        print(sum(results.rate_monitor_rates()))

        self.assertEqual(4000, len(results.rate_monitor_t()))
        self.assertAlmostEqual(3999.0, sum(results.rate_monitor_t() * ms / second))
        self.assertAlmostEqual(0.99975, mean(results.rate_monitor_t() * ms / second))

        self.assertEqual(4000, len(results.rate_monitor_rates()))
        self.assertAlmostEqual(3999.9999999999995, sum(results.rate_monitor_rates()))
        self.assertAlmostEqual(0.999999999999, mean(results.rate_monitor_rates()))

        assert_allclose([108.9773691, 488.40268401, 805.23989379, 488.40268401,
                         108.9773691, 0., 0., 0.,
                         0., 0.], results.rate_monitor_rates()[570:580])
        assert_allclose([0., 0., 0., 0.,
                         108.9773691, 488.40268401, 805.23989379, 488.40268401,
                         108.9773691, 0.], results.rate_monitor_rates()[2215:2225])

    def test_spikes_are_correctly_extracted(self):

        for i in range(2):
            config = {

                Experiment.KEY_IN_TESTING: True,
                Experiment.KEY_SIMULATION_METHOD: "euler",
                "panel": "Exemplifying up and down states without NMDA input",

                Experiment.KEY_SIMULATION_CLOCK: 0.5,
                "g": 1,
                "g_ampa": 2.4e-06,
                "g_gaba": 2.4e-06,
                "g_nmda": 0,

                "up_state": {
                    "N_E": 1000,
                    "gamma": 1.1,
                    "nu": 100,
                },
                "down_state": {
                    "N_E": 100,
                    "gamma": 3,
                    "nu": 10,
                },
                Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,

                "t_range": [[0, 2000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
            }
            results = sim_and_plot(Experiment(config))


            self.assertEqual(13, len(results.spikes.t))
            self.assertEqual(13, len(results.spikes.i))
            '''
            assert_allclose([16. ,   49. ,   72.5,  113.5,  124.5,  156. ,  261. ,  301. ,
            336.5,  446. ,  450.5, 1059. , 1067. , 1176.5, 1201. , 1212. ,
           1282.5, 1307.5, 1437. ], results.spikes.t)

            self.assertEqual(1, len(results.spikes.all_values.items()))
            self.assertEqual(["t"], list(results.spikes.all_values.keys()))
            assert_allclose([16. ,   49. ,   72.5,  113.5,  124.5,  156. ,  261. ,  301. ,
            336.5,  446. ,  450.5, 1059. , 1067. , 1176.5, 1201. , 1212. ,
           1282.5, 1307.5, 1437. ], results.spikes.all_values["t"][0] / ms)
           '''


if __name__ == '__main__':
    unittest.main()
