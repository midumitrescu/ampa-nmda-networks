import unittest

from brian2 import *
from numpy.testing import assert_allclose

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot, simulate_with_nmda_cut_off_in_down_state

from loguru import logger

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

config = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "euler",
    "panel": "SimulationResultsPickled",

    Experiment.KEY_SIMULATION_CLOCK: 0.5,

    "g": 1,
    "g_ampa": 2.4e-06,
    "g_gaba": 2.4e-06,
    "g_nmda": 2e-05,

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
    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "v_minus_e_gaba"],
    Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_ampa", "I_gaba", "I_nmda"],

    "t_range": [[0, 2000]],
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS]
}

experiment = Experiment(config)


class SimulationResultsPickled(unittest.TestCase):

    def test_experiment_without_currrents_extracts_variables_correctly(self):
        simulate_with_nmda_cut_off_in_down_state(experiment.with_property("panel", self._testMethodName).with_property(Experiment.KEY_CURRENTS_TO_RECORD,
                                                                                                                       []))
        simulate_with_nmda_cut_off_in_down_state(experiment.with_property("panel", self._testMethodName).with_property(Experiment.KEY_CURRENTS_TO_RECORD,
                                                                                                                       None))

    def test_rate_is_correctly_extracted(self):
        results = simulate_with_nmda_cut_off_in_down_state(experiment.with_property("panel", self._testMethodName))

        self.assertEqual(4000, len(results.rate_monitor_t()))
        self.assertAlmostEqual(3999.0, sum(results.rate_monitor_t() * ms / second))
        self.assertAlmostEqual(0.99975, mean(results.rate_monitor_t() * ms / second))

        self.assertEqual(4000, len(results.rate_monitor_rates()))
        self.assertAlmostEqual(70000.0, sum(results.rate_monitor_rates()))
        self.assertAlmostEqual(17.5, mean(results.rate_monitor_rates()))

        assert_allclose([0., 0., 0., 108.9773691,
                         488.40268401, 805.23989379, 488.40268401, 108.9773691,
                         0., 0.], results.rate_monitor_rates()[335:345])

        assert_allclose([0., 0., 0., 108.9773691,
                         488.40268401, 805.23989379, 488.40268401, 108.9773691,
                         0., 0.], results.rate_monitor_rates()[385:395])
        assert_allclose(np.zeros(100), results.rate_monitor_rates()[600:700])

    def test_spikes_are_correctly_extracted(self):
        for i in range(2):
            results = simulate_with_nmda_cut_off_in_down_state(experiment.with_property("panel", self._testMethodName))

            self.assertEqual(35, len(results.spikes.t))
            self.assertEqual(35, len(results.spikes.i))

            assert_allclose(results.spikes.t, [170., 195., 208., 272., 285.5, 291., 294.5, 353.5, 380.5, 398.,
                                               402.5, 422., 462., 489.5, 1056., 1088., 1098., 1104.5, 1128., 1131.5,
                                               1140.5, 1145.5, 1154., 1235.5, 1260., 1275., 1304., 1317.5, 1338., 1348.,
                                               1355.5, 1400.5, 1460., 1474., 1488.5])

            self.assertEqual(1, len(results.spikes.all_values.items()))
            self.assertEqual(["t"], list(results.spikes.all_values.keys()))
            assert_allclose([170., 195., 208., 272., 285.5, 291., 294.5, 353.5, 380.5, 398.,
                             402.5, 422., 462., 489.5, 1056., 1088., 1098., 1104.5, 1128., 1131.5,
                             1140.5, 1145.5, 1154., 1235.5, 1260., 1275., 1304., 1317.5, 1338., 1348.,
                             1355.5, 1400.5, 1460., 1474., 1488.5], results.spikes.all_values["t"][0] / ms)

    def test_voltage_is_properly_extracted(self):
        results = simulate_with_nmda_cut_off_in_down_state(experiment.with_property("panel", self._testMethodName))

        self.assertEqual(4000, len(results.voltages.t))
        self.assertAlmostEqual(mean(results.voltages.v[0]), -48.96536203021542)
        assert_allclose(results.voltages.v[0][500:510],
                        [-41.845082, -41.282747, -41.9245, -42.575567, -42.330862,
                         - 42.210128, -42.771936, -43.027321, -42.3377, -41.240019])

    def test_g_values_are_correctly_extracted(self):
        results = simulate_with_nmda_cut_off_in_down_state(Experiment(config))

        self.assertEqual(4000, len(results.g_s.t))
        self.assertEqual((1, 4000), results.g_s.g_e.shape)
        self.assertEqual((1, 4000), results.g_s.g_i.shape)
        self.assertEqual((1, 4000), results.g_s.g_nmda.shape)

        assert_allclose(results.g_s.g_e[0][100:110] * 1000,
                        [0.440026, 0.44282, 0.473715, 0.465686, 0.486065, 0.477348,
                         0.482811, 0.491708, 0.491181, 0.478786], rtol=1e-6)
        assert_allclose(results.g_s.g_e[0][200:210] * 1000,
                        [0.462357, 0.457168, 0.448476, 0.449157, 0.476068, 0.493851,
                         0.509588, 0.535791, 0.505043, 0.467582], rtol=1e-5)
        self.assertEqual(mean(results.g_s.g_e[0]), 0.0002429621612919005)
        self.assertEqual(max(results.g_s.g_e[0]), 0.0005743223398179232)

        assert_allclose(results.g_s.g_i[0][100:110] * 1000,
                        [0.492125, 0.486694, 0.49702, 0.545565, 0.557974, 0.54808,
                         0.56466, 0.541095, 0.516221, 0.521566], rtol=1e-6)
        assert_allclose(results.g_s.g_i[0][200:210] * 1000,
                        [0.510036, 0.512127, 0.544895, 0.547872, 0.547704, 0.561978,
                         0.517483, 0.496112, 0.489684, 0.477663], rtol=1e-5)
        self.assertEqual(mean(results.g_s.g_i[0]), 0.0002703743586954965)
        self.assertEqual(max(results.g_s.g_i[0]), 0.0006230677890759162)

        assert_allclose(results.g_s.g_nmda[0][100:110] * 1000,
                        [0.014809, 0.014762, 0.014757, 0.014824, 0.014627, 0.014547,
                         0.014482, 0.014652, 0.014859, 0.015039], rtol=1e-4)
        assert_allclose(results.g_s.g_nmda[0][200:210] * 1000,
                        [0.011624, 0.01162, 0.011565, 0.011384, 0.01126, 0.011266,
                         0.011257, 0.011395, 0.011559, 0.011532], rtol=1e-4)
        self.assertEqual(mean(results.g_s.g_nmda[0]), 1.1896835808947705e-05)
        self.assertEqual(max(results.g_s.g_nmda[0]), 1.5540320994275763e-05)

    def test_currents_are_correctly_extracted(self):
        results = sim_and_plot(Experiment(config))

        self.assertEqual(len(results.currents.I_L[0]), 4000)
        assert_allclose(results.currents.I_L[0][100:110],
                        [0.93759652, 0.91855994, 0.91535038, 0.93191551, 0.89616343, 0.88737815,
                         0.8832445, 0.87368038, 0.89457566, 0.92242463])
        assert_allclose(results.currents.I_L[0][500:510],
                        [0.926197, 0.94869, 0.92302, 0.896977, 0.906766, 0.911595,
                         0.889123, 0.878907, 0.906492, 0.950399], rtol=1e-6)
        self.assertEqual(0.6413855187913836, mean(results.currents.I_L[0]))
        self.assertEqual(0.9997855321934335, max(results.currents.I_L[0]))

        self.assertEqual(len(results.currents.I_ampa[0]), 4000)
        assert_allclose(results.currents.I_ampa[0][100:110],
                        [-18.287529, -18.614369, -19.951085, -19.420093, -20.704364,
                         - 20.437933, -20.721724, -21.22115, -20.941816, -20.079991])
        assert_allclose(results.currents.I_ampa[0][500:510],
                        [-20.233881, -19.132777, -19.402354, -20.397722, -19.883691,
                         - 19.732841, -18.897412, -20.763415, -21.521211, -21.265085], rtol=1e-6)
        # avoid small floating point issue
        self.assertAlmostEqual(-10.786422810589743, mean(results.currents.I_ampa[0]))
        self.assertEqual(0, max(results.currents.I_ampa[0]))
        self.assertEqual(-37.33095208816501, min(results.currents.I_ampa[0]))

        self.assertEqual(len(results.currents.I_gaba[0]), 4000)
        assert_allclose(results.currents.I_gaba[0][100:110],
                        [18.917233, 18.476834, 18.828993, 20.893992, 20.870502, 20.38007,
                         20.938232, 19.935035, 19.288299, 19.851125])
        assert_allclose(results.currents.I_gaba[0][500:510],
                        [18.759045, 20.037766, 20.353478, 19.584835, 19.304967, 20.511047,
                         19.085871, 19.0966, 19.018681, 18.493723], rtol=1e-6)
        self.assertEqual(9.596921754233323, mean(results.currents.I_gaba[0]))
        self.assertEqual(22.520609324412863, max(results.currents.I_gaba[0]))
        self.assertEqual(0, min(results.currents.I_gaba[0]))

        self.assertEqual(len(results.currents.I_nmda[0]), 4000)
        self.assertEqual(results.currents.I_nmda.shape, (1, 4000))
        assert_allclose(results.currents.I_nmda[0][100:110],
                        [-0.615472, -0.620547, -0.621515, -0.618211, -0.623037, -0.622832,
                         - 0.621546, -0.632329, -0.633507, -0.630739], rtol=1e-6)
        assert_allclose(results.currents.I_nmda[0][500:510],
                        [-0.57603, -0.570174, -0.57201, -0.573502, -0.56951, -0.566185,
                          - 0.566812, -0.591333, -0.599324, -0.598527], rtol=1e-6)
        self.assertEqual(-0.5717391879916969, mean(results.currents.I_nmda[0]))
        self.assertEqual(0, max(results.currents.I_nmda[0]))
        self.assertEqual(-0.678079614766296, min(results.currents.I_nmda[0]))


if __name__ == '__main__':
    unittest.main()
