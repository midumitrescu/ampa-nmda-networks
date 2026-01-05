import unittest

from brian2 import siemens, nsiemens, psiemens, Hz
import numpy as np

from iteration_10_meanfield_limit.meanfield_simulation import sim_and_plot_meanfield_with_upstate_and_steady_state, \
    simulate_meanfield_with_up_state_and_steady_state, prepare_mean_field, weak_mean_field
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, MeanField
from iteration_8_compute_mean_steady_state.test_wang_numbers import steady_model

from numpy.testing import assert_allclose, assert_equal

from loguru import logger
import sys

# Remove the default logger
logger.remove()

# Add a new handler with INFO level
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

'''
Issue: we want to take the Limit N -> infinity

We have: 80% N -> AMPA, 20% -> GABA, 0.5% -> NMDA
By this, having N -> infinity => N_E -> infinity, N_I -> infinity, N_NMDA -> infinity
'''

meanfield_config = {

    Experiment.KEY_IN_TESTING: True,
    Experiment.KEY_SIMULATION_METHOD: "euler",
    "panel": "NMDA Network with UP/Down and Wang numbers from recurrent connections (network generated input)",

    Experiment.KEY_SIMULATION_CLOCK: 0.05,

    NeuronModelParams.KEY_NEURON_C: 0.5e-3,
    # Wang: Cm = 0.5 nF for pyramidal cells. We are using microFahrad / cm^2 => we need the extra e-3.
    # check tau membrane to be 20 ms!
    # ␶tau m = Cm/gL = 20 ms for excitatory cells

    # VL = -70 mV, the firing threshold Vth = - 50 mV, a reset potential Vreset = -55 mV
    NeuronModelParams.KEY_NEURON_E_L: -70,
    NeuronModelParams.KEY_NEURON_THRESHOLD: -40,
    NeuronModelParams.KEY_NEURON_V_R: -55,
    NeuronModelParams.KEY_NEURON_G_L: 25e-9,  # gL = 25 nS for pyramidal

    # Wang: I used the following values for the recurrent synaptic conductances (in nS)
    # I: we use overall in the configuration siemens / cm ** 2 => respect the scaling from Wang
    # for pyramidal cells: gext,AMPA = 2.1, g recurrent,AMPA = 0.05, grecurrent, NMDA = 0.165, and g recurrent, GABA = 1.3
    # NOTE: weirdly, this is how Wang compensates for the need of more inhibition than excitation in the balance
    SynapticParams.KEY_G_AMPA: 0.05e-9,
    SynapticParams.KEY_G_GABA: 0.04e-9,
    SynapticParams.KEY_G_NMDA: 0.165e-9,

    # ␶where the decay time constant of GABA currents is taken to be tau GABA = 5 ms
    SynapticParams.KEY_TAU_AMPA: 2,
    SynapticParams.KEY_TAU_GABA: 5,

    SynapticParams.KEY_TAU_NMDA_RISE: 2,
    SynapticParams.KEY_TAU_NMDA_DECAY: 100,

    "up_state": {
        "N": 2000,
        "nu": 100,

        "N_nmda": 10,
        "nu_nmda": 10,
    },
    "down_state": {
        "N": 2000,
        "nu": 100,

        "N_nmda": 10,
        "nu_nmda": 10,
    },

    PlotParams.KEY_PLOT_SMOOTH_WIDTH: 10,
    Experiment.KEY_SELECTED_MODEL: single_compartment_with_nmda_and_logged_variables,
    Experiment.KEY_STEADY_MODEL: steady_model,
    Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda", "g_e"],
    #Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["g_nmda"],

    Experiment.KEY_CURRENTS_TO_RECORD: ["I_L", "I_nmda", "I_fast"],

    "t_range": [[0, 4000]],
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                        PlotParams.AvailablePlots.HIDDEN_VARIABLES]
}

class MyTestCase(unittest.TestCase):

    def test_meanfield_scaling(self):
        wang_config = Experiment(meanfield_config)

        self.assertAlmostEqual(0.05e-9, weak_mean_field(wang_config.synaptic_params.g_ampa / siemens, 2000, 2000))
        self.assertAlmostEqual(0.04e-9, weak_mean_field(wang_config.synaptic_params.g_gaba / siemens, 2000, 2000))
        self.assertAlmostEqual(0.165e-9, weak_mean_field(wang_config.synaptic_params.g_nmda / siemens, 2000, 2000))


    def test_up_with_weak_meanfield_scaling_with_wang_numbers(self):
        #results = simulate_meanfield_with_up_state_and_steady_state(Experiment(meanfield_config))
        results = sim_and_plot_meanfield_with_upstate_and_steady_state(Experiment(meanfield_config))

        self.assertAlmostEqual(0.05e-9, results.mean_field_values.g_ampa / siemens)
        self.assertAlmostEqual(0.04e-9, results.mean_field_values.g_gaba / siemens)
        self.assertAlmostEqual(1, results.mean_field_values.g_x)

    '''
    Shows that mean field process works.
    '''
    '''
    Poisson Input AMPA (3200, 100. * hertz, 25. * psiemens),        Poisson Input GABA (800, 100. * hertz, 20. * psiemens),         Poisson Input NMDA (10, 10. * hertz, 0.5)
    Poisson Input AMPA (4800, 100. * hertz, 16.66666667 * psiemens),Poisson Input GABA (1200, 100. * hertz, 13.33333333 * psiemens),Poisson Input NMDA (10, 10. * hertz, 0.3333333333333333)
    Poisson Input AMPA (6400, 100. * hertz, 12.5 * psiemens),       Poisson Input GABA (1600, 100. * hertz, 10. * psiemens),        Poisson Input NMDA (10, 10. * hertz, 0.25)
    Poisson Input AMPA (16000, 100. * hertz, 5. * psiemens),        Poisson Input GABA (4000, 100. * hertz, 4. * psiemens),         Poisson Input NMDA (10, 10. * hertz, 0.1)
    Poisson Input AMPA (32000, 100. * hertz, 2.5 * psiemens),       Poisson Input GABA (8000, 100. * hertz, 2. * psiemens),         Poisson Input NMDA (10, 10. * hertz, 0.05)
    Poisson Input AMPA (160000, 100. * hertz, 0.5 * psiemens),      Poisson Input GABA (40000, 100. * hertz, 0.4 * psiemens),       Poisson Input NMDA (10, 10. * hertz, 0.01)
    '''
    def test_mean_field_progression(self):
        meanfield_results: list[MeanField] = []
        for scaling in [2, 3, 4, 10, 20, 100]:
            result = sim_and_plot_meanfield_with_upstate_and_steady_state(
                prepare_mean_field(Experiment(meanfield_config),
                                   N=scaling * 2000, N_reference=2000))
            meanfield_results.append(result.mean_field_values)

        assert_allclose([item.g_ampa / psiemens for item in meanfield_results], [25, 16.6666666666, 12.5, 5, 2.5, 0.5])
        assert_allclose([item.g_gaba / psiemens for item in meanfield_results], [20, 13.3333333333, 10, 4, 2, 0.4])
        assert_allclose([item.g_x for item in meanfield_results], [0.5, 0.33333333333, 0.25, 0.1, 0.05, 0.01])

    def test_refactoring_meanfield_values(self):
        meanfield_experiments: list[Experiment] = []
        wang_experiment = Experiment(meanfield_config)
        for scaling in [1, 2, 3, 4, 10, 20, 100]:
            meanfield_experiments.append(prepare_mean_field(wang_experiment, N=scaling * 2000, N_reference=2000))

        assert_equal([item.network_params.up_state.N for item in meanfield_experiments],
                     [2000, 4000, 6000, 8000, 20_000, 40_000, 200_000])

        assert_allclose([item.synaptic_params.g_ampa / psiemens for item in meanfield_experiments], [50, 25, 16.6666666666, 12.5, 5, 2.5, 0.5])
        assert_equal([item.network_params.up_state.N_E for item in meanfield_experiments], [1600, 3200, 4800, 6400, 16_000, 32_000, 160_000])

        assert_allclose([item.synaptic_params.g_gaba / psiemens for item in meanfield_experiments], [40, 20, 13.3333333333, 10, 4, 2, 0.4])
        assert_equal([item.network_params.up_state.N_I for item in meanfield_experiments],
                     [400, 800, 1200, 1600, 4000, 8000, 40_000])
        assert_allclose([item.synaptic_params.x_nmda for item in meanfield_experiments], [1, 0.5, 0.33333333333, 0.25, 0.1, 0.05, 0.01])
        assert_equal([item.network_params.up_state.N_NMDA for item in meanfield_experiments],
                     [10, 20, 30, 40, 100, 200, 1000])

    def test_refactoring_meanfield_simulation(self):
        wang_experiment = Experiment(meanfield_config)
        for scaling in [2, 3, 4, 10, 20, 100]:
            meanfield_experiment = prepare_mean_field(wang_experiment, N=scaling * 2000, N_reference=2000)
            sim_and_plot_meanfield_with_upstate_and_steady_state(meanfield_experiment)

    '''
    with amp 1, the NMDA input is very very low. We do not see the NMDA input in the Down State
    '''
    def test_why_isnt_NMDA_inputting(self):

        for nmda_amp in [1, 10, 100, 1000]:
            only_nmda = Experiment(meanfield_config).with_properties({
                SynapticParams.KEY_G_AMPA: 0,
                SynapticParams.KEY_G_GABA: 0,
                SynapticParams.KEY_G_NMDA: nmda_amp * 0.165e-9,
            })
            sim_and_plot_meanfield_with_upstate_and_steady_state(only_nmda)

    def test_meanfield_progression_for_nmda(self):

        only_nmda = Experiment(meanfield_config).with_properties({
            SynapticParams.KEY_G_AMPA: 0,
            SynapticParams.KEY_G_GABA: 0,
            SynapticParams.KEY_G_NMDA: 0.165e-9,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda", "g_e"],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES]
        })
        for scaling in [1, 1E1, 1E2, 1E3, 1E4, 1E5]:
            meanfield_experiment = prepare_mean_field(only_nmda, N=scaling * 2000, N_reference=2000)
            sim_and_plot_meanfield_with_upstate_and_steady_state(meanfield_experiment)

    def test_try_very_steep_meanfield_progression(self):
        for scaling in 10**np.array(range(10)):
            sim_and_plot_meanfield_with_upstate_and_steady_state(
                prepare_mean_field(Experiment(meanfield_config),
                                   N=scaling.item() * 2000, N_reference=2000))

    def test_understand_why_meanfield_estimation_makes_wrong_g_nmda_prediction(self):
        sim_and_plot_meanfield_with_upstate_and_steady_state(
            prepare_mean_field(Experiment(meanfield_config),
                               N=10**10 * 2000, N_reference=2000))


if __name__ == '__main__':
    unittest.main()
