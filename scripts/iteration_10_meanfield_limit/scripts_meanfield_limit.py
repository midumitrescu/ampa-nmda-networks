import sys
import unittest

import numpy as np
from loguru import logger

from iteration_10_meanfield_limit.meanfield_simulation import sim_and_plot_meanfield_with_upstate_and_steady_state, \
    prepare_mean_field
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    NeuronModelParams, SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import steady_model

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

class ScriptsMeanField(unittest.TestCase):

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
        for scaling in 10**np.array(range(15)):
            sim_and_plot_meanfield_with_upstate_and_steady_state(
                prepare_mean_field(Experiment(meanfield_config),
                                   N=scaling.item() * 2000, N_reference=2000))

    def test_understand_why_meanfield_estimation_makes_wrong_g_nmda_prediction(self):
        sim_and_plot_meanfield_with_upstate_and_steady_state(
            prepare_mean_field(Experiment(meanfield_config),
                               N=10**10 * 2000, N_reference=2000))

    def test_verify_plotting_of_conductance_in_rater_and_rate_plot(self):
        only_nmda = Experiment(meanfield_config).with_properties({
            SynapticParams.KEY_G_AMPA: 0,
            SynapticParams.KEY_G_GABA: 0,
            SynapticParams.KEY_G_NMDA: 0.165e-9,
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE],
            "t_range": [[0, 5_000]],
        })
        meanfield_experiment = prepare_mean_field(only_nmda, N=2000, N_reference=2000)
        sim_and_plot_meanfield_with_upstate_and_steady_state(meanfield_experiment)


if __name__ == '__main__':
    unittest.main()
