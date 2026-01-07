import unittest

from brian2 import *

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams
from iteration_7_one_compartment_step_input.grid_computations import \
    sim_and_plot_experiment_grid_with_nmda_cut_off_in_down_state, \
    sim_and_plot_experiment_grid_with_increasing_nmda_input
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    single_compartment_with_nmda_and_logged_variables, sim_and_plot
from iteration_7_one_compartment_step_input.second_scripts import \
    show_up_down_states_with_different_nmda_rates_up_vs_down_example_1

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class OneCompartmentUpDownStates(unittest.TestCase):

    def test_script_2(self):
        show_up_down_states_with_different_nmda_rates_up_vs_down_example_1()


if __name__ == '__main__':
    unittest.main()
