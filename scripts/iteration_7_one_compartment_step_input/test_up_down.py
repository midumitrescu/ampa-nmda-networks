import unittest

from brian2 import *

from iteration_7_one_compartment_step_input.second_scripts import \
    show_up_down_states_with_different_nmda_rates_up_vs_down_example_1

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True


class OneCompartmentUpDownStates(unittest.TestCase):

    def test_script_2(self):
        show_up_down_states_with_different_nmda_rates_up_vs_down_example_1()


if __name__ == '__main__':
    unittest.main()
