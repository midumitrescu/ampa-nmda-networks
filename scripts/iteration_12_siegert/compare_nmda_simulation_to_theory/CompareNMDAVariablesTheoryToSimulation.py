import sys

from loguru import logger

from iteration_12_siegert.EffectiveMembraneConstantsSimulationVsTheory import compute_theoretical_mean_sigma_and_rate
from iteration_12_siegert.compare_nmda_simulation_to_theory.plot import plot_nmda_theory_vs_simulation
from iteration_12_siegert.compare_nmda_simulation_to_theory.simulation import sigle_compartment_with_nmda_only, \
    run_nmda_input_simulation_and_compute_statistics, scan_N_for_nmda_variables, scan_for_nu_nmda_variables
from iteration_12_siegert.compare_nmda_simulation_to_theory.theory import compute_theoretical_nmda_mean_sigma_and_rate, \
    nmda_variables
from iteration_12_siegert.df_utils import load_df_without_metadata
from iteration_12_transfer_function_of_lif_neurons.siegerts_formula_in_3_d import rate_LIF_whitenoise

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config

import unittest
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

from brian2 import plt, mpl

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from brian2 import mV, second, nsiemens

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment


def compute_siegert_firing_rate(mu, sigma, tau_0_membrane, experiment: Experiment):
    return rate_LIF_whitenoise(mu / mV, tau_0_membrane / second, sigma / mV, experiment.neuron_params.theta / mV,
                               experiment.neuron_params.V_r / mV, experiment.neuron_params.tau_rp / second)


class ScriptsNMDAWithWangNumbers(unittest.TestCase):

    def test_run_simulation_for_one_N_nmda(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
            "pannel": "NMDA_input_with_firing",
            "t_range": [0, 1000],
        })

        up_state_base = experiment.network_params.up_state.params
        object_under_test = run_nmda_input_simulation_and_compute_statistics(n=10, up_state_base=up_state_base,
                                                                             base=experiment,
                                                                             skip_start_simulation=1000)

        print(object_under_test)

        self.assertEqual(0.034946352952140124, object_under_test['g_nmda_mean'])
        self.assertEqual(0.034946352952140124, object_under_test['g_nmda_mean'])

    def test_compute_theoretical(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
            "pannel": "NMDA_input_with_firing",
            "t_range": [0, 1000],
        })

        up_state_base = {
            "N": 0,
            "nu": 0,
            "N_nmda": 10,
            "nu_nmda": 10,
        }
        object_under_test =  nmda_variables(n=10, up_state_base=up_state_base, base=experiment)
        print(object_under_test)


    def test_scan_N_nmda(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "pannel": "NMDA_input_with_firing",
            "up_state": {
                "N": 0,
                "nu": 0,
                "N_nmda": 10,
                "nu_nmda": 1,
            }, })
        scan_N_for_nmda_variables(experiment, N_max=1000, test=True)

    def test_scan_nu_nmda(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "pannel": "scan_nu_nmda",
            "up_state": {
                "N": 0,
                "nu": 0,
                "N_nmda": 1,
                "nu_nmda": 0,
            }, })
        scan_for_nu_nmda_variables(experiment, test=True)

    def test_plot_simulation_vs_theory_for_nu_nmda_10(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "pannel": "scan_nu_nmda",
            "up_state": {
                "N": 0,
                "nu": 0,
                "N_nmda": 1,
                "nu_nmda": 10,
            }})
        generated_filename = scan_N_for_nmda_variables(experiment, N_max=1000, test=True)
        df_simulation = load_df_without_metadata(generated_filename)
        df_theory = compute_theoretical_nmda_mean_sigma_and_rate(1000, base=palmer_control)

        plot_nmda_theory_vs_simulation(df_theory, df_simulation, N_max = 1000)

    def test_plot_simulation_vs_theory_for_nu_nmda_1(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "pannel": "scan_nu_nmda",
            "up_state": {
                "N": 0,
                "nu": 0,
                "N_nmda": 1,
                "nu_nmda": 1,
            }})
        generated_filename = scan_N_for_nmda_variables(experiment, N_max=1000, test=True)
        df_simulation = load_df_without_metadata(generated_filename)
        df_theory = compute_theoretical_nmda_mean_sigma_and_rate(1000, base=palmer_control)

        plot_nmda_theory_vs_simulation(df_theory, df_simulation, N_max=1000)

    def test_plot_nmda_variables(self):
        # attention here to plot same experimental conditions!!
        file_control_simulation = "simulations_2/Control_N_1000_T_1000.csv"
        df_simulation = load_df_without_metadata(file_control_simulation)

        for max_n in [500]:
            df = compute_theoretical_nmda_mean_sigma_and_rate(max_n, base=palmer_control)
            # self.plot_for_N(df=df, base=palmer_control, plot_simulation=False)
            plot_nmda_theory_vs_simulation(df=df, base=palmer_control, plot_theory=True, plot_simulation=True)

    ''' Shows that there are errors/differences between computed and simulated values
    E_0  -0.00022108196975523242
    g_e  6.572520305780927e-14
    g_i  8.260059303211165e-14
    g_nmda  -0.00022229982853693223
    x_nmda  5.551115123125783e-16
    s_nmda  -0.009090909090898824
    '''

    def test_print_difference_between_theoretical_and_newton_steady_state(self):
        base = Experiment(wang_recurrent_config)
        steady_up_state_results = sim_steady_state(base, state=base.network_params.up_state)

        print("E_0 ", (base.effective_time_constant_up_state.E_0_with_nmda() / mV - steady_up_state_results.v_steady))
        print("g_e ", (
                base.effective_time_constant_up_state.mean_excitatory_conductance() / nsiemens - steady_up_state_results.g_e_steady))
        print("g_i ", (
                base.effective_time_constant_up_state.mean_inhibitory_conductance() / nsiemens - steady_up_state_results.g_i_steady))
        print("g_nmda ", (
                base.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens - steady_up_state_results.g_nmda_steady))
        print("x_nmda ", (base.effective_time_constant_up_state.mean_x_nmda() - steady_up_state_results.x_nmda_steady))
        print("s_nmda ", (base.effective_time_constant_up_state.mean_s_nmda() - steady_up_state_results.s_nmda_steady))

    def test_generated_file_can_be_read(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "pannel": "NMDA_input_with_firing"})
        scan_N_for_nmda_variables(experiment, N_max=10, test=True)
        df_simulation = load_df_without_metadata("simulations_2/Control_N_10_T_1000.csv")

        print(df_simulation.N)

    def test_generated_file_can_be_read(self):
        df_simulation = load_df_without_metadata("simulations_2/Control_N_1000_T_1000.csv")
        print(df_simulation.N)


if __name__ == '__main__':
    unittest.main()
