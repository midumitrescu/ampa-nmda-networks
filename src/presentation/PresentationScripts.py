import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, Hz

from Plotting import show_plots_non_blocking, prepare_bigger_fonts
from iteration_12_siegert.compare_nmda_simulation_to_theory.CompareNMDAVariablesTheoryToSimulation import \
    NMDAVarsTheoryVsSimulation
from iteration_12_siegert.compare_nmda_simulation_to_theory.plot import plot_nu_scan_theory_vs_simulation
from iteration_12_siegert.compare_nmda_simulation_to_theory.simulation import sigle_compartment_with_nmda_only
from iteration_12_siegert.compare_nmda_simulation_to_theory.theory import \
    compute_theoretical_nmda_mean_sigma_and_rate_nu_scan
from iteration_12_siegert.df_utils import filename_for_nu_scan_experiment, load_df_without_metadata
from iteration_12_siegert.gain_computations import RateGainSearchParams, plot_rate_and_gain_combined, GainScripts
from iteration_12_siegert.look_for_all_mu_sigma_for_fixed_rate_scripts import LookForAllSolutionsScripts
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import rate_LIF_whitenoise
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    SynapticParams
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import \
    simulate_and_plot_with_up_and_down_state_and_nmda
from iteration_7_one_compartment_step_input.one_compartment_with_up_only import sim_and_plot_up_with_state_and_nmda
from iteration_8_compute_mean_steady_state.grid_computations import parallelize_simulate_with_up_state_and_nmda
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import \
    plot_and_compare_two_voltages_curves
from iteration_8_compute_mean_steady_state.test_wang_numbers import wang_recurrent_config


def plot_schematics():
    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Draw neuron (soma)
    soma_circle = plt.Circle((0, 0), 0.25, color='black', fill=False, linewidth=2, linestyle='dotted')
    ax.add_patch(soma_circle)
    ax.text(0, 0, "Soma", ha='center', va='center', fontsize=12, fontweight='bold')

    # Define input positions
    N_E, N_I = 4, 5  # Number of regular excitatory and inhibitory inputs
    angles_E = np.linspace(-0.5 * np.pi / 3, np.pi / 3, N_E)  # Top excitatory
    angles_I = np.linspace(2 * np.pi / 3, 3.5 * np.pi / 3, N_I)  # Bottom inhibitory

    # Draw excitatory inputs (AMPA)
    for index, angle in enumerate(angles_E):
        if index != 1:
            x, y = np.cos(angle) * 0.8, np.sin(angle) * 0.8
            arrow_excitatory = ax.arrow(x, y, -0.5 * x, -0.5 * y, head_width=0.05, head_length=0.05, fc='blue',
                                        ec='blue', color='blue')
            ax.text(x, y + 0.1, "+", color='blue', fontsize=14, ha='center')

    arrow_excitatory.set_label('$N_E$ Excitatory synapses')

    # Draw inhibitory inputs (GABA)
    for angle in angles_I:
        x, y = np.cos(angle) * 0.8, np.sin(angle) * 0.8
        arrow_inhibitory = ax.arrow(x, y, -0.5 * x, -0.5 * y, head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(x, y - 0.1, "−", color='red', fontsize=14, ha='center')

    arrow_inhibitory.set_label('$N_I$ Inhibitory synapses')

    N_N = 3
    angles_N = np.linspace(4 * np.pi / 3, 5 * np.pi / 3, N_N)

    for i, angle in enumerate(angles_N):
        x, y = np.cos(angle) * 0.8, np.sin(angle) * 0.8
        arrow_nmda = ax.arrow(
            x, y,
            -0.5 * x, -0.5 * y,
            head_width=0.08,
            head_length=0.08,
            fc='purple',
            ec='purple',
            linewidth=2
        )

        # optional + sign like excitatory
        ax.text(x, y + 0.1, "+", color='purple', fontsize=14, ha='center')

    # Set legend label only once
    arrow_nmda.set_label('$N_N$ NMDA')

    # x_nmda, y_nmda = 0, -1
    # ax.arrow(x_nmda, y_nmda, 0, .5, head_width=0.1, head_length=0.1, fc='purple', ec='purple', linewidth=2, label='$N_N$ NMDA')
    # ax.text(x_nmda, y_nmda - 0.1, "NMDA", color='purple', fontsize=14, ha='center', fontweight='bold')

    x_leak, y_leak = 0, 0.15
    ax.arrow(x_leak, y_leak, 0, .3, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linestyle='-.', label='Leak')
    ax.text(x_leak, y_leak + 0.45, "Leak", color='gray', fontsize=14, ha='center')

    ax.arrow(0.3, 0, 0.3, 0, head_width=0.05, head_length=0.05, fc='black', ec='black', linewidth=2)
    ax.text(0.7, 0, r"-$(\theta - V_R) \cdot \dot{N}$", fontsize=12, va='center')

    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

    # Show plot
    plt.show()


palmer_mk_801_block = (Experiment(wang_recurrent_config).with_properties({
    "up_state": {
        "N": 2000,
        "nu": 81,
        "N_nmda": 0,
        "nu_nmda": 0,
    },
    "t_range": [[0, 20_000]],
    PlotParams.KEY_PANEL: "MK 801 Block (model without NDMA)",
    PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE]
}))

palmer_control = palmer_mk_801_block.with_properties({
    "up_state": {
        "N": 2000,
        "nu": 81,
        "N_nmda": 10,
        "nu_nmda": 5,
    },
    SynapticParams.KEY_G_NMDA: 0.211e-8,
    PlotParams.KEY_PANEL: "Control (model with NDMA)",
})


class PresentationScripts(unittest.TestCase):
    def test_script_create_single_compartment_schema(self):
        plot_schematics()

    def test_show_one_example_with_g_ampa_g_gaba_low_nu(self):
        experiment = Experiment(wang_recurrent_config).with_properties(
            {
                "up_state": {
                    "N": 20,
                    "nu": 5,
                    "N_nmda": 0,
                    "nu_nmda": 0,
                },
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["g_e", "g_i"],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                "t_range": [[0, 100], [0, 500], [0, 1000]],
                PlotParams.KEY_PANEL: "Example of excitatory and inhibitory synaptic input. Low presynaptic rate"
            }
        )
        sim_and_plot_up_with_state_and_nmda(experiment)
        simulate_and_plot_with_up_and_down_state_and_nmda(experiment)

    def test_show_one_example_with_g_ampa_g_gaba_high_nu(self):
        experiment = Experiment(wang_recurrent_config).with_properties(
            {
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["g_e", "g_i"],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                "t_range": [[0, 100], [0, 500], [0, 1000]],
                PlotParams.KEY_PANEL: "Example of excitatory and inhibitory synaptic input. High presynaptic rate"
            }
        )
        sim_and_plot_up_with_state_and_nmda(experiment)

    def test_show_one_example_with_x_s(self):
        experiment = Experiment(wang_recurrent_config).with_properties(
            {
                "up_state": {
                    "N": 0,
                    "nu": 0,
                    "N_nmda": 1,
                    "nu_nmda": 2,
                },
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda"],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                "t_range": [[0, 100], [0, 500], [0, 1000]],
                PlotParams.KEY_PANEL: "Show $x_{NMDA}, s_{NMDA}$ for low NMDA rate"
            }
        )
        sim_and_plot_up_with_state_and_nmda(experiment)
        experiment = Experiment(wang_recurrent_config).with_properties(
            {
                "up_state": {
                    "N": 0,
                    "nu": 0,
                    "N_nmda": 10,
                    "nu_nmda": 10,
                },
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda"],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                "t_range": [[0, 100], [0, 500], [0, 1000]],
                PlotParams.KEY_PANEL: "Show $x_{NMDA}, s_{NMDA}$ for high NMDA rate"
            }
        )
        sim_and_plot_up_with_state_and_nmda(experiment)

        experiment = Experiment(wang_recurrent_config).with_properties(
            {
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "sigmoid_v", "g_nmda"],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                    PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                "t_range": [[0, 2000]],
                PlotParams.KEY_PANEL: "Show $x_{NMDA}, s_{NMDA}$ for high NMDA rate"
            }
        )
        simulate_and_plot_with_up_and_down_state_and_nmda(experiment)

    def test_show_palmer_control_and_nmda_block(self):
        '''
        palmer_experiment_to_plot = palmer_control.with_properties({
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,  PlotParams.AvailablePlots.HIDDEN_VARIABLES],
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
        })

        sim_and_plot_up_with_state_and_nmda(palmer_experiment_to_plot)
        sim_and_plot_up_down(palmer_experiment_to_plot)
        '''
        for nu in [0, 5, 50, 75, 100, 125, 150, 200]:
            palmer_experiment_plotted = palmer_experiment.with_properties({
                "up_state": {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 1,
                    "nu_nmda": nu
                },
                "t_range": [[0, 10_000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                    PlotParams.AvailablePlots.HIDDEN_VARIABLES],
                Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "g_nmda"],
                PlotParams.KEY_PANEL: f"Palmer control XXXXX. Nu = {nu}"
            })

            sim_and_plot_up_with_state_and_nmda(palmer_experiment_plotted)
            # sim_and_plot_up_down(palmer_experiment_plotted)

    def test_plot_already_run_simulation(self):
        experiment = palmer_control.with_properties({
            Experiment.KEY_SELECTED_MODEL: sigle_compartment_with_nmda_only,
            "panel": "scan_nu_nmda",
            "t_range": [0, 30 * 1000],
            "up_state": {
                "N": 0,
                "nu": 0,
                "N_nmda": 1,
                "nu_nmda": 1,
            }})
        nu_max = 1_000
        file_name = filename_for_nu_scan_experiment(experiment=experiment, output_dir="simulations_2", nu_max=nu_max)
        print(file_name)
        df_simulation = load_df_without_metadata(file_name)
        df_theory = compute_theoretical_nmda_mean_sigma_and_rate_nu_scan(max_nu=nu_max, base=experiment)

        plot_nu_scan_theory_vs_simulation(experiment=experiment, df_theory=df_theory, df_simulation=df_simulation,
                                          nu_max=nu_max)

    # slide to compare NMDA block to control
    def test_run_palmer_with_found_nu_exc(self):

        sim_and_plot_up_down(palmer_mk_801_block)
        sim_and_plot_up_down(palmer_control)

        t_range = [0, 100_000]
        exp_1 = palmer_control.with_property("t_range", t_range).with_property("theta", 100)
        exp_2 = palmer_mk_801_block.with_property("t_range", t_range).with_property("theta", 100)

        exp_results = parallelize_simulate_with_up_state_and_nmda([exp_1, exp_2])

        plot_and_compare_two_voltages_curves(results_1=exp_results[0], results_2=exp_results[1], rate_exp_1=0.2,
                                             rate_exp_2=0.05)

    def test_simulation_can_be_plotted(self):
        # check the test
        NMDAVarsTheoryVsSimulation().test_simulation_can_be_plotted()

    def test_plot_gain(self):
        plot_rate_and_gain_combined(
            params=RateGainSearchParams(rate_nmda_block=0.05, rate_with_nmda=0.18, mu_v_nmda_block=-47.4,
                                        mu_v_with_nmda=-47.4 + 0.4, d_sigma=0.08))

    def test_plot_LIF_curve_in_ROI(self):
        experiment = palmer_control
        prepare_bigger_fonts()
        plt.figure(figsize=(10, 6))

        L = 1001  # #datapoints
        mu = np.linspace(-65, -45, L) * mV
        sigmaV = np.array([0.5, 1., 2., 3]) * mV
        rate = np.zeros((len(sigmaV), L))

        taum = experiment.effective_time_constant_up_state.tau_eff()
        Vth = experiment.neuron_params.theta
        Vreset = experiment.neuron_params.V_r
        tref = experiment.neuron_params.tau_rp

        for i in range(len(sigmaV)):
            print(sigmaV[i])
            for j in range(L):
                rate[i, j] = rate_LIF_whitenoise(mu[j], taum, sigmaV[i], Vth, Vreset, tref)

        # firing rate for sigma=0 (no noise)
        rate_determ = np.zeros(L)
        for j in range(L):
            if mu[j] > Vth:
                T = taum * np.log((mu[j] - Vreset) / (mu[j] - Vth))
                rate_determ[j] = 1. / (T + tref)

        plt.plot(mu / mV, rate_determ / Hz, ls='--', color='k', label=r'$\sigma_V=0$ mV')
        for i in range(len(sigmaV)):
            plt.plot(mu / mV, rate[i] / Hz, label=r'$\sigma_V=%g$ mV' % (sigmaV[i] / mV,))
        plt.xlabel(r'input $\mu$ [mV]')
        plt.ylabel('firing rate [Hz]')

        plt.axhline(y=0.05, color='orange', linestyle='--', label='Mk801')
        plt.axhline(y=0.3, color='black', linestyle='-.', label='Control')
        # Set y-axis limits
        plt.ylim(0, 0.5)

        # Threshold in mV
        theta = Vth / mV

        # Very faint gray ROI: 10 mV to 5 mV below threshold
        plt.axvspan(theta - 10, theta - 5, color='gray', alpha=0.1, label='ROI')

        # Vertical line at threshold
        plt.axvline(theta, color='red', linestyle=':', linewidth=1.5, label=r'$\theta$ (threshold)')

        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.subplots_adjust(right=0.75)
        plt.title("Siegert's first passage time formula for the difussion approximation of our neuron model \n in (-10, -5) mV bellow $\\theta$")

        plt.tight_layout()
        show_plots_non_blocking(caller_test_case=self)

    def test_show_linear_fit_mu_sigma(self):
        LookForAllSolutionsScripts().test_scripts_look_for_all_solutions_using_binary_search_palmer_rates()

    def test_plot_gain_with_fitted_solution(self):
        plot_rate_and_gain_combined(
            params=RateGainSearchParams(rate_nmda_block=0.05, rate_with_nmda=0.18, mu_v_nmda_block=-47.33,
                                        mu_v_with_nmda=-47.33 + 0.4, d_sigma=0.08))

    def test_plot_gain_with_fitted_solution_2(self):
        plot_rate_and_gain_combined(
            params=RateGainSearchParams(rate_nmda_block=0.05, rate_with_nmda=0.18, mu_v_nmda_block=-44.12,
                                        mu_v_with_nmda=-44.12 + 0.4, d_sigma=0))


if __name__ == '__main__':
    unittest.main()
