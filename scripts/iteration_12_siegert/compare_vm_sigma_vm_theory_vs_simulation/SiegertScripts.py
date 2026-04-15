import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import mV, nsiemens
from joblib import delayed, Parallel
from loguru import logger

from iteration_12_siegert.compare_vm_sigma_vm_theory_vs_simulation.computations import mean_and_sigma
from iteration_12_siegert.compare_vm_sigma_vm_theory_vs_simulation.plot import plot_theory_vs_simulation, \
    plot_theory_vs_simulation_without_simulation_in_lower_graphs
from iteration_12_siegert.df_utils import load_df_without_metadata, \
    without_elements_after_n_max, with_elements_between
from iteration_12_transfer_function_of_lif_neurons.siegerts_formula_in_3_d import rate_LIF_whitenoise


logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


def compute_theoretical_mean_sigma_and_rate(max_n, base):
    up_state_base = {
        "N": 2000,
        "nu": 82,
        "N_nmda": 10,
        "nu_nmda": 10,
    }
    N = np.arange(1, max_n)
    results = Parallel(n_jobs=-1)(delayed(mean_and_sigma)(n, up_state_base, base) for n in N)
    return pd.DataFrame.from_records(results)

class ScriptsNMDAWithWangNumbers(unittest.TestCase):

    def test_compute_mean_for_one_N(self):
        base = palmer_control
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 10,
            "nu_nmda": 10,
        },
        N= np.arange(1, 2000)
        results = Parallel(
            n_jobs=-1,  # use all cores
            backend="loky")(delayed(lambda n: mean_and_sigma(n, up_state_base, palmer_control))(n) for n in N)

        df = pd.DataFrame(results)
        means, sigmas, rate, means_with_nmda = map(np.array, zip(*results))
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=(8, 10)
        )
        # Mean membrane voltage
        axes[0, 0].plot(df.N, df.means, label="Without NMDA")
        axes[0, 0].plot(N, means_with_nmda, label="NMDA")
        axes[0, 0].set_title("Mean Membrane Voltage")
        axes[0, 0].set_ylabel("mV")
        axes[0, 0].axhline(
            y=base.neuron_params.theta / mV,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Threshold"
        )
        # STD membrane voltage
        axes[0, 1].plot(N, sigmas, label="STD")
        axes[0, 1].plot(N, sigmas ** 2, label="Variance")
        axes[0, 1].set_title("STD/Variance Membrane Voltage")
        axes[0, 1].set_ylabel("(mV)")
        # Predicted firing rate
        axes[1, 0].plot(N, rate)
        axes[1, 0].set_title("Predicted firing rate")
        axes[1, 0].set_ylabel("Hz")
        axes[1, 0].set_xlabel("N")

        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()
        plt.tight_layout()
        plt.show()

    def test_plot_only_LIF_rate_theoretical_computation(self):
        df = self.compute_theoretical_mean_sigma_and_rate(2500, base=palmer_control)
        df = with_elements_between(df, 1700, 2200)
        self.plot_for_N(df=df, base=palmer_control, plot_simulation=False)


    def test_plot_membrane_mean_and_std_with_firing(self):
        experiment = palmer_control.with_property("panel", "Firing rate predicted by Siegert's formula vs brian2 simulation")
        for max_n in [500, 2000, 2500, 3000, 4000]:
            df = self.compute_theoretical_mean_sigma_and_rate(max_n, base=experiment)
            file_control_simulation = "../simulations_2/Control_N_10000_T_60000.csv"
            file_nmda_block_simulation = "../simulations_2/NMDA_block_N_10000_T_60000.csv"
            df_control_simulation = load_df_without_metadata(file_control_simulation)
            df_control_simulation = without_elements_after_n_max(df_control_simulation, max_n=max_n)
            df_nmda_block_simulation = load_df_without_metadata(file_nmda_block_simulation)
            df_nmda_block_simulation = without_elements_after_n_max(df_nmda_block_simulation, max_n=max_n)

            plot_theory_vs_simulation(base=experiment, df_theory = df, df_control_simulation=df_control_simulation, df_nmda_block_simulation=df_nmda_block_simulation)

        df = with_elements_between(df, 1700, 2200)
        df_control_simulation = with_elements_between(df, 1700, 2200)
        df_nmda_block_simulation = with_elements_between(df, 1700, 2200)


    def test_plot_membrane_mean_and_std_no_firing(self):
        experiment = palmer_control.with_property("panel", '''
        Check $\sigma_v$ simulation vs theory
        with consequence of Siegert's formula
        ''')
        file_control_simulation = "../simulations_2/Control_no_firing_N_10000_T_60000.csv"
        file_nmda_block_simulation = "../simulations_2/NMDA_block_no_firing_N_10000_T_60000.csv"
        df_control_simulation = load_df_without_metadata(file_control_simulation)
        df_nmda_block_simulation = load_df_without_metadata(file_nmda_block_simulation)
        for max_n in [500, 2000, 2500, 3000, 4000, 6000, 10_000]:
            df = self.compute_theoretical_mean_sigma_and_rate(max_n, base=experiment)
            df_control_snippet = without_elements_after_n_max(df_control_simulation, max_n=max_n)
            df_nmda_block_snippet = without_elements_after_n_max(df_nmda_block_simulation, max_n=max_n)

            plot_theory_vs_simulation_without_simulation_in_lower_graphs(base=experiment, df_theory = df, df_control_simulation=df_control_snippet, df_nmda_block_simulation=df_nmda_block_snippet)

        start_ROI, end_ROI = (1900, 2100)
        df = with_elements_between(df, start_ROI, end_ROI)
        df_control_simulation = with_elements_between(df_control_simulation, start_ROI, end_ROI)
        df_nmda_block_simulation = with_elements_between(df_nmda_block_simulation, start_ROI, end_ROI)
        plot_theory_vs_simulation_without_simulation_in_lower_graphs(base=experiment, df_theory=df, df_control_simulation=df_control_simulation,
                                  df_nmda_block_simulation=df_nmda_block_simulation)

    def test_plot_NMDA_variables_comparrison(self):
        for max_n in [500, 2000, 2500, 3000, 4000]:
            df = self.compute_theoretical_mean_sigma_and_rate(max_n, base=palmer_control)
            self.plot_for_N(df=df, base=palmer_control, plot_simulation=True)


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

    def test_plot_2_d(self):
        L = 1001  # #datapoints
        mu = np.linspace(0, 30, L)
        sigmaV = [0.01, 0.5, 1., 2., 4., 6.]
        rate = np.zeros((len(sigmaV), L))

        taum = 0.02  # seconds
        Vth = 15.0  # mV
        Vreset = 0.  # mV
        tref = 0.002  # absolute refractory period in s

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

        plt.figure(1)
        plt.clf()
        plt.plot(mu, rate_determ, ls='--', color='k', label=r'$\sigma_V=0$mV')
        for i in range(len(sigmaV)):
            plt.plot(mu, rate[i], label=r'$\sigma_V=%g$mV' % (sigmaV[i],))
        plt.xlabel(r'input $\mu$ [mV]')
        plt.ylabel('firing rate [Hz]')
        plt.legend(loc=0)
        # plt.title(r'transfer function $F(\mu,\sigma_V)$')
        #plt.savefig('lif_transferfunc.svg')
        #plt.savefig('lif_transferfunc.png', dpi=200)
        plt.show()


if __name__ == '__main__':
    unittest.main()
