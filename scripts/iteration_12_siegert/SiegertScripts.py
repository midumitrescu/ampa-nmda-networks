import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import mV, ms, nsiemens
from joblib import delayed, Parallel
from loguru import logger

from iteration_12_siegert.df_utils import prepare_experiment_with_N_tot, load_df_without_metadata, \
    without_elements_after_n_max
from iteration_12_transfer_function_of_lif_neurons.siegerts_formula_in_3_d import rate_LIF_whitenoise
from utils import ExtendedDict

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.models_and_configs import wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


def compute_siegert_firing_rate(mu, sigma, tau_0_membrane, experiment: Experiment):
    return rate_LIF_whitenoise(mu / mV, tau_0_membrane / ms, sigma / mV, experiment.neuron_params.theta / mV,
                               experiment.neuron_params.V_r / mV, experiment.neuron_params.tau_rp / ms)


def mean_and_sigma(n, up_state_base: dict, base: Experiment):
    exp = prepare_experiment_with_N_tot(n, up_state_base, base)

    mu_no_v = exp.effective_time_constant_up_state.E_0()
    sigma_no_v = exp.effective_time_constant_up_state.std_voltage()
    tau_0_no_nmda = exp.neuron_params.C / (exp.effective_time_constant_up_state.mean_total_conductance())
    siegert_no_rate_without_nmda = compute_siegert_firing_rate(mu_no_v, sigma_no_v, tau_0_no_nmda, exp)

    mu_v_with_nmda = exp.effective_time_constant_up_state.E_0_with_nmda()
    sigma_v_with_nmda = exp.effective_time_constant_up_state.std_voltage_with_nmda()
    tau_0_with_nmda = exp.neuron_params.C / (exp.effective_time_constant_up_state.mean_total_conductance_with_nmda())
    siegert_firing_rate_with_nmda = compute_siegert_firing_rate(mu_v_with_nmda, sigma_v_with_nmda, tau_0_with_nmda, exp)

    return ExtendedDict({
        "N": n,
        "mu_v_no_nmda": mu_no_v / mV,
        "sigma_v_no_nmda": sigma_no_v / mV,
        "firing_rate_no_nmda": siegert_no_rate_without_nmda,
        "mu_v_with_nmda": mu_v_with_nmda / mV,
        "sigma_v_with_nmda": sigma_v_with_nmda / mV,
        "firing_rate_with_nmda": siegert_firing_rate_with_nmda
    })

class ScriptsNMDAWithWangNumbers(unittest.TestCase):

    def test_call_one_mean_sigma_with_no_nmda(self):
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 0,
            "nu_nmda": 0,
        }

        result =  mean_and_sigma(n=1, up_state_base=up_state_base, base=palmer_control)

        self.assertEqual(result.mu_v_no_nmda, result.mu_v_with_nmda)
        self.assertEqual(result.sigma_v_no_nmda, result.sigma_v_with_nmda)
        self.assertEqual(result.firing_rate_no_nmda, result.firing_rate_with_nmda)

    def test_call_one_mean_sigma_with_nmda(self):
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 10,
            "nu_nmda": 10,
        }

        result =  mean_and_sigma(n=1, up_state_base=up_state_base, base=palmer_control)

        self.assertEqual(result.mu_v_no_nmda, result.mu_v_with_nmda)
        self.assertEqual(result.sigma_v_no_nmda, result.sigma_v_with_nmda)
        self.assertEqual(result.firing_rate_no_nmda, result.firing_rate_with_nmda)



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


    def test_plot_membrane_mean_and_std(self):

        for max_n in [100, 2000, 4000, 6000, 10_000]:
            df = self.compute_theoretical_mean_sigma_and_rate(max_n, base=palmer_control)
            self.plot_for_N(df=df, base=palmer_control, plot_simulation=False)
            self.plot_for_N(df=df, base=palmer_control, plot_simulation=True)


    def plot_for_N(self, df, base, plot_simulation=True):

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=(14, 12)
        )
        # Mean membrane voltage
        axes[0, 0].plot(df.N, df.mu_v_no_nmda, label="No NMDA")
        axes[0, 0].plot(df.N, df.mu_v_with_nmda, label="With NMDA")


        axes[0, 0].set_title("$V_m$ mean")
        axes[0, 0].set_ylabel("mV")
        axes[0, 0].axhline(
            y=base.neuron_params.theta / mV,
            color="red",
            linestyle="-.",
            linewidth=1,
            label="Threshold"
        )
        # STD membrane voltage
        axes[0, 1].plot(df.N, df.sigma_v_no_nmda**2, label="Variance, no NMDA")
        axes[0, 1].plot(df.N, df.sigma_v_with_nmda**2, label="Variance, with NMDA")


        axes[0, 1].set_title("$V_m$ Variance")
        axes[0, 1].set_ylabel("(mV)")


        # Predicted firing rate
        axes[1, 0].plot(df.N, df.firing_rate_no_nmda, label="rate, no NDMA")
        axes[1, 0].plot(df.N, df.firing_rate_with_nmda, label="rate, with NDMA")


        axes[1, 0].set_title("Predicted firing rate")
        axes[1, 0].set_ylabel("Hz")
        axes[1, 0].set_xlabel("N")

        if plot_simulation:
            file_control_simulation = "simulations/Control_N_10000_T_60000.csv"
            file_nmda_block_simulation = "simulations/NMDA_block_N_10000_T_60000.csv"

            df_control_simulation = load_df_without_metadata(file_control_simulation)
            df_control_simulation = without_elements_after_n_max(df_control_simulation, df.N.max())
            df_nmda_block_simulation = load_df_without_metadata(file_nmda_block_simulation)
            df_nmda_block_simulation = without_elements_after_n_max(df_nmda_block_simulation, df.N.max())

            axes[0, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_mean, label="Simulation, No NMDA",
                         linestyle="--")
            axes[0, 0].plot(df_control_simulation.N, df_control_simulation.v_mean, label="Simulation, With NMDA",
                         linestyle="--")

            axes[0, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_var,
                         label="Simulation, Variance, no NMDA", linestyle="--")
            axes[0, 1].plot(df_control_simulation.N, df_control_simulation.v_var, label="Simulation, Variance, with NMDA",
                         linestyle="--")

            axes[1, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.mean_rate,
                         label="Simulation, rate, no NDMA", linestyle="--")
            axes[1, 0].plot(df_control_simulation.N, df_control_simulation.mean_rate, label="Simulation, rate, with NDMA",
                         linestyle="--")


        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()

        if plot_simulation:
            fig.suptitle("Firing rate predicted by Siegert's formula vs brian2 simulation")
        else:
            fig.suptitle("Firing rate predicted by Siegert's formula")
        plt.tight_layout()
        plt.show()

    def compute_theoretical_mean_sigma_and_rate(self, max_n, base):
        up_state_base = {
            "N": 2000,
            "nu": 82,
            "N_nmda": 10,
            "nu_nmda": 10,
        }
        N = np.arange(1, max_n)
        results = Parallel(
            n_jobs=-1,  # use all cores
            backend="loky")(delayed(lambda n: mean_and_sigma(n, up_state_base, base))(n) for n in N)
        return pd.DataFrame.from_records(results)

    def test_computing_E_0_with_nmda(self):
        base = Experiment(wang_recurrent_config)

        self.assertEqual(0.2, base.effective_time_constant_up_state.mean_x_nmda())
        self.assertEqual(0.9, base.effective_time_constant_up_state.mean_s_nmda())
        self.assertAlmostEqual(0.02198201216, base.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens)
        self.assertAlmostEqual(-48.75363871, base.effective_time_constant_up_state.E_0_with_nmda() / mV)

    def test_compute_E_0_using_steady_state(self):
        base = Experiment(wang_recurrent_config)
        steady_up_state_results = sim_steady_state(base, state=base.network_params.up_state)

        self.assertAlmostEqual(-48.75341763069946, steady_up_state_results.v_steady)
        self.assertAlmostEqual(15.999999999999934, steady_up_state_results.g_e_steady)
        self.assertAlmostEqual(7.999999999999916, steady_up_state_results.g_i_steady)
        self.assertAlmostEqual(0.02220431199227894, steady_up_state_results.g_nmda_steady)
        self.assertAlmostEqual(0.19999999999999946, steady_up_state_results.x_nmda_steady)
        self.assertAlmostEqual(0.9090909090908988, steady_up_state_results.s_nmda_steady)

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


if __name__ == '__main__':
    unittest.main()
