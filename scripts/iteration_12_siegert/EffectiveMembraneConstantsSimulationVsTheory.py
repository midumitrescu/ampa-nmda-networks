import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import mV, nsiemens, second
from joblib import delayed, Parallel
from loguru import logger

from Plotting import show_plots_non_blocking
from iteration_12_siegert.df_utils import prepare_experiment_with_N_tot, load_df_without_metadata, \
    without_elements_after_n_max
from iteration_12_transfer_function_of_lif_neurons.siegerts_formula_in_3_d import rate_LIF_whitenoise
from utils import ExtendedDict

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control


def compute_siegert_firing_rate(mu, sigma, tau_0_membrane, experiment: Experiment):
    return rate_LIF_whitenoise(mu / mV, tau_0_membrane / second, sigma / mV, experiment.neuron_params.theta / mV,
                               experiment.neuron_params.V_r / mV, experiment.neuron_params.tau_rp / second)

linestyle_dict = {
    "loosely dotted":        (0, (1, 10)),
    "dotted":                (0, (1, 5)),
    "densely dotted":        (0, (1, 1)),

    "long dash with offset": (5, (10, 3)),
    "loosely dashed":        (0, (5, 10)),
    "dashed":                (0, (5, 5)),
    "densely dashed":        (0, (5, 1)),

    "loosely dashdotted":    (0, (3, 10, 1, 10)),
    "dashdotted":            (0, (3, 5, 1, 5)),
    "densely dashdotted":    (0, (3, 1, 1, 1)),

    "dashdotdotted":         (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def effective_time_constant_estimation(n, up_state_base: dict, base: Experiment):
    exp = prepare_experiment_with_N_tot(n, up_state_base, base)

    return ExtendedDict({
        "N": n,
        "g_e_mean": exp.effective_time_constant_up_state.mean_excitatory_conductance() / nsiemens,
        "g_i_mean": exp.effective_time_constant_up_state.mean_inhibitory_conductance() / nsiemens,
        "g_nmda_mean": exp.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens,
        "g_e_var": (exp.effective_time_constant_up_state.std_excitatory_conductance() / nsiemens)**2,
        "g_i_var": (exp.effective_time_constant_up_state.std_inhibitory_conductance() / nsiemens)**2,
        "g_nmda_var": (exp.effective_time_constant_up_state.std_g_nmda() / nsiemens) ** 2,
    })

def compute_theoretical_mean_sigma_and_rate(max_n, base):
    up_state_base = {
        "N": 2000,
        "nu": 82,
        "N_nmda": 10,
        "nu_nmda": 10,
    }
    N = np.arange(1, max_n)
    results = Parallel(
        n_jobs=-1,  # use all cores
        backend="loky")(delayed(lambda n: effective_time_constant_estimation(n, up_state_base, base))(n) for n in N)
    return pd.DataFrame.from_records(results)


def plot_v_m_and_g_s_plot(df_theory: pd.DataFrame, df_s_sim: list[tuple[pd.DataFrame, str]]):
    to_plot = ["g_e", "g_i", "g_nmda"]
    fig, axes = plt.subplots(
        nrows=len(to_plot),
        ncols=2,
        sharex=True,
        figsize=(18, 12)
    )
    alpha = 0.6

    for var_index, var_name in enumerate(to_plot):
        ax_mean = axes[var_index, 0]
        ax_var = axes[var_index, 1]

        ax_mean.set_title("$" + var_name + "$ mean")
        ax_var.set_title("$" + var_name + "$ var")

        ax_mean.plot(df_theory.N, df_theory[f"{var_name}_mean"], label="Theoretical", alpha=alpha)
        ax_var.plot(df_theory.N, df_theory[f"{var_name}_var"], label="Theoretical", alpha=alpha)

        line_styles = ["loosely dotted", "dotted", "densely dotted", "loosely dashed", "dashed", "densely dashed"]

        for index, (df_simulation, sim_label) in enumerate(df_s_sim):
            line_style_steady_state, line_style_mean, line_style_std = line_styles[index*3: index*3 + 3]
            ax_mean.plot(df_simulation.N, df_simulation[f"{var_name}_mean"], label=f"Steady state, {sim_label}",
                             linestyle=linestyle_dict[line_style_steady_state], alpha=alpha)
            ax_mean.plot(df_simulation.N, df_simulation[f"{var_name}_mean"], label=f"Simulation, {sim_label}",
                             linestyle=linestyle_dict[line_style_mean], alpha=alpha)
            ax_var.plot(df_simulation.N, df_simulation[f"{var_name}_var"], label=f"Simulation, {sim_label}",
                            linestyle=linestyle_dict[line_style_std], alpha=alpha)


    for ax in np.array(axes).flatten():
        ax.set_ylabel("[nS]")
        ax.legend()

    fig.suptitle("Richardson vs Simulation")
    fig.tight_layout()
    show_plots_non_blocking()

def plot_for_N(df):
    file_control_simulation = "simulations/Control_N_10000_T_60000.csv"
    file_nmda_block_simulation = "simulations/NMDA_block_N_10000_T_60000.csv"
    df_control_simulation = load_df_without_metadata(file_control_simulation)
    df_control_simulation = without_elements_after_n_max(df_control_simulation, df.N.max())
    df_nmda_block_simulation = load_df_without_metadata(file_nmda_block_simulation)
    df_nmda_block_simulation = without_elements_after_n_max(df_nmda_block_simulation, df.N.max())

    plot_v_m_and_g_s_plot(df, [(df_control_simulation, "Control"), (df_nmda_block_simulation, "NMDA block")])




class ScriptsPlotDifussionProcessVsSimulation(unittest.TestCase):

    @staticmethod
    def test_g_e_g_i_g_nmda_s_nmda():
        for max_n in [100, 2000, 4000, 6000, 10_000]:
        #for max_n in [100]:
            df = compute_theoretical_mean_sigma_and_rate(max_n, base=palmer_control)
            plot_for_N(df=df)

    def test_plot_and_print_g_s_simulated_vs_computed(self):
        pass

if __name__ == '__main__':
    unittest.main()


'''





    ax_g_e_mean.plot(df_control.N, df_nmda_block_simulation.g_e_mean, label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    ax_g_e_mean.plot(df_control.N, df_control.g_e_mean, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)

    axes[0, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation., label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    axes[0, 1].plot(df_control.N, df_control.g_e_var, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)

    axes[1, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_i_steady,
                    label="Steady state simulation, no NMDA",
                    linestyle=":", alpha=alpha)
    axes[1, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_i_mean, label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    axes[1, 0].plot(df_nmda_block_simulation.N, df_control_simulation.g_i_mean, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)


    # g_i
    axes[1, 0].plot(df_theory.N, df_theory.g_i_mean, label="Theoretical", alpha=alpha)
    axes[1, 0].set_title("$g_i$ mean")
    axes[1, 0].set_ylabel("[nS]")

    axes[1, 1].plot(df_theory.N, df_theory.g_i_std, label="Theoretical", alpha=alpha)
    axes[1, 1].set_title("$g_i$ std")
    axes[1, 1].set_ylabel("[nS]")

    # g_nmda
    axes[2, 0].plot(df_theory.N, df_theory.g_nmda_mean, label="Theoretical", alpha=alpha)
    axes[2, 0].set_title("$g_\mathrm{NMDA}$ mean")
    axes[2, 0].set_ylabel("[nS]")

    axes[2, 1].plot(df_theory.N, df_theory.g_nmda_std, label="Theoretical", alpha=alpha)
    axes[2, 1].set_title("$g_\mathrm{NMDA}$ std")
    axes[2, 1].set_ylabel("[nS]")




    axes[1, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_i_var, label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    axes[1, 1].plot(df_nmda_block_simulation.N, df_control_simulation.g_i_var, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)

    axes[2, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_nmda_steady,
                    label="Steady state simulation, no NMDA",
                    linestyle=":", alpha=alpha)
    axes[2, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_nmda_mean, label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    axes[2, 0].plot(df_nmda_block_simulation.N, df_control_simulation.g_nmda_mean, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)

    axes[2, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.g_nmda_var, label="Simulation, no NMDA",
                    linestyle="--", alpha=alpha)
    axes[2, 1].plot(df_nmda_block_simulation.N, df_control_simulation.g_nmda_var, label="Simulation, Control",
                    linestyle="-.", alpha=alpha)
'''