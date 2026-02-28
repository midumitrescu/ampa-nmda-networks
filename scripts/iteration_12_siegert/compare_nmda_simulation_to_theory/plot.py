import sys

import pandas as pd
import numpy as np
from loguru import logger

from Plotting import show_plots_non_blocking
from iteration_12_siegert.df_utils import load_df_without_metadata, \
    without_elements_after_n_max, without_elements_after_max_val

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from brian2 import plt, mpl

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from brian2 import plt

def plot_nmda_theory_vs_simulation(experiment, df_theory: pd.DataFrame, df_simulation: pd.DataFrame, N_max=1000):
    return plot_nmda_theory_vs_simulation_w_column(experiment=experiment, df_theory=df_theory, df_simulation=df_simulation, column_name="n_nmda", max_val=N_max)

def plot_nu_scan_theory_vs_simulation(experiment, df_theory: pd.DataFrame, df_simulation: pd.DataFrame, nu_max=1000):
    return plot_nmda_theory_vs_simulation_w_column(experiment=experiment, df_theory=df_theory, df_simulation=df_simulation, column_name="nu_nmda", max_val=nu_max)


def plot_nmda_theory_vs_simulation_w_column(experiment, df_theory: pd.DataFrame, df_simulation: pd.DataFrame, max_val=1000, column_name="N"):

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 20
    })

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        sharex=True,
        figsize=(14, 12)
    )

    axes[0, 0].set_title("$x_\mathrm{NMDA}$ mean")
    axes[0, 0].set_ylabel("x [unitless]")
    axes[0, 1].set_title("$x_\mathrm{NMDA}$ variance")
    axes[0, 1].set_ylabel("unitless")
    axes[1, 0].set_title("$s_\mathrm{NMDA}$ mean")
    axes[1, 0].set_ylabel("s [unitless]")
    axes[1, 1].set_title("$s_\mathrm{NMDA}$ variance")
    axes[1, 1].set_ylabel("unitless")

    axes[2, 0].set_title("$g_\mathrm{NMDA}$ mean")
    axes[2, 0].set_ylabel("[ns]")
    axes[2, 1].set_title("$g_\mathrm{NMDA}$ variance")
    axes[2, 1].set_ylabel("[nS$^2$]")

    axes[3, 0].set_title("$V_\mathrm{NMDA}$ mean")
    axes[3, 0].set_ylabel("[mV]")
    axes[3, 1].set_title("$V_\mathrm{NMDA}$ variance")
    axes[3, 1].set_ylabel("[mV$^2$]")

    axes[4, 0].set_title("Correlation coeeficient (x,s)")
    axes[4, 0].set_ylabel("[unitless]")

    alpha = 0.5
    if df_theory is not None:
        df_theory = without_elements_after_max_val(df_theory, max_val=max_val, column_name=column_name)
        # x
        axes[0, 0].plot(df_theory[column_name], df_theory.x_mean, label="Theory", alpha=alpha)
        axes[0, 1].plot(df_theory[column_name], df_theory.x_std ** 2, label="Theory", alpha=alpha)
        # s
        axes[1, 0].plot(df_theory[column_name], df_theory.s_mean, label="Theory", alpha=alpha)
        #axes[1, 1].plot(df_theory[column_name], df_theory.s_std ** 2, label="Theory", alpha=alpha)

        # s
        axes[1, 0].plot(df_theory[column_name], df_theory.s_nmda_crazy_mean, label="Crazy", alpha=alpha)
        #axes[1, 1].plot(df_theory[column_name], df_theory.s_nmda_crazy_var, label="Crazy", alpha=alpha)
        # g
        axes[2, 0].plot(df_theory[column_name], df_theory.g_nmda_mean, label="Theory", alpha=alpha)
        #axes[2, 1].plot(df_theory[column_name], df_theory.g_nmda_std ** 2, label="Theory", alpha=alpha)

        # v
        axes[3, 0].plot(df_theory[column_name], df_theory.mu_v_with_nmda, label="Theory", alpha=alpha)
        #axes[3, 1].plot(df_theory[column_name], df_theory.sigma_v_with_nmda**2, label="Theory", alpha=alpha)


    if df_simulation is not None:
        df_simulation = without_elements_after_max_val(df_simulation, max_val=max_val, column_name=column_name)

        axes[0, 0].plot(df_simulation[column_name], df_simulation.x_nmda_mean, label="Simulation",
                        linestyle="--", alpha=alpha)
        axes[0, 1].plot(df_simulation[column_name], df_simulation.x_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)

        axes[0, 0].plot(df_simulation[column_name], df_simulation.x_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[1, 0].plot(df_simulation[column_name], df_simulation.s_nmda_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[1, 1].plot(df_simulation[column_name], df_simulation.s_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[1, 0].plot(df_simulation[column_name], df_simulation.s_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[2, 0].plot(df_simulation[column_name], df_simulation.g_nmda_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[2, 1].plot(df_simulation[column_name], df_simulation.g_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[2, 0].plot(df_simulation[column_name], df_simulation.g_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[3, 0].plot(df_simulation[column_name], df_simulation.v_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[3, 1].plot(df_simulation[column_name], df_simulation.v_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[3, 0].plot(df_simulation[column_name], df_simulation.v_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[4, 0].plot(df_simulation[column_name], df_simulation.corr_coef_x_s, label="simulation",
                     alpha=alpha)

        axes[4, 1].plot(df_simulation[column_name], df_simulation.s_nmda_var / df_simulation.s_nmda_mean , label="FF s",
                        alpha=alpha)
        #axes[4, 1].plot(df_simulation[column_name], np.sqrt(df_simulation.s_nmda_var) / df_simulation.s_nmda_mean, label="CV s",
        #                alpha=alpha)
        axes[4, 1].plot(df_simulation[column_name], df_simulation.x_nmda_var / df_simulation.x_nmda_mean, label="FF x",
                        alpha=alpha)
        #axes[4, 1].plot(df_simulation[column_name], np.sqrt(df_simulation.x_nmda_var) / df_simulation.x_nmda_mean, label="CV x",
        #                alpha=alpha)

    [ax.legend() for ax in axes.flatten()]
    axes[0, 1].legend(loc="upper left")

    if df_theory is not None and df_simulation is not None:
        fig.suptitle("Plot theory vs simulation")
    elif df_simulation is not None:
        fig.suptitle("Plot theory")
    else:
        fig.suptitle("Plot simulation")
    plt.tight_layout()
    show_plots_non_blocking()