import sys

import matplotlib.pyplot as plt
import pandas as pd
from brian2 import mV
from loguru import logger

from Plotting import prepare_bigger_fonts, show_plots_non_blocking
from iteration_12_siegert.compare_vm_sigma_vm_theory_vs_simulation.computations import dr_over_d_mu

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment


def plot_theory_vs_simulation(base: Experiment, df_theory: pd.DataFrame,
                              df_control_simulation: pd.DataFrame,
                              df_nmda_block_simulation: pd.DataFrame):
    prepare_bigger_fonts()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        figsize=(14, 12)
    )
    # Mean membrane voltage
    axes[0, 0].plot(df_theory.N, df_theory.mu_v_no_nmda, label="No NMDA")
    axes[0, 0].plot(df_theory.N, df_theory.mu_v_with_nmda, label="With NMDA")

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
    axes[0, 1].plot(df_theory.N, df_theory.sigma_v_no_nmda ** 2, label="Variance, no NMDA")
    axes[0, 1].plot(df_theory.N, df_theory.sigma_v_with_nmda ** 2, label="Variance, with NMDA")

    axes[0, 1].set_title("$V_m$ Variance")
    axes[0, 1].set_ylabel("(mV)")

    # Predicted firing rate
    # axes[1, 0].plot(df.N, 1E3 * df.firing_rate_no_nmda, label="rate, no NDMA")
    # axes[1, 0].plot(df.N, 1E3 * df.firing_rate_with_nmda, label="rate, with NDMA")

    axes[1, 0].plot(df_theory.N, df_theory.firing_rate_no_nmda, label="rate, no NDMA")
    axes[1, 0].plot(df_theory.N, df_theory.firing_rate_with_nmda, label="rate, with NDMA")

    axes[1, 0].set_title("Predicted firing rate")
    axes[1, 0].set_ylabel("Hz")
    axes[1, 0].set_xlabel("N")

    axes[1, 1].plot(df_theory.N[1:], dr_over_d_mu(df_theory.firing_rate_with_nmda, df_theory.mu_v_no_nmda), label="with NMDA", alpha=0.55,
                    lw=2)
    axes[1, 1].plot(df_theory.N[1:], dr_over_d_mu(df_theory.firing_rate_with_nmda, df_theory.mu_v_with_nmda), label="without NMDA",
                    alpha=0.55)

    axes[1, 1].set_title(r"dr / d$\mu$")
    axes[1, 1].set_xlabel("N")
    axes[1, 1].set_ylabel("[Hz/mV]")


    if df_control_simulation is not None:
        axes[0, 0].plot(df_control_simulation.N, df_control_simulation.v_mean, label="Simulation, With NMDA",
                        linestyle="--")
        axes[0, 1].plot(df_control_simulation.N, df_control_simulation.v_var, label="Simulation, Variance, with NMDA",
                        linestyle="--")

        axes[1, 0].plot(df_control_simulation.N, df_control_simulation.mean_rate, label="Simulation, rate, with NDMA",
                        linestyle="--", alpha=0.5)
        axes[1, 1].plot(df_control_simulation.N[1:], dr_over_d_mu(df_control_simulation.mean_rate, df_control_simulation.v_mean),
                        label="with NMDA",  alpha=0.55, lw=2)

    if df_nmda_block_simulation is not None:
        axes[0, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_mean, label="Simulation, No NMDA",
                        linestyle="--")
        axes[0, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_var,
                        label="Simulation, Variance, no NMDA", linestyle="--")
        axes[1, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.mean_rate,
                        label="Simulation, rate, no NDMA", linestyle="--", alpha=0.5)

    axes[0, 0].legend()
    axes[0, 1].legend(loc="upper left")
    axes[1, 0].legend()
    axes[1, 1].legend()

    fig.suptitle(base.plot_params.panel)
    plt.tight_layout()
    show_plots_non_blocking()

def plot_theory_vs_simulation_without_simulation_in_lower_graphs(base: Experiment, df_theory: pd.DataFrame,
                              df_control_simulation: pd.DataFrame,
                              df_nmda_block_simulation: pd.DataFrame):
    prepare_bigger_fonts()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        figsize=(14, 12)
    )

    alpha = 0.6
    if df_control_simulation is not None:
        axes[0, 0].plot(df_control_simulation.N, df_control_simulation.v_mean, label="Simulation, With NMDA",
                        linestyle="--", alpha=alpha)
        axes[0, 1].plot(df_control_simulation.N, df_control_simulation.v_var, label="Simulation, Variance, with NMDA",
                        linestyle="--", alpha=alpha)

    if df_nmda_block_simulation is not None:
        axes[0, 0].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_mean, label="Simulation, No NMDA",
                        linestyle="--", alpha=alpha)
        axes[0, 1].plot(df_nmda_block_simulation.N, df_nmda_block_simulation.v_var,
                        label="Simulation, Variance, no NMDA", linestyle="--", alpha=alpha)

    # Mean membrane voltage
    axes[0, 0].plot(df_theory.N, df_theory.mu_v_no_nmda, label="No NMDA", alpha=alpha)
    axes[0, 0].plot(df_theory.N, df_theory.mu_v_with_nmda, label="With NMDA", alpha=alpha)

    axes[0, 0].set_title("$V_m$ mean")
    axes[0, 0].set_ylabel("[mV]")
    axes[0, 0].axhline(
        y=base.neuron_params.theta / mV,
        color="red",
        linestyle="-.",
        linewidth=1,
        label="Threshold"
    )
    # STD membrane voltage
    axes[0, 1].plot(df_theory.N, df_theory.sigma_v_no_nmda ** 2, label="Variance, no NMDA", alpha=alpha)
    axes[0, 1].plot(df_theory.N, df_theory.sigma_v_with_nmda ** 2, label="Variance, with NMDA", alpha=alpha)

    axes[0, 1].set_title("$V_m$ Variance")
    axes[0, 1].set_ylabel("[mV]")

    # Predicted firing rate
    # axes[1, 0].plot(df.N, 1E3 * df.firing_rate_no_nmda, label="rate, no NDMA")
    # axes[1, 0].plot(df.N, 1E3 * df.firing_rate_with_nmda, label="rate, with NDMA")

    axes[1, 0].plot(df_theory.N, df_theory.firing_rate_no_nmda, label="rate, no NDMA", alpha=alpha)
    axes[1, 0].plot(df_theory.N, df_theory.firing_rate_with_nmda, label="rate, with NDMA", alpha=alpha)

    axes[1, 0].set_title("Predicted firing rate")
    axes[1, 0].set_ylabel("[Hz]")
    axes[1, 0].set_xlabel("N")

    axes[1, 1].plot(df_theory.N[1:], dr_over_d_mu(df_theory.firing_rate_with_nmda, df_theory.mu_v_no_nmda),
                    label="with NMDA", alpha=alpha,
                    lw=2)
    axes[1, 1].plot(df_theory.N[1:], dr_over_d_mu(df_theory.firing_rate_with_nmda, df_theory.mu_v_with_nmda),
                    label="without NMDA", alpha=alpha)

    axes[1, 1].set_title(r"dr / d$\mu$")
    axes[1, 1].set_xlabel("N")
    axes[1, 1].set_ylabel("[Hz/mV]")



    axes[0, 0].legend()
    axes[0, 1].legend(loc="upper left")
    axes[1, 0].legend()
    axes[1, 1].legend()

    fig.suptitle(base.plot_params.panel)
    plt.tight_layout()
    show_plots_non_blocking()