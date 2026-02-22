import sys

from loguru import logger

from Plotting import show_plots_non_blocking
from iteration_12_siegert.df_utils import load_df_without_metadata, \
    without_elements_after_n_max

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")

from brian2 import plt, mpl

plt.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
from brian2 import plt


def plot_nmda_theory_vs_simulation(base, df_theory=False, df_simulation=False, N_max=1000):

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 20
    })

    fig, axes = plt.subplots(
        nrows=4,
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

    alpha = 0.5
    if df_theory:
        df_theory = without_elements_after_n_max(df_theory, N_max)
        # x
        axes[0, 0].plot(df_theory.N, df_theory.x_mean, label="Theory", alpha=alpha)
        axes[0, 1].plot(df_theory.N, df_theory.x_std ** 2, label="Theory", alpha=alpha)
        # s
        axes[1, 0].plot(df_theory.N, df_theory.s_mean, label="Theory", alpha=alpha)
        axes[1, 1].plot(df_theory.N, df_theory.s_std ** 2, label="Theory", alpha=alpha)

        # s
        axes[1, 0].plot(df_theory.N, df_theory.s_nmda_crazy_mean, label="Crazy", alpha=alpha)
        axes[1, 1].plot(df_theory.N, df_theory.s_nmda_crazy_var, label="Crazy", alpha=alpha)
        # g
        axes[2, 0].plot(df_theory.N, df_theory.g_nmda_mean, label="Theory", alpha=alpha)
        axes[2, 1].plot(df_theory.N, df_theory.g_nmda_std ** 2, label="Theory", alpha=alpha)

        # v
        axes[3, 0].plot(df_theory.N, df_theory.mu_v_with_nmda, label="Theory", alpha=alpha)
        axes[3, 1].plot(df_theory.N, df_theory.sigma_v_with_nmda**2, label="Theory", alpha=alpha)


    if df_simulation:
        df_simulation = without_elements_after_n_max(df_simulation, N_max)

        axes[0, 0].plot(df_simulation.N, df_simulation.x_nmda_mean, label="Simulation",
                        linestyle="--", alpha=alpha)
        axes[0, 1].plot(df_simulation.N, df_simulation.x_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)

        axes[0, 0].plot(df_simulation.N, df_simulation.x_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[1, 0].plot(df_simulation.N, df_simulation.s_nmda_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[1, 1].plot(df_simulation.N, df_simulation.s_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[1, 0].plot(df_simulation.N, df_simulation.s_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[2, 0].plot(df_simulation.N, df_simulation.g_nmda_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[2, 1].plot(df_simulation.N, df_simulation.g_nmda_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[2, 0].plot(df_simulation.N, df_simulation.g_nmda_steady, label="Steady",
                        linestyle="--", alpha=alpha)

        axes[3, 0].plot(df_simulation.N, df_simulation.v_mean,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[3, 1].plot(df_simulation.N, df_simulation.v_var,
                        label="Simulation", linestyle="--", alpha=alpha)
        axes[3, 0].plot(df_simulation.N, df_simulation.v_steady, label="Steady",
                        linestyle="--", alpha=alpha)

    [ax.legend() for ax in axes.flatten()]
    axes[0, 1].legend(loc="upper left")

    if plot_simulation and plot_theory:
        fig.suptitle("Plot theory vs simulation")
    elif plot_theory:
        fig.suptitle("Plot theory")
    else:
        fig.suptitle("Plot simulation")
    plt.tight_layout()
    show_plots_non_blocking()