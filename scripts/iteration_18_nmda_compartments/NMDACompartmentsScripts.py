import math
import unittest

from brian2 import Hz, mV, second
from joblib import Parallel, delayed

from iteration_16.Simulate_K_NMDA_Compartments import plot_k_sweep_results
from iteration_16.model import config_with_weak_synapses, ConductanceDiffusionSimulationConfig
from iteration_16.nmda_compartment_model import NMDASimulationWangCompartments

def run_simulations_in_parallel_and_compare(base_config: ConductanceDiffusionSimulationConfig, k_s):
    run_one = lambda k: NMDASimulationWangCompartments.run(base_config.with_property(k_comp=k))

    results = Parallel(n_jobs=len(k_s))(
        delayed(run_one)(k) for k in k_s
    )

    result_by_k = dict(zip(k_s, results))

    plot_k_sweep_results(result_by_k, k_s, base_config)

class NMDAWithCompartmentScripts(unittest.TestCase):

    def test_run_with_one_compartment_works(self):
        config = config_with_weak_synapses.with_property(r_e=2 * Hz, r_i=2 * Hz, N_E=10, N_I=5, seed=200,
                                                         simulation_time=2 * second, e_L=-45 * mV, k_comp=1)
        NMDASimulationWangCompartments.run_and_plot(config)

    def test_compare_many_compartments_to_only_one(self):

        k_s = [1, 2, 5, 10]
        config = config_with_weak_synapses.with_property(r_e=2 * Hz, r_i=2 * Hz, N_E=10, N_I=5, seed=200, simulation_time=2 * second, e_L=-45 * mV)
        run_one = lambda k: NMDASimulationWangCompartments.run(config.with_property(k_comp = k))

        results = Parallel(n_jobs=len(k_s))(
            delayed(run_one)(k) for k in k_s
        )

        result_by_k = dict(zip(k_s, results))

        plot_k_sweep_results(result_by_k, k_s, config)

        run_simulations_in_parallel_and_compare(config, k_s)

    def test_compare_many_compartments_to_only_one_with_meanfield_scaling(self, scaling = math.sqrt):

        k_s = [1, 2, 5, 10]
        config = config_with_weak_synapses.with_property(r_e=2 * Hz, r_i=2 * Hz, N_E=10, N_I=5, seed=200, simulation_time=2 * second, e_L=-45 * mV)
        w_x_1_compartment = config.w_x
        run_one = lambda k: NMDASimulationWangCompartments.run(config.with_property(w_x = w_x_1_compartment / scaling(k)), k=k)

        results = Parallel(n_jobs=len(k_s))(
            delayed(run_one)(k) for k in k_s
        )

        result_by_k = dict(zip(k_s, results))

        plot_k_sweep_results(result_by_k, k_s, config, experiment_title="Neuronal dynamics for cluster-grouped input")

    def test_create_model_schematics(self):
        def point_on_circle_edge(start, center, radius):
            """
            Returns point on circle boundary from start -> center direction.
            """
            v = np.array(center) - np.array(start)
            v = v / np.linalg.norm(v)
            return np.array(center) - radius * v

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Circle

        def connect_ampa_to_cluster(y_ampa, cluster, ax):
            # AMPA 1 position (first red spike train anchor)
            ampa_train_end = -5 + 6
            ampa1_start = (ampa_train_end, y_ampa)

            # =========================================================
            # AMPA 1 → Cluster 1 (red arrow)
            # =========================================================

            ax.annotate(
                "",
                xy=cluster,
                xytext=ampa1_start,
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
            )

        def connect_cluster_to_soma(cluster, soma_center, soma_radius, ax):

            cluster = np.array(cluster)

            # endpoint must lie on soma boundary
            soma_edge = point_on_circle_edge(
                start=cluster,
                center=soma_center,
                radius=soma_radius
            )

            ax.annotate(
                "",
                xy=soma_edge,
                xytext=cluster,
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=1.2
                ),
            )

        def connect_train_to_soma(x0, y, duration, soma_center, soma_radius, ax, color="red"):
            start = (x0 + duration, y)

            soma_edge = point_on_circle_edge(
                start=start,
                center=soma_center,
                radius=soma_radius
            )

            ax.plot(
                [start[0], soma_edge[0]],
                [start[1], soma_edge[1]],
                linestyle=":",
                color=color,
                lw=1.2,
            )

            ax.annotate(
                "",
                xy=soma_edge,  # arrow head here
                xytext=start,  # start of dotted line
                arrowprops=dict(
                    arrowstyle="-|>",
                    linestyle=":",
                    color=color,
                    lw=1.2,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )

        def draw_schematic(n_ampa=4, n_gaba=4, K=10, seed=2):

            rng = np.random.default_rng(seed)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(-7, 9)
            ax.set_ylim(-5, 5)
            ax.axis("off")

            # =========================================================
            # SOMA (right side)
            # =========================================================
            soma_x, soma_y = 6.0, 0.0
            soma_center = (soma_x, soma_y)
            soma_radius = 0.7

            soma = Circle((soma_x, soma_y), soma_radius, fill=False, lw=2)
            ax.add_patch(soma)

            ax.text(soma_x, soma_y + 0.15, "soma", ha="center", va="center", fontsize=12)
            ax.text(soma_x, soma_y - 0.2, "v", ha="center", va="center", fontsize=12)

            # =========================================================
            # CLUSTERS (3 visible positions)
            # angles: 5π/4, π/2, π/4
            # =========================================================
            angles = [5 * np.pi / 4, np.pi - 0.1, 3 * np.pi / 4 - 0.2]
            radius = 2.6

            cluster_labels = [
                "Cluster 1",
                "Cluster 2\n...",
                "Cluster K\n"
            ]

            cluster_positions = []

            for i, ang in enumerate(angles):
                cx = soma_x + radius * np.cos(ang)
                cy = soma_y + radius * np.sin(ang)

                cluster_positions.append((cx, cy))

                # cluster dot
                ax.scatter(cx, cy, s=200, color="black")

                # label BELOW dot
                ax.text(
                    cx,
                    cy - 0.45,
                    cluster_labels[i],
                    ha="center",
                    va="top",
                    fontsize=10,
                )

            # =========================================================
            # SPIKE TRAIN FUNCTION
            # =========================================================
            def spike_train(x0, y0, color, duration=5.5, height=0.8):

                n_spikes = rng.integers(4, 9)
                spike_times = np.sort(rng.uniform(0, duration, n_spikes))

                ax.plot([x0, x0 + duration], [y0, y0], color=color, lw=1)

                for st in spike_times:
                    ax.plot(
                        [x0 + st, x0 + st],
                        [y0, y0 + height],
                        color=color,
                        lw=1.5,
                    )

            # =========================================================
            # INPUTS (left side)
            # =========================================================
            y_ampa = np.linspace(3.5, 1.2, n_ampa)
            y_gaba = np.linspace(-1.2, -3.5, n_gaba)

            for y in y_ampa:
                ax.scatter(-5, y, color="red", s=60)
                spike_train(-4.5, y, "red")

            for y in y_gaba:
                ax.scatter(-5, y, color="blue", s=60)
                spike_train(-4.5, y, "blue")

            connect_ampa_to_cluster(y_ampa=y_ampa[3], cluster=cluster_positions[0], ax=ax)
            connect_ampa_to_cluster(y_ampa=y_ampa[2], cluster=cluster_positions[1], ax=ax)
            connect_ampa_to_cluster(y_ampa=y_ampa[1], cluster=cluster_positions[1], ax=ax)
            connect_ampa_to_cluster(y_ampa=y_ampa[0], cluster=cluster_positions[2], ax=ax)

            for c in cluster_positions:
                connect_cluster_to_soma(
                    cluster=c,
                    soma_center=soma_center,
                    soma_radius=soma_radius,
                    ax=ax
                )

            for y in y_ampa:
                connect_train_to_soma(
                    x0=-5.5,
                    y=y,
                    duration=6.5,
                    soma_center=soma_center,
                    soma_radius=soma_radius,
                    ax=ax
                )

            for y in y_gaba:
                connect_train_to_soma(
                    x0=-5.5,
                    y=y,
                    duration=6.5,
                    soma_center=soma_center,
                    soma_radius=soma_radius,
                    ax=ax,
                    color="blue"
                )

            import matplotlib.lines as mlines

            ampa_handle = mlines.Line2D(
                [],
                [],
                color="red",
                marker="o",
                linestyle="None",
                label="AMPA Synapse"
            )

            gaba_handle = mlines.Line2D(
                [],
                [],
                color="blue",
                marker="o",
                linestyle="None",
                label="GABA Synapse"
            )

            cluster_handle = mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                label="Synaptic cluster"
            )

            # =========================
            # AMPA (red dotted arrow)
            # =========================
            ampa_spike_handle = mlines.Line2D(
                [0, 1], [0, 0],
                color="red",
                lw=1.5,
                linestyle=":",
                marker=r"$\rightarrow$",
                markersize=12,
                markevery=[1],
                label="AMPA spike"
            )

            # =========================
            # GABA (blue dotted arrow)
            # =========================
            gaba_spike_handle = mlines.Line2D(
                [0, 1], [0, 0],
                color="blue",
                lw=1.5,
                linestyle=":",
                marker=r"$\rightarrow$",
                markersize=12,
                markevery=[1],
                label="GABA spike"
            )

            nmda_handle = mlines.Line2D(
                [0, 1], [0, 0],
                color="black",
                lw=0,
                marker=r"$\rightarrow$",
                markersize=12,
                markevery=[1],
                label="NMDA current"
            )

            ax.legend(
                handles=[ampa_handle, gaba_handle, cluster_handle, ampa_spike_handle, gaba_spike_handle, nmda_handle],
                loc="upper right"
            )

            ax.set_title(
                "Schematics of iso-potential single-compartment model with NMDA Clusters",
                fontsize=14,
                pad=12
            )

            return fig, ax


        # Example
        draw_schematic(n_ampa=4, n_gaba=4, K=10, seed=3)
        plt.show()


if __name__ == '__main__':
    unittest.main()
