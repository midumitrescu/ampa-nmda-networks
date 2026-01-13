import unittest

import numpy as np
from brian2 import mV, mmole, second, ms, Hz, nS
import matplotlib.pyplot as plt

from BinarySeach import binary_search_for_target_value
from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, PlotParams, \
    SynapticParams
from iteration_7_one_compartment_step_input.models_and_configs import \
    single_compartment_without_nmda_deactivation_and_logged_variables
from iteration_7_one_compartment_step_input.one_compartment_with_up_down import SimulationResults
from iteration_7_one_compartment_step_input.one_compartment_with_up_only import simulate_with_up_state_and_nmda, \
    sim_and_plot_up_with_state_and_nmda
from iteration_8_compute_mean_steady_state.grid_computations import \
    sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment, \
    palmer_experiment_0_1_Hz_with_NMDA_block, wang_recurrent_config
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_and_plot_up_down, \
    plot_voltage_trace_comparisons


def sigmoid_v(experiment, v):
    MG_C = experiment.synaptic_params.MG_C
    return 1 / (1 + (MG_C / mmole) / 3.57 * np.exp(-0.062 * (v / mV)))


def find_firing_rate_without_NMDA_with_N(experiment, N, sim_time=10 * second):
    up_state = experiment.params["up_state"]
    up_state["N"] = N

    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        "up_state": up_state})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz


def find_firing_rate_without_NMDA_with_nu(experiment, nu, sim_time=10 * second):
    up_state = experiment.params["up_state"]
    up_state["nu"] = nu

    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        "up_state": up_state})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz


def run_with_NMDA_and_obtain_firing_rate(experiment, g_nmda_max, sim_time=10 * second):
    experiment_with_nmda = experiment.with_properties({
        Experiment.KEY_SIM_TIME: sim_time / ms,
        SynapticParams.KEY_G_NMDA: g_nmda_max})
    results_with_nmda = simulate_with_up_state_and_nmda(experiment_with_nmda)

    rate = results_with_nmda.total_spike_counts() / experiment_with_nmda.sim_time

    return rate / Hz

palmer_control = (Experiment(wang_recurrent_config).with_properties({
            SynapticParams.KEY_G_NMDA: 0.9e-9,
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 10,
                    "nu_nmda": 10,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS],
            Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
            "panel": "Control"
        }))
palmer_nmda_block = palmer_control.with_properties({
    SynapticParams.KEY_X_NMDA: 0,
    "panel": "NMDA block"
})

class ScriptsNMDAWithWangNumbers(unittest.TestCase):

    def test_up_down_with_wang_numbers(self):
        sim_and_plot_up_down(Experiment(wang_recurrent_config))

    '''
    Model starts firing between 80Hz input -> 0 Hz output, 90 Hz -> 20-40 Hz
    '''

    def test_only_up_with_wang_numbers(self):
        various_nu_ext = np.arange(0, 100, step=10)
        for nu_ext in various_nu_ext:
            same_state = {
                "N": 2000,
                "nu": nu_ext,
                "N_nmda": 10,
                "nu_nmda": 10,
            }
            experiment = Experiment(wang_recurrent_config).with_property("up_state", same_state).with_property(
                "down_state", same_state)
            sim_and_plot_up_down(experiment)

    def test_grid_increasing_N_E_vs_increasing_g_nmda(self):
        increasing_nmda = np.array([0, 0.165e-9, 0.3e-9, 0.5e-9])
        increasing_nmda = np.array([0.5e-9, 1e-9, 1.5e-9, 3e-9])
        # increasing_N_E = [500, 1000, 1500, 2000] -> here, 1000 no firing, 1500 firing a lot
        # increasing_N_E = [1100, 1200, 1300, 1400] -> 1100, 1200 no firing. 1300 -> 0.05 Hz. 1400 Too much => 1300 is the right number of excitatory N
        increasing_N_E = [1300]
        for n_e in increasing_N_E:
            current_experiment = (Experiment(wang_recurrent_config).with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW,
                                                                                  [PlotParams.AvailablePlots.RASTER_AND_RATE])
                                  .with_property("up_state", {
                "N_E": n_e,
                "N_I": 1000,
                "nu": 100,

                "N_nmda": 10,
                "nu_nmda": 10,
            }))
            sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(current_experiment,
                                                                                     "Look for Palmer firing rates",
                                                                                     increasing_nmda)

    '''
    Looking for No NMDA -> 0.05 Hz and NMDA -> 0.3 Hz
    Search for these values using binary search
    '''

    def test_run_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property(PlotParams.KEY_WHAT_PLOTS_TO_SHOW,
                                                                             [PlotParams.AvailablePlots.RASTER_AND_RATE])
                             .with_property("up_state",
                                            {
                                                "N": 2000,
                                                "nu": 82,
                                                "N_nmda": 0,
                                                "nu_nmda": 10,
                                            })).with_property("t_range", [[0, 1_000]])
        # sim_and_plot_up_down(palmer_experiment)
        results_with_nmda = sim_and_plot_up_down(palmer_experiment.with_property("up_state",
                                                                                 {
                                                                                     "N": 2000,
                                                                                     "nu": 82,
                                                                                     "N_nmda": 10,
                                                                                     "nu_nmda": 5,
                                                                                 }))

        print(results_with_nmda)
        number_of_spikes = len(results_with_nmda.spikes['all_values']['t'][0]) / palmer_experiment.sim_time

    def test_search_for_nmda_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 0,
                                                                                 "nu_nmda": 10,
                                                                             })).with_property("t_range", [[0, 1_000]])

        execute_palmer = lambda nmda_strength: run_with_NMDA_and_obtain_firing_rate(palmer_experiment, nmda_strength)
        res = binary_search_for_target_value(lower_value=0, upper_value=5E-8, func=execute_palmer, target_result=0.3)
        print(f"XXXXXXXXXXXXX {res}")
        # (1.5625e-10, 3.125e-10)

    '''
    What I wanted to do, is produce a trace with NMDA block that produces rate of 0.1 Hz. I.e. 1 spike in 10 s.
    (for 0.05 Hz, which is actually what we require, but for this, we would need 20s of simulation showing 1 spike).
    
    However, I forgot to set the downstate also do N=0, i.e. in the downstate there is in the simulation some NMDA input.
    
    When the model transitions from down to up state, the S variable is still active from previous input (down state) 
    and sigma(v) transitions to its value in the upstate "instantaneously".i.e. the NMDA input in the down state becomes
    relevant, although there were no gates "available" in the down state (the counting process is not really correctly modelled).
    
    However, the examples with down state N_nmda set to zero is different because different random realizations .
    '''
    def test_(self):
        palmer_experiment = (Experiment(wang_recurrent_config)
        .with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 82.3,
                    "N_nmda": 0,
                },
            "t_range": [[0, 10_000]],
            Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE,
                                                PlotParams.AvailablePlots.CURRENTS,
                                                PlotParams.AvailablePlots.HIDDEN_VARIABLES],
            #PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE],
            Experiment.KEY_HIDDEN_VARIABLES_TO_RECORD: ["x_nmda", "s_nmda", "v_minus_e_gaba", "sigmoid_v"],


        }))
        sim_and_plot_up_down(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment.with_property(SynapticParams.KEY_G_NMDA,0).with_property("panel", "G_NMDA set to 0"))
        sim_and_plot_up_down(palmer_experiment.with_property(SynapticParams.KEY_X_NMDA,0).with_property("panel", "X_NMDA set to 0"))
        sim_and_plot_up_down(palmer_experiment.with_property("down_state", {
                "N_E": 100,
                "gamma": 4,
                "nu": 10,
                "N_nmda": 0,
                "nu_nmda": 2,
             }).with_property("panel", "Down State set to N_NMDA = 0"))


class ScriptsPalmerResultsWithoutNMDA(unittest.TestCase):

    # produces rate 0.05 Hz with up/down
    def test_example_1(self):
        palmer_experiment = (Experiment(wang_recurrent_config)
        .with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 0,
                },
            #"down_state": {
            #    "N_E": 100,
            #    "gamma": 4,
            #    "nu": 10,

            #    "N_nmda": 0,
            #    "nu_nmda": 2,
            #},
            "t_range": [[0, 10_000]]
        }))
        sim_and_plot_up_down(palmer_experiment)
        #sim_and_plot_up_down(palmer_experiment.with_property(Experiment.KEY_CURRENTS_TO_RECORD, ["I_nmda"]))

    # produces rate 0.05 Hz with up/down
    def test_example_2(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_properties({
            "up_state":
                {
                    "N": 2000,
                    "nu": 81.54187093603468,
                    "N_nmda": 0,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        sim_and_plot_up_with_state_and_nmda(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment)

    def test_search_for_nmda_rates_without_NMDA(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 0,
                                                                                 "nu_nmda": 0,
                                                                             })).with_property("t_range",
                                                                                               [[0, 1_000]])

        execute_palmer = lambda nu: find_firing_rate_without_NMDA_with_nu(palmer_experiment, nu)
        res = binary_search_for_target_value(lower_value=80, upper_value=100, func=execute_palmer,
                                             target_result=0.05)
        self.assertEqual(81.5418709360165, res[0])
        self.assertEqual(81.54187093603468, res[1])


class ScriptsPalmerResultsWithNMDA(unittest.TestCase):

    def test_example_2_produ(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_properties({
            SynapticParams.KEY_G_NMDA: 1.35E-9,
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 10,
                    "nu_nmda": 5,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW:
                [PlotParams.AvailablePlots.RASTER_AND_RATE]
        }))

        sim_and_plot_up_with_state_and_nmda(palmer_experiment)
        sim_and_plot_up_down(palmer_experiment)

    # (2.5e-10, 5e-10)
    def test_search_for_nmda_palmer_rates(self):
        palmer_experiment = (Experiment(wang_recurrent_config).with_property("up_state",
                                                                             {
                                                                                 "N": 2000,
                                                                                 "nu": 82,
                                                                                 "N_nmda": 10,
                                                                                 "nu_nmda": 5,
                                                                             }))

        execute_palmer = lambda nmda_strength: run_with_NMDA_and_obtain_firing_rate(palmer_experiment, nmda_strength)
        res = binary_search_for_target_value(lower_value=0, upper_value=1E-9, func=execute_palmer, target_result=0.3)
        print(f"XXXXXXXXXXXXX {res}")

    def test_run_grid(self):
        interesting_nmdas = [1E-10, 1E-9, 1.35E-9]
        sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(
            palmer_experiment_0_1_Hz_with_NMDA_block,
            "Palmer", interesting_nmdas)


class ScriptsMeetingsWeek12to16January2026(unittest.TestCase):

    def test_exemplify_model_with_up_and_down_state(self):
        # sim_and_plot_up_down(palmer_experiment_0_1_Hz_with_NMDA_block)
        # sim_and_plot_up_down(palmer_experiment_0_1_Hz_with_NMDA_block.with_property(Experiment.KEY_CURRENTS_TO_RECORD, ["I_nmda"]))
        experiment = palmer_experiment_0_1_Hz_with_NMDA_block.with_properties({
            "panel": "\nPalmer run.  Figure 2 D, E rate for MK801",
            "t_range": [0, 10_000]
        })
        sim_and_plot_up_down(experiment)
        sim_and_plot_up_down(experiment.with_property(Experiment.KEY_CURRENTS_TO_RECORD, ["I_nmda"]))

    def test_compare_models_normal_nmda_and_full_activation(self):
        experiment = palmer_experiment.with_property( "t_range", [0, 10_000])

    def test_compare_models_normal_nmda_and_full_activation(self):
        experiment = palmer_experiment.with_property( "t_range", [0, 10_000])
        sim_and_plot_up_down(experiment.with_property( "panel", r"Experiment with correct $\sigma(v)$ "))
        sim_and_plot_up_down(experiment.with_properties({
            Experiment.KEY_SELECTED_MODEL: single_compartment_without_nmda_deactivation_and_logged_variables,
            "panel": r"Experiment with $\sigma(v) = 1 $ constant ",
        }))


    def test_palmer_with_NMDA_block_and_without(self):
        palmer_experiment_with_nmda = (Experiment(wang_recurrent_config)
        .with_properties({
            SynapticParams.KEY_G_NMDA: 0.9e-9,
            "up_state":
                {
                    "N": 2000,
                    "nu": 82,
                    "N_nmda": 10,
                    "nu_nmda": 10,
                },
            "t_range": [[0, 10_000]],
            PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS],
            Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
            "panel": "Control"
        }))
        palmer_experiment_with_nmda_block = palmer_experiment_with_nmda.with_properties({
            SynapticParams.KEY_X_NMDA: 0,
            "panel": "NMDA block"
        })
        sim_results_no_nmda = sim_and_plot_up_down(palmer_experiment_with_nmda_block)
        sim_resuts_control = sim_and_plot_up_down(palmer_experiment_with_nmda)

        plot_voltage_trace_comparisons(sim_resuts_control, sim_results_no_nmda, params_t_range=[[0, 10_000], [5300, 5600], [6800, 8100], [9250, 9550]])

    def test_find_control_palmer_with_lowest_ndma(self):
        for nmda_strength in np.linspace(.6, 1, 5) * 1E-9:
            palmer_experiment_with_nmda = (Experiment(wang_recurrent_config)
            .with_properties({
                SynapticParams.KEY_G_NMDA: nmda_strength,
                "up_state":
                    {
                        "N": 2000,
                        "nu": 82,
                        "N_nmda": 10,
                        "nu_nmda": 10,
                    },
                "t_range": [[0, 10_000]],
                PlotParams.KEY_WHAT_PLOTS_TO_SHOW: [PlotParams.AvailablePlots.RASTER_AND_RATE, PlotParams.AvailablePlots.CURRENTS],
                Experiment.KEY_CURRENTS_TO_RECORD: ["I_nmda"],
                "panel": "Control"
            }))
            sim_and_plot_up_down(palmer_experiment_with_nmda)

    '''
    The point that Palmer makes is:
    Of these, 22 ± 3% of hindpaw stimulation trials had a large local Ca 2+ transient, which always resulted in an action potential at the soma
(n = 11 branches; Fig. 2a–c). In contrast, hindpaw stimulation that did not evoke dendritic activity also did not usually evoke action
potentials. In those cases in which action potentials were evoked without a dendritic Ca2+ transient, the number of action potentials was
significantly less (2.1 ± 0.2 versus 3.9 ± 0.7 action potentials, n = 11 branches, P < 0.05)
    '''
    def test_make_nmda_in_up_state_fire_very_seldomly(self):
        pass

    def test_compute_rate_for_palmer_experiment(self):
        control_results = sim_and_plot_up_with_state_and_nmda(palmer_control.with_property("t_range", [0, 100_000]))
        results_with_nmda_block = sim_and_plot_up_with_state_and_nmda(palmer_nmda_block.with_property("t_range", [0, 100_000]))

        spikes_control = len(control_results.spikes)
        spikes_with_nmda_block = len(results_with_nmda_block.spikes)
        print(f"YYYYYYYYYYYYYYYY {spikes_control}, {spikes_with_nmda_block}")

    def test_compute_vm_distribution_for_palmer_exp(self):
        #simulate_with_up_state_and_nmda(palmer_experiment_with_nmda)
        results_with_nmda = simulate_with_up_state_and_nmda(palmer_control.with_property("t_range", [0, 100_000]).with_property("theta", -40))
        results_with_nmda_block = simulate_with_up_state_and_nmda(palmer_nmda_block.with_property("t_range", [0, 100_000]).with_property("theta", -40))

        plot_and_compare_two_voltages_curves(results_with_nmda, results_with_nmda_block)


def plot_and_compare_two_voltages_curves(results_1: SimulationResults,results_2: SimulationResults, ignore_first_items=10_000):
    from scipy.stats import norm


    counts_1, bin_centers_1, bin_edges_1 = fit_voltages_curve_to_gaussian(results_1.voltages.v[0], ignore_first_items)
    counts_2, bin_centers_2, bin_edges_2 = fit_voltages_curve_to_gaussian(results_2.voltages.v[0], ignore_first_items)

    print(f"XXXXXXXXXXXXXXXX {overlap_from_histograms(counts_1, bin_centers_1, counts_2, bin_centers_2)} ")
    print(f"XXXXXXXXXXXXXXXX {overlap_from_histograms(counts_1, bin_centers_1, counts_1, bin_centers_1)} ")
    print(f"XXXXXXXXXXXXXXXX {overlap_from_histograms(counts_2, bin_centers_2, counts_2, bin_centers_2)} ")

    x = np.linspace(np.min([bin_edges_1[0], bin_centers_2[0]]), np.max([bin_edges_1[-1], bin_centers_2[-1]]), 1_000)

    mean_1 = np.mean(results_1.voltages.v[0][ignore_first_items:])
    std_1 = np.std(results_1.voltages.v[0][ignore_first_items:])

    mean_2 = np.mean(results_2.voltages.v[0][ignore_first_items:])
    std_2 = np.std(results_2.voltages.v[0][ignore_first_items:])

    gaussian_pdf_1 = norm.pdf(x, mean_1, std_1)
    gaussian_pdf_2 = norm.pdf(x, mean_2, std_2)

    plt.figure(figsize=(8, 5))
    # Bar plot
    plt.bar(
        bin_centers_1,
        counts_1,
        width=np.diff(bin_edges_1),
        align="center",
        alpha=0.5,
        color="black",
        label=results_1.experiment.plot_params.panel
    )
    # Gaussian fit
    plt.plot(x, gaussian_pdf_1, "r-", linewidth=2,
             label=f"{results_1.experiment.plot_params.panel}\n mean={mean_1: .4f}, variance={std_1: .4f}", color="black")

    plt.bar(
        bin_centers_2,
        counts_2,
        width=np.diff(bin_edges_2),
        align="center",
        alpha=0.7,
        color="orange",
        label=results_2.experiment.plot_params.panel
    )
    plt.plot(x, gaussian_pdf_2, "r-", linewidth=2,
             label=f"{results_2.experiment.plot_params.panel} \n mean={mean_2: .4f}, variance={std_2: .4f}", color="darkorange")

    ymax = np.max([counts_1.max(), gaussian_pdf_1.max(), counts_2.max(), gaussian_pdf_2.max()]) + 0.3

    plt.vlines(
        mean_1,
        ymin=0,
        ymax=ymax,
        colors="black",
        linestyles="--",
        linewidth=2,
        label=f"{results_1.experiment.plot_params.panel}, Mean = {mean_1:.3f}"
    )

    plt.vlines(
        mean_2,
        ymin=0,
        ymax=ymax,
        colors="darkorange",
        linestyles="--",
        linewidth=2,
        label=f"{results_2.experiment.plot_params.panel}, Mean = {mean_2:.3f}"
    )

    plt.title("Comparison of Gaussian fits of membrane voltage for Control and NMDA Blocks Palmer simulations \n"
              f"{results_1.experiment.plot_params.panel} = [Mean = {mean_1: .3f}, STD = {std_1: .3f}]\n "
              f"{results_2.experiment.plot_params.panel} = [Mean = {mean_2: .3f}, STD = {std_2: .3f}] \n"
              f"Histogram overlap {overlap_from_histograms(counts_1, bin_centers_1, counts_2, bin_centers_2) * 100: .4f} %")

    plt.legend()
    plt.tight_layout()
    show_plots_non_blocking()


def fit_voltages_curve_to_gaussian(v_curve, ignore_first_items=10_000):
    data = v_curve[ignore_first_items:]

    counts, bin_edges = np.histogram(data, bins=1000, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return counts, bin_centers, bin_edges


def generate_title(experiment: Experiment):
    return fr"""Compare control and NMDA-blocked voltage traces
    Up State: [{experiment.network_params.up_state.gen_plot_title()}, {experiment.effective_time_constant_up_state.gen_plot_title()}]
    Down State: [{experiment.network_params.down_state.gen_plot_title()}, {experiment.effective_time_constant_down_state.gen_plot_title()}]    
    Neuron: [$C={experiment.neuron_params.C}$, $g_L={experiment.neuron_params.g_L}$, $\theta={experiment.neuron_params.theta}$, $V_R={experiment.neuron_params.V_r}$, $E_L={experiment.neuron_params.E_leak}$, $\tau_M={experiment.neuron_params.tau}$, $\tau_{{\mathrm{{ref}}}}={experiment.neuron_params.tau_rp}$]
    Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa / nS:.2f} nS$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba / nS:.2f}, nS$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda / nS:.2f} nS $]"""


def bayes_error_rate(pdf1, pdf2, x_min=-55, x_max=-35):
    x=np.linspace(x_min, x_max, len(pdf1))
    min_pdf = np.min([pdf1, pdf2], axis=0)
    return .5 * np.trapz(y=min_pdf, x=x, dx=1E-6)

def overlap_from_histograms(counts1, centers1, counts2, centers2,
                            bayes_error=False, d_x=2000):

    # Common support
    x_min = max(centers1.min(), centers2.min())
    x_max = min(centers1.max(), centers2.max())

    x = np.linspace(x_min, x_max, d_x)

    # Interpolate PDFs onto common grid
    pdf1 = np.interp(x, centers1, counts1, left=0, right=0)
    pdf2 = np.interp(x, centers2, counts2, left=0, right=0)

    overlap = np.trapezoid(np.minimum(pdf1, pdf2), x)

    if bayes_error:
        return 0.5 * overlap
    else:
        return overlap


if __name__ == '__main__':
    unittest.main()
