from typing import Callable

import matplotlib.pyplot as plt
from brian2 import cm
from joblib import Parallel, delayed
from matplotlib import gridspec

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.grid_computations import sim_and_plot_experiment_grid_with_lambda, \
    grid_title, parallelize
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import \
    simulate_with_up_and_down_state_and_nmda_and_steady_state, SimulationResultsWithSteadyState, plot_raster_and_rates, \
    plot_voltages_and_g_s, plot_currents, plot_simulation


def sim_and_plot_experiment_grid_with_increasing_nmda_input_and_steady_state(experiment: Experiment, title, nmda_schedule: list[float]):
    experiments = [experiment.with_property("g_nmda", nmda_strength) for nmda_strength in nmda_schedule]
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_with_steady_state)

def run_simulate_with_steady_state(experiments: list[Experiment]):
    return parallelize(experiments, simulate_with_up_and_down_state_and_nmda_and_steady_state)

def sim_and_plot_experiment_grid_with_lambda(experiments: list[Experiment], title,
                                             obtain_results_function: Callable[
                                                 [list[Experiment]], list[SimulationResultsWithSteadyState]]):
    results = obtain_results_function(experiments)

    t_range = experiments[0].plot_params.t_range
    if t_range:
        params_t_range = t_range

        if isinstance(params_t_range[0], list):
            for time_slot in params_t_range:
                plot_results_grid(results, time_slot, title=title)
        else:
            plot_results_grid(results, t_range, title=title)

    for result in results:
        plot_simulation(result)
        show_plots_non_blocking()

    return results

def parallelize(experiments: list[Experiment],
                function_simulate_one_experiment: Callable[[Experiment], SimulationResultsWithSteadyState]):
    def sim_unpickled(experiment: Experiment):
        simulation_results = function_simulate_one_experiment(experiment)
        simulation_results.internal_states_monitor = None
        return simulation_results

    return Parallel(n_jobs=1)(
        delayed(sim_unpickled)(current_experiment) for current_experiment in experiments
    )


def plot_results_grid(results: list[SimulationResultsWithSteadyState], time_range: tuple[int, int], title: str):
    plot_grid_raster_population_and_g_s(results, time_range, title=title)
    plot_grid_currents(results, time_range, title=title)


def plot_grid_raster_population_and_g_s(results: list[SimulationResultsWithSteadyState], time_range: tuple[int, int], title: str):
    if not results[0].experiment.plot_params.show_raster_and_rate():
        return

    fig = plt.figure(figsize=(35, 25))
    fig.suptitle(title, size=25)

    outer = gridspec.GridSpec(2, len(results), figure=fig, hspace=0.2,
                              wspace=0.1)

    for index, result in enumerate(results):
        ax_spikes, _ = plot_raster_and_rates(result, time_range, outer[0, index])
        plot_voltages_and_g_s(result, time_range, outer[1, index])

        ax_spikes.set_title(f"{gen_raster_and_rates_grid_subtitle(result)} \n {ax_spikes.get_title()}")

    show_plots_non_blocking()


def gen_raster_and_rates_grid_subtitle(results: SimulationResultsWithSteadyState):
    experiment = results.experiment
    return fr"""Synapse: [$g_{{\mathrm{{AMPA}}}}={experiment.synaptic_params.g_ampa * (cm ** 2):.2f}$, $g_{{\mathrm{{GABA}}}}={experiment.synaptic_params.g_gaba * cm ** 2:.2f}$, $g={experiment.network_params.g}$, $g_{{\mathrm{{NMDA}}}}={experiment.synaptic_params.g_nmda * cm ** 2:.2f}$]"""


def plot_grid_currents(results, time_range, title: str):
    if not results[0].experiment.plot_params.show_currents_plots():
        return

    fig = plt.figure(figsize=(35, 25))
    fig.suptitle(title, size=25)

    outer = gridspec.GridSpec(1, len(results), figure=fig, hspace=0.2,
                              wspace=0.1)

    for index, result in enumerate(results):
        plot_currents(result, time_range, outer[index])

    show_plots_non_blocking()