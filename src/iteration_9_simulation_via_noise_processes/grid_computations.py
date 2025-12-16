from typing import Callable

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from iteration_7_one_compartment_step_input.grid_computations import grid_title, parallelize
from iteration_8_compute_mean_steady_state.grid_computations import plot_results_grid
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import \
    SimulationResultsWithSteadyState, plot_simulation
from iteration_9_simulation_via_noise_processes.one_compartment_with_difusion_process import \
    sim_diffusion_process_with_up_down


def sim_and_plot_experiment_grid_with_increasing_nmda_input_and_diffusion_process(experiment: Experiment, title, nmda_schedule: list[float]):
    experiments = [experiment.with_property("g_nmda", nmda_strength) for nmda_strength in nmda_schedule]
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_diffusion_process_with_up_down)


def run_simulate_diffusion_process_with_up_down(experiments: list[Experiment]):
    return parallelize(experiments, sim_diffusion_process_with_up_down)

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