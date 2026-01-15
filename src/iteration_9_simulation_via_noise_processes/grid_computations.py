from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, State
from iteration_7_one_compartment_step_input.grid_computations import grid_title, parallelize
from iteration_8_compute_mean_steady_state.grid_computations import sim_and_plot_experiment_grid_with_lambda
from iteration_9_simulation_via_noise_processes.one_compartment_with_difusion_process import \
    sim_diffusion_process_with_up_down


def sim_and_plot_experiment_grid_with_increasing_nmda_input_and_diffusion_process(experiment: Experiment, title,
                                                                                  nmda_schedule: list[float]):
    experiments = [[experiment.with_property("g_nmda", nmda_strength) for nmda_strength in nmda_schedule]]
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_diffusion_process_with_up_down)


def sim_and_plot_experiment_grid_with_increasing_nmda_noise(experiment: Experiment, title,
                                                            nmda_noise_schedule: list[float]):
    experiments = [[experiment.with_property("up_state",
                                            {
                                                "N_E": 1600,
                                                "N_I": 400,
                                                "nu": 75,

                                                "N_nmda": 10,
                                                "nu_nmda": 10,
                                                State.KEY_X_VAR_MULT:  noise_mult

                                            }) for noise_mult in nmda_noise_schedule]]
    title = grid_title(panel_title=title, experiment=experiment)
    return sim_and_plot_experiment_grid_with_lambda(experiments, title, run_simulate_diffusion_process_with_up_down)


def run_simulate_diffusion_process_with_up_down(experiments: list[Experiment]):
    return parallelize(experiments, sim_diffusion_process_with_up_down)

