"""
Runnable Siegert μ–σ scripts: plots and scans.

Convention: test_scripts_* = runnable experiments/plots (discovered by IntelliJ as tests).
"""
import unittest
import numpy as np
from brian2 import mV, Hz, mvolt

from iteration_12_transfer_function_of_lif_neurons.LookForAllSolutionsMuSigma import (
    compute_mu_to_sigma_curve,
    compute_mu_to_sigma_fsolve_scan_mus,
    compute_mu_to_sigma_fsolve_scan_sigmas,
    plot_mus_vs_sigmas,
    plot_loss_landscape_with_curve,
)
from iteration_12_transfer_function_of_lif_neurons.SiegertGradientDescent import (
    SiegertGradientDescent,
    SiegertGradients,
)
from iteration_8_compute_mean_steady_state.scripts_with_wang_numbers import palmer_control

try:
    from src.Plotting import NeuronModelParams
except ImportError:
    from Plotting import NeuronModelParams


class LookForAllSolutionsScripts(unittest.TestCase):
    """Runnable scripts: μ–σ curves, loss landscape, fsolve scans. test_scripts_* = run directly in IntelliJ."""

    def test_scripts_look_for_all_solutions(self):
        experiment = palmer_control
        r_target = 0.3 * Hz
        solver = SiegertGradientDescent(
            tau_m=experiment.effective_time_constant_up_state.tau_eff(),
            theta=experiment.neuron_params.theta,
            v_reset=experiment.neuron_params.V_r,
            tau_ref=experiment.neuron_params.tau_rp,
            unit="mV",
        )
        fig, _ = plot_loss_landscape_with_curve(
            solver, r_target, (-60, -35), (0, 6),
            resolution=50, target_rate_Hz=0.3, caller_test_case=self,
        )
        fig.show()

    def test_scripts_look_for_all_solutions_using_binary_search(self):
        experiment = palmer_control
        palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve(
            experiment.with_label("NMDA Block"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve(
            experiment.with_label("Control"), r_target=0.3 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        mu_60_1_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_60_1 = control_mu_to_sigma.mus[mu_60_1_arg] * mvolt
        sigma_for_mu_60_1 = control_mu_to_sigma.sigmas[mu_60_1_arg] * mvolt
        rate_60_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        rate_60_1 = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_60_1, sigma_v=sigma_for_mu_60_1
        )
        print(rate_60_mv, rate_60_1)
        print(
            f"Delta in mu {(mu_close_to_60_1 - mu_close_to_60) / mV}. "
            f"Delta in sigma {(sigma_for_mu_60_1 - sigma_for_mu_60) / mV}."
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_binary_search_palmer_rates(self):
        experiment = palmer_control
        palmer_control.with_property(NeuronModelParams.KEY_NEURON_V_R, -55)
        nmda_block_mu_to_sigma = compute_mu_to_sigma_curve(
            experiment.with_label("NMDA Block"), r_target=0.05 * Hz
        )
        control_mu_to_sigma = compute_mu_to_sigma_curve(
            experiment.with_label("Control"), r_target=0.18 * Hz
        )
        mu_60_arg = np.abs(nmda_block_mu_to_sigma.mus - (-60)).argmin()
        mu_close_to_60 = nmda_block_mu_to_sigma.mus[mu_60_arg] * mvolt
        sigma_for_mu_60 = nmda_block_mu_to_sigma.sigmas[mu_60_arg] * mvolt
        rate_60_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_60, sigma_v=sigma_for_mu_60
        )
        mu_59_9_arg = np.abs(control_mu_to_sigma.mus + (60 - 0.1)).argmin()
        mu_close_to_59_9 = control_mu_to_sigma.mus[mu_59_9_arg] * mvolt
        sigma_for_mu_59_9 = control_mu_to_sigma.sigmas[mu_59_9_arg] * mvolt
        rate_59_9_mv = SiegertGradients.for_experiment(experiment).firing_rate(
            mu_v=mu_close_to_59_9, sigma_v=sigma_for_mu_59_9
        )
        print(
            f"mu = {mu_close_to_60 / mV:.4f}, sigma = {sigma_for_mu_60 / mV:.4f} "
            f"produces rate {rate_60_mv / Hz:.4f}"
        )
        print(
            f"mu = {mu_close_to_59_9 / mV:.4f}, sigma = {sigma_for_mu_59_9 / mV:.4f} "
            f"produces rate {rate_59_9_mv / Hz:.4f}"
        )
        plot_mus_vs_sigmas(
            [nmda_block_mu_to_sigma, control_mu_to_sigma],
            caller_test_case=self,
        )

    def test_scripts_look_for_all_solutions_using_fsolve_scan_sigma(self):
        result = compute_mu_to_sigma_fsolve_scan_sigmas(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(101, len(result.mus))
        self.assertEqual(101, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)

    def test_scripts_look_for_all_solutions_using_fsolve_scan_mu(self):
        result = compute_mu_to_sigma_fsolve_scan_mus(
            palmer_control, r_target=0.3 * Hz
        )
        print(result)
        self.assertEqual(1001, len(result.mus))
        self.assertEqual(1001, len(result.sigmas))
        plot_mus_vs_sigmas([result], caller_test_case=self)
