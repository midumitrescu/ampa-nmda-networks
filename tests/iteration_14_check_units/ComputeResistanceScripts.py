import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import mV, Mohm
from brian2.units.allunits import nsiemens
from joblib import Parallel, delayed

from Plotting import show_plots_non_blocking
from iteration_14_check_units.simulate_V_Clamp import simulate_current_injection, \
    run_current_injection_simulation
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment, CurrentClampParams
from iteration_8_compute_mean_steady_state.models_and_configs import palmer_experiment, \
    wang_recurrent_config, steady_model_with_full_activation
from iteration_8_compute_mean_steady_state.one_compartment_with_up_down_and_steady import sim_steady_state

plt.rcParams['text.usetex'] = False


class MyTestCase(unittest.TestCase):

    def test_r_in_computation(self):
        palmer_current_clamp = Experiment(wang_recurrent_config)

        injected_currents = np.linspace(-100, 100, 100)
        current_clamp_experiments = [palmer_current_clamp.with_property(CurrentClampParams.KEY_I_INJECTED, current) for
                                     current in injected_currents]

        current_simulations = Parallel(n_jobs=-1)(
            delayed(simulate_current_injection)(current_experiment) for current_experiment in current_clamp_experiments
        )

        no_up_state, in_up_state = map(list, zip(*current_simulations))

        df_no_up_state = pd.DataFrame([res.to_dict() for res in no_up_state])
        df_in_up_state = pd.DataFrame([res.to_dict() for res in in_up_state])

        plt.plot(injected_currents, df_no_up_state.v_steady_mV, label="no up")
        plt.plot(injected_currents, df_in_up_state.v_steady_mV, label="with steady up state")

        r_in_theory_no_input = 1 / palmer_current_clamp.neuron_params.g_L / Mohm
        r_in_theory_up_state = 1 / palmer_current_clamp.effective_time_constant_up_state.mean_total_conductance_with_nmda() / Mohm

        plt.title(
            f"No input: {r"$R_\text{in, theory}=$"}{r_in_theory_no_input: .3f} MΩ. {r"$R_\text{in, sim}=$"}{df_no_up_state.r_in_MOhm.mean(): .3f} MΩ. var={df_no_up_state.r_in_MOhm.var() : 4f} \n"
            f"Up State, Steady: {r"$R_\text{in, theory}=$"}{r_in_theory_up_state: .3f} MΩ. {r"$R_\text{in, sim}=$"}{df_in_up_state.r_in_MOhm.mean(): .3f} MΩ. var={df_in_up_state.r_in_MOhm.var(): 4f}")

        plt.xlabel(r"$I_\text{inj}$ [pAmpere]")
        plt.ylabel(r"$V_\text{m}$ [mV]")

        plt.show()
        show_plots_non_blocking()

        print(
            f"No up. Theoretical={r_in_theory_no_input} Mean {df_no_up_state.r_in_MOhm.mean()}, variance {df_no_up_state.r_in_MOhm.var()},  {df_no_up_state.r_in_MOhm}")
        print(
            f"With up. Theoretical={r_in_theory_up_state} Mean {df_in_up_state.r_in_MOhm.mean()}, variance {df_in_up_state.r_in_MOhm.var()},  {df_no_up_state.r_in_MOhm}")

    def test_steady_state_g_nmda_vs_theoretical_computation(self):
        experiment_under_test = Experiment(wang_recurrent_config)

        steady_state_results = sim_steady_state(experiment_under_test, experiment_under_test.network_params.up_state)

        r_in_from_steady_state = 1 / (
                experiment_under_test.neuron_params.g_L + steady_state_results.g_e_steady * nsiemens + steady_state_results.g_i_steady * nsiemens + steady_state_results.g_nmda_steady * nsiemens) / Mohm
        r_in_from_effective_time_constant = 1 / palmer_experiment.effective_time_constant_up_state.mean_total_conductance_with_nmda() / Mohm

        print(
            f"Steady State Results g nmda {steady_state_results.g_nmda_steady: .4f} vs {experiment_under_test.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens: .4f} from effective time constant")
        print(
            f"Delta {abs(steady_state_results.g_nmda_steady - experiment_under_test.effective_time_constant_up_state.compute_mean_g_nmda() / nsiemens)}")

        print(
            f"Steady state estimated r_in {r_in_from_steady_state} vs {r_in_from_effective_time_constant} from effective time constant")
        print(
            f"Delta {abs(r_in_from_steady_state - r_in_from_effective_time_constant)}")

    def test_blabla_2(self):
        wang_current_clamp = Experiment(wang_recurrent_config)

        injected_currents = np.linspace(-100, 100, 100)
        injected_currents = injected_currents[injected_currents != 0]

        df_no_up_state, df_in_up_state = run_current_injection_simulation(experiment=wang_current_clamp,
                                                                          injected_currents=injected_currents)

        rest_state_no_current = sim_steady_state(wang_current_clamp.with_no_current())
        up_state_no_current = sim_steady_state(wang_current_clamp.with_no_current(),
                                               state=wang_current_clamp.network_params.up_state)

        plt.plot(injected_currents, df_no_up_state.v_steady_mV, label="no up")
        plt.plot(injected_currents, df_in_up_state.v_steady_mV, label="with steady up state")

        r_in_theory_no_input = 1 / wang_current_clamp.neuron_params.g_L / Mohm
        r_in_theory_up_state = 1 / wang_current_clamp.effective_time_constant_up_state.mean_total_conductance_with_nmda() / Mohm

        # no up state
        slope_no_up_state, intercept_no_up = np.polyfit(
            injected_currents,
            df_no_up_state.v_steady_mV,
            1
        )

        # with steady up state
        slope_with_up_state, intercept_up = np.polyfit(
            injected_currents,
            df_in_up_state.v_steady_mV,
            1
        )

        plt.title(
            "I-V Curve for our model \n"
            f"Theory: Rest = {r"$R_\mathrm{in}=$"}{r_in_theory_no_input: .3f} MΩ, Up State = {r"$R_\mathrm{in}=$"}{r_in_theory_up_state: .3f} MΩ \n"
            f"Steady state: Rest = {r"$R_\mathrm{in}=$"}{rest_state_no_current.r_in: .3f} MΩ, Up State = {r"$R_\mathrm{in}=$"}{up_state_no_current.r_in: .3f} MΩ \n"
            f"Simulation: Rest = {r"$R_\text{in}=$"}{df_no_up_state.r_in_MOhm.mean(): .3f} MΩ. var={df_no_up_state.r_in_MOhm.var() * 1E9: .1f} {r"$\cdot 10^{-9}$"} \n "
            f"Up State = {df_in_up_state.r_in_MOhm.mean(): .3f} MΩ. var={df_in_up_state.r_in_MOhm.var() * 1E9: .1f} {r"$\cdot 10^{-9}$"}\n"
            f"from slope Rest = {slope_no_up_state * 1E3: .3f} MΩ , UP from slope = {slope_with_up_state * 1E3: .3f} MΩ"
        )

        plt.xlabel(r"$I_\text{inj}$ [pAmpere]")
        plt.ylabel(r"$V_\text{m}$ [mV]")
        plt.tight_layout()
        plt.show()
        show_plots_non_blocking()

        print(
            f"No up. Theoretical={r_in_theory_no_input} Mean {df_no_up_state.r_in_MOhm.mean()}, variance {df_no_up_state.r_in_MOhm.var()},  {df_no_up_state.r_in_MOhm}")
        print(
            f"With up. Theoretical={r_in_theory_up_state} Mean {df_in_up_state.r_in_MOhm.mean()}, variance {df_in_up_state.r_in_MOhm.var()},  {df_no_up_state.r_in_MOhm}")

    def test_why_is_r_in_up_state_weird(self):
        injected_currents = np.linspace(-100, 100, 3)

        experiment = Experiment(wang_recurrent_config)
        steady_state_results_no_current = sim_steady_state(
            experiment.with_property(CurrentClampParams.KEY_I_INJECTED, 0),
            state=experiment.network_params.up_state)
        print(
            f"============================= Steady {steady_state_results_no_current.v_steady * mV} ==========================")

        current_clamp_for_up_state = [experiment.with_properties({
            CurrentClampParams.KEY_I_INJECTED: current,
        }) for current in injected_currents]
        res = [sim_steady_state(current_experiment, current_experiment.network_params.up_state, plot_details=True).recompute_r_in(
            other=steady_state_results_no_current) for current_experiment in current_clamp_for_up_state]


        # look at conductances
        print(res[0].g_e_steady)

        self.assertAlmostEqual(20.422650116904904, res[0].r_in)
        self.assertAlmostEqual(20.398919510752673, res[1].r_in) # this one I don't like!
        self.assertAlmostEqual(20.42277505780683, res[2].r_in)

        print(res[0].to_dict())
        print(res[1].to_dict())
        print(res[2].to_dict())


        print("Is r_in[0] > r_in[2]? ", res[0].r_in > res[2].r_in)
        print("Even in recompute?")
        r_in_0_recomp = (res[0].v_steady - steady_state_results_no_current.v_steady) / (injected_currents[0]) * 1000
        r_in_2_recomp = (res[2].v_steady - steady_state_results_no_current.v_steady) / (injected_currents[2]) * 1000
        print(f"Is r_in recomp [0] [{r_in_0_recomp}] > r_in recomp [2] [{r_in_2_recomp}]? ", r_in_0_recomp > r_in_2_recomp)

        self.assertLess(abs(r_in_0_recomp - res[0].r_in), 1E-15)
        self.assertLess(abs(r_in_2_recomp - res[2].r_in), 1E-15)

    def test_why_r_in_for_injected_current_larger(self):
        injected_currents = np.linspace(-100, 100, 3)

        experiment = Experiment(wang_recurrent_config).with_property(Experiment.KEY_STEADY_MODEL, steady_model_with_full_activation)
        steady_state_results_no_current = sim_steady_state(
            experiment.with_property(CurrentClampParams.KEY_I_INJECTED, 0),
            state=experiment.network_params.up_state)
        print(f"============================= Steady {steady_state_results_no_current.v_steady * mV} ==========================")

        current_clamp_for_up_state = [experiment.with_properties({
            CurrentClampParams.KEY_I_INJECTED: current,
        }) for current in injected_currents]
        res = [sim_steady_state(current_experiment, current_experiment.network_params.up_state,
                                plot_details=True).recompute_r_in(
            other=steady_state_results_no_current) for current_experiment in current_clamp_for_up_state]

        df = pd.DataFrame([r.to_dict() for r in res])

        print(res[0].to_dict())
        print(res[1].to_dict())
        print(res[2].to_dict())

        df['g_tot'] = df.g_e_steady_nS + df.g_i_steady_nS + df.g_nmda_steady_nS
        df['r_in_comp'] = (df.v_steady_mV - steady_state_results_no_current.v_steady) / injected_currents * 1_000

        print(f"Compare  g tot [0] [{res[0].g_e_steady + res[0].g_i_steady + res[0].g_nmda_steady}]  to g tot [2] [{res[2].g_e_steady + res[2].g_i_steady + res[2].g_nmda_steady}]")
        print("Is g tot [0] > g tot [2]? ", res[0].g_e_steady + res[0].g_i_steady + res[0].g_nmda_steady > res[2].g_e_steady + res[2].g_i_steady + res[2].g_nmda_steady)
        print(f"Is r_in[0] > r_in[2]? {res[0].r_in > res[2].r_in}. Delta {res[0].r_in - res[2].r_in}")

        print(f"[0] Is computed r_in same? {res[0].r_in} vs {df.r_in_comp[0]}. Delta = {res[0].r_in - df.r_in_comp[0]}")
        print(f"[2] Is computed r_in same? {res[2].r_in} vs {df.r_in_comp[2]}. Delta = {res[2].r_in - df.r_in_comp[2]}")

        print(df)

        print("Is r_in[0] > r_in[2]? ", res[0].r_in > res[2].r_in)
        print("Even in recompute?")
        r_in_0_recomp = (res[0].v_steady - steady_state_results_no_current.v_steady) / (injected_currents[0]) * 1000
        r_in_2_recomp = (res[2].v_steady - steady_state_results_no_current.v_steady) / (injected_currents[2]) * 1000
        print(f"Is r_in recomp [0] [{r_in_0_recomp}] > r_in recomp [2] [{r_in_2_recomp}]? ",
              r_in_0_recomp > r_in_2_recomp)

        self.assertLess(abs(r_in_0_recomp - res[0].r_in), 1E-15)
        self.assertLess(abs(r_in_2_recomp - res[2].r_in), 1E-15)


if __name__ == '__main__':
    unittest.main()
