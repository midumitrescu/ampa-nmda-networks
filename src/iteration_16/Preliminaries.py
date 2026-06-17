import unittest

import numpy as np

from brian2 import ms, Hz, nS, mV

from BinarySeach import binary_search_for_target_value
from iteration_16.model import ConductanceDiffusionSimulationConfig, WANG_MODEL_FOR_FULL_NMDA_INPUT, WANG_MODEL, \
    config_with_weak_synapses
from iteration_16.simulation import WangSimulation, plot


class CalibratePresynapticDVs(unittest.TestCase):
    def test_ampa_dv_is_0_5_mv_at_soma(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            ampa_spike_times=np.array([50]))

        find_dv_of_ampa = lambda x: WangSimulation.run(config.with_property(w_ampa=x)).ampa_spike_delta_v()

        lower, upper = binary_search_for_target_value(lower_value=0.5 * nS, upper_value=3 * nS, func=find_dv_of_ampa,
                                                      target_result=0.5,
                                                      precision=1E-6 * nS)

        target_config = config.with_property(w_ampa=lower)
        dv_result = WangSimulation.run(target_config)
        plot(dv_result)

        self.assertAlmostEqual(2.399104714393616, lower / nS, places=1)
        self.assertAlmostEqual(0.5, dv_result.ampa_spike_delta_v())

    def test_gaba_dv_is_0_5_mv_at_soma(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            gaba_spike_times=np.array([50]))

        find_dv_of_gaba = lambda x: WangSimulation.run(config.with_property(w_gaba=x)).gaba_spike_delta_v()

        lower, _ = binary_search_for_target_value(lower_value=10 * nS, upper_value=0.5 * nS, func=find_dv_of_gaba,
                                                  target_result=-0.5,
                                                  precision=1E-6 * nS)

        target_config = config.with_property(w_gaba=lower)
        dv_result = WangSimulation.run(target_config)
        plot(dv_result)

        # self.assertAlmostEqual(4.131385296583174, lower / nS)
        self.assertAlmostEqual(5.077383026480675, lower / nS)
        self.assertAlmostEqual(-0.5, dv_result.gaba_spike_delta_v())

    '''
    Jackie Schiller in her famous paper NMDA spikes in basal dendrites of cortical pyramidal neurons: 
     the
amplitude of the cable-filtered basal dendritic spike was 5.2 +- 1.7
mV, as measured at the soma (n = 14)
    '''

    def test_nmda_dv_is_5_mv_at_soma(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=200 * ms,
            model=WANG_MODEL_FOR_FULL_NMDA_INPUT,
            nmda_spike_times=np.array([50]),
            seed=12345)

        find_dv_of_nmda = lambda x: WangSimulation.run(config.with_property(g_nmda_max=x)).nmda_spike_delta_v()

        lower, _ = binary_search_for_target_value(lower_value=3 * nS, upper_value=10 * nS, func=find_dv_of_nmda,
                                                  target_result=5,
                                                  precision=1E-6 * nS)

        config_with_calibrated_nmda = config.with_property(g_nmda_max=lower)
        dv_result = WangSimulation.run_and_plot(config_with_calibrated_nmda)

        self.assertAlmostEqual(4.074575871229172, lower / nS)
        self.assertAlmostEqual(5, dv_result.nmda_spike_delta_v(), places=6)

        dv_result_with_sigma_v = WangSimulation.run_and_plot(
            config_with_calibrated_nmda.with_property(model=WANG_MODEL))
        self.assertAlmostEqual(0.3223709291851691, dv_result_with_sigma_v.nmda_spike_delta_v(), places=6)

    def test_try_wang_numbers(self):
        # first from recurrent
        wang_numbers = config_with_weak_synapses.with_property(w_ampa=0.05 * nS,
                                                               w_gaba=1.3 * nS,
                                                               g_nmda_max=0.165 * nS)

        ampa_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(ampa_spike_times=np.array([50])), plot_title="Compute ΔV for 1 AMPA presynaptic spike at 50 ms. "r"$R_{\mathrm{in}}=$"" 50 MΩ")
        self.assertAlmostEqual(0.01046456134456264, ampa_dv_result.ampa_spike_delta_v(), places=6)

        gaba_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(gaba_spike_times=np.array([50])), plot_title="Compute ΔV for 1 GABA presynaptic spike at 50 ms "r"$R_{\mathrm{in}}=$"" 50 MΩ")
        self.assertAlmostEqual(-0.12998195886417818, gaba_dv_result.gaba_spike_delta_v(), places=6)

        nmda_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(nmda_spike_times=np.array([20]), model=WANG_MODEL_FOR_FULL_NMDA_INPUT),  plot_title="Compute ΔV for 1 NMDA presynaptic spike at 20 ms "r"$R_{\mathrm{in}}=$"" 50 MΩ")
        self.assertAlmostEqual(0.21571430433701266, nmda_dv_result.nmda_spike_delta_v(), places=6)

        ampa_external_dv = WangSimulation.run_and_plot(
            wang_numbers.with_property(ampa_spike_times=np.array([50]), w_ampa = 2.1 * nS),  plot_title="Compute ΔV for 1 external AMPA presynaptic spike at 50 ms "r"$R_{\mathrm{in}}=$"" 50 MΩ")
        self.assertAlmostEqual(0.4378979958376874, ampa_external_dv.ampa_spike_delta_v(), places=6)

        wang_numbers = wang_numbers.with_property(g_L = 25 * nS)

        ampa_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(ampa_spike_times=np.array([50])),  plot_title="Compute ΔV for 1 AMPA presynaptic spike at 50 ms "r"$R_{\mathrm{in}}=$"" 40 MΩ")

        gaba_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(gaba_spike_times=np.array([50])),  plot_title="Compute ΔV for 1 GABA presynaptic spike at 50 ms "r"$R_{\mathrm{in}}=$"" 40 MΩ")

        nmda_dv_result = WangSimulation.run_and_plot(
            wang_numbers.with_property(nmda_spike_times=np.array([20]), model=WANG_MODEL_FOR_FULL_NMDA_INPUT),  plot_title="Compute ΔV for 1 NMDA presynaptic spike at 20 ms "r"$R_{\mathrm{in}}=$"" 40 MΩ")

        ampa_external_dv = WangSimulation.run_and_plot(
            wang_numbers.with_property(ampa_spike_times=np.array([50]), w_ampa=2.1 * nS),  plot_title="Compute ΔV for 1 AMPA presynaptic spike at 50 ms "r"$R_{\mathrm{in}}=$"" 50 MΩ")
        print("Results for 25 nS")
        print(f"ampa dv {ampa_dv_result.ampa_spike_delta_v()}")
        print(f"gaba dv {gaba_dv_result.gaba_spike_delta_v()}")
        print(f"nmda dv {nmda_dv_result.nmda_spike_delta_v()}")
        print(f"ampa external dv {ampa_external_dv.ampa_spike_delta_v()}")

        self.assertAlmostEqual(0.0100969534049824, ampa_dv_result.ampa_spike_delta_v(), places=6)
        self.assertAlmostEqual(-0.1225055939641635, gaba_dv_result.gaba_spike_delta_v(), places=6)
        self.assertAlmostEqual(0.18323729557288004, nmda_dv_result.nmda_spike_delta_v(), places=6)
        self.assertAlmostEqual(0.4225476489631461, ampa_external_dv.ampa_spike_delta_v(), places=6)

    def test_find_config_for_medium_synapses(self):
        config = config_with_weak_synapses

        ampa_config = config.with_property(ampa_spike_times=np.array([50]))
        # the target is 2 mV for one presynaptic input
        find_dv_of_ampa = lambda x: WangSimulation.run(ampa_config.with_property(w_ampa=x)).ampa_spike_delta_v()

        lower, upper = binary_search_for_target_value(lower_value=3 * nS, upper_value=10 * nS, func=find_dv_of_ampa,
                                                      target_result=2,
                                                      precision=1E-6 * nS)

        print(f"AMPA conductance for 2 mV of spike: {lower}")
        self.assertAlmostEqual(9.7228533, lower / nS)

        gaba_config = config.with_property(gaba_spike_times=np.array([50]))
        # the target is 2 mV for one presynaptic input
        find_dv_of_gaba = lambda x: WangSimulation.run(gaba_config.with_property(w_gaba=x)).gaba_spike_delta_v()

        lower, upper = binary_search_for_target_value(lower_value=30 * nS, upper_value=10 * nS, func=find_dv_of_gaba,
                                                      target_result=-2,
                                                      precision=1E-6 * nS)
        self.assertAlmostEqual(21.6878134, lower / nS)

        print(f"GABA conductance for 2 mV of spike {lower}")








if __name__ == '__main__':
    unittest.main()
