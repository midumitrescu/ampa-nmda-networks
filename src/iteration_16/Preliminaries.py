import unittest

import numpy as np

from brian2 import ms, Hz, nS, mV

from BinarySeach import binary_search_for_target_value
from iteration_16.model import ConductanceDiffusionSimulationConfig, WANG_MODEL_FOR_FULL_NMDA_INPUT, WANG_MODEL
from iteration_16.simulation import WangSimulation, plot


class CalibratePresynapticDVs(unittest.TestCase):
    def test_ampa_dv_is_0_5_mv_at_soma(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            ampa_spike_times=np.array([50]))

        find_dv_of_ampa = lambda x: WangSimulation.run(config.with_property(w_ampa = x)).ampa_spike_delta_v()

        lower, upper = binary_search_for_target_value(lower_value=0.5 * nS, upper_value= 3 * nS, func=find_dv_of_ampa, target_result=0.5,
                                                      precision=1E-6 * nS)


        target_config = config.with_property(w_ampa = lower)
        dv_result = WangSimulation.run(target_config)
        plot(dv_result)

        self.assertAlmostEqual(1.84, lower / nS, places=1)
        self.assertAlmostEqual(0.5, dv_result.ampa_spike_delta_v())

    def test_gaba_dv_is_0_5_mv_at_soma(self):
        config = ConductanceDiffusionSimulationConfig(
            simulation_time=100 * ms,
            gaba_spike_times=np.array([50]))

        find_dv_of_gaba = lambda x: WangSimulation.run(config.with_property(w_gaba = x)).gaba_spike_delta_v()

        lower, _ = binary_search_for_target_value(lower_value=10 * nS, upper_value= 0.5 * nS, func=find_dv_of_gaba, target_result=-0.5,
                                                      precision=1E-6 * nS)


        target_config = config.with_property(w_gaba = lower)
        dv_result = WangSimulation.run(target_config)
        plot(dv_result)

        self.assertAlmostEqual(4.131385296583174, lower / nS)
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

        self.assertAlmostEqual(3.55138904, lower/nS)
        self.assertAlmostEqual(5, dv_result.nmda_spike_delta_v(), places=6)

        dv_result_with_sigma_v = WangSimulation.run_and_plot(config_with_calibrated_nmda.with_property(model=WANG_MODEL))
        self.assertAlmostEqual(0.23880459719786984, dv_result_with_sigma_v.nmda_spike_delta_v(), places=6)



if __name__ == '__main__':
    unittest.main()
